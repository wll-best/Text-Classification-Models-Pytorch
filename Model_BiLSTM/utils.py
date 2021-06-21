# utils.py

import torch
from torchtext import data
from torchtext.vocab import Vectors
import spacy
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
import csv
from torch import nn

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}
        self.pos_vocab=[]
    
    # def parse_label(self, label):
    #     '''
    #     Get the actual labels from label string
    #     Input:
    #         label (string) : labels of the form '__label__2'
    #     Returns:
    #         label (int) : integer value corresponding to label string
    #     '''
    #     return int(label.strip()[-1])

    def get_pandas_df(self, filename):
        '''
        Load the data into Pandas.DataFrame object
        This will be used to convert data to torchtext object
        '''
        text_li=[]
        label_li=[]
        with open(filename, 'r', encoding='utf-8') as fi:
            next(fi)
            rowes = csv.reader(fi, delimiter='\t')
            for row in rowes:
                text = row[0]
                text_li.append(text)
                label = row[1]
                label_li.append(label)
        full_df = pd.DataFrame({"text": text_li, "label": label_li})
        return full_df

    # 新增函数
    # df.values.tolist()形如[[text1, label1], [text2, label2], ..., [textn, labeln]]
    # 向train_df.values.tolist()增加了一个特征，也就是复制了一遍text。
    # 返回值read_data形如[[text1, text1, label1], [text2, text2, label2], ..., [textn, textn, labeln]]
    def df_process(self,df):
        df_va=df.values.tolist()
        return [[x[0],x[0],x[1]] for x in df_va]
        #return [[[text, text, label] for text, label in example] for example in df.values.tolist()]

    def load_data(self, w2v_file, train_file, test_file, val_file=None):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data
        
        Inputs:
            w2v_file (String): absolute path to file containing word embeddings (GloVe/Word2Vec)
            train_file (String): absolute path to training file
            test_file (String): absolute path to test file
            val_file (String): absolute path to validation file
        '''

        NLP = spacy.load('en')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
        pos_generator = lambda sent: [x.pos_ for x in NLP(sent) if x.text != " "]# 新增函数，获取词性特征

        # Creating Field for data
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        # 新增POS Field，这里用pos_generator代替了tokenizer函数对原文进行处理，从而得到词性特征。
        POS = data.Field(sequential=True, tokenize=pos_generator, lower=True,
                         fix_length=self.config.max_sen_len)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text",TEXT),("pos",POS),("label",LABEL)]#新增POS Field
        
        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file)
        train_read_data = self.df_process(train_df)  # 给数据增加一个特征，也就是复制了一遍text，它在经过pos_generator处理后变成词性特征
        #train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_examples = [data.Example.fromlist(i, datafields) for i in train_read_data]
        train_data = data.Dataset(train_examples, datafields)
        
        test_df = self.get_pandas_df(test_file)
        test_read_data = self.df_process(test_df)
        test_examples = [data.Example.fromlist(i, datafields) for i in test_read_data]
        test_data = data.Dataset(test_examples, datafields)
        
        # If validation file exists, load it. Otherwise get validation data from training data
        if val_file:
            val_df = self.get_pandas_df(val_file)
            val_read_data = self.df_process(val_df)
            val_examples = [data.Example.fromlist(i, datafields) for i in val_read_data]
            val_data = data.Dataset(val_examples, datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.8)
        
        #TEXT.build_vocab(train_data, vectors=Vectors(w2v_file))
        TEXT.build_vocab(train_data, val_data, test_data, vectors=Vectors(w2v_file))#

        self.word_embeddings = TEXT.vocab.vectors
        self.vocab = TEXT.vocab

        POS.build_vocab(train_data)                                     #不用vectors初始化
        self.pos_vocab = POS.vocab                                     #在__init__里也要加上self.POS_vocab

        #print(self.pos_vocab.itos)#统计pos维度---20
        # ['<unk>', '<pad>', 'noun', 'punct', 'det', 'adj', 'verb', 'aux', 'adp', 'adv', 'pron', 'cconj', 'propn', 'part', 'sconj', 'num', 'sym', 'intj', 'x', 'space']

        #self.pos_embeddings=
        
        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)
        
        self.val_iterator, self.test_iterator = data.BucketIterator.splits(
            (val_data, test_data),
            batch_size=self.config.batch_size,
            sort=False,#这里为了打印label.tsv不改变原始顺序
            repeat=False,
            shuffle=False)
        #                        #sort_key=lambda x: len(x.text),
        print ("Loaded {} training examples".format(len(train_data)))
        print ("Loaded {} test examples".format(len(test_data)))
        print ("Loaded {} validation examples".format(len(val_data)))


def evaluate_model(model, iterator):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()  # 加了torch
    criterion = criterion.to(device)

    all_preds = []
    all_y = []
    epoch_loss = 0
    for idx,batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text.cuda()
            pos = batch.pos.cuda()
        else:
            x = batch.text
            pos = batch.pos
        y_pred = model(x,pos)

        loss = criterion(y_pred.view(-1, 5), (batch.label - 1).to(device).view(-1))  # 5是标签种类数

        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())

        epoch_loss += loss.mean().item()

    #print('预测样本数：'+str(len(all_y)))
    accscore = accuracy_score(all_y, np.array(all_preds).flatten())
    macro_f1=f1_score(all_y, np.array(all_preds).flatten(), average='macro')
    return epoch_loss / len(iterator),accscore,macro_f1

def evaluate_model_te(model, iterator):#有时间得到tensoboard图

    all_preds = []
    all_y = []
    all_logits=[]
    for idx,batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text.cuda()
            pos = batch.pos.cuda()
        else:
            x = batch.text
            pos = batch.pos
        y_pred = model(x,pos)
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1#[0]是最大值，[1]是最大值的索引
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
        norm=nn.Softmax(dim=1)#按最后一个维度
        all_logits=np.append(all_logits,norm(y_pred.cpu().data))#每种分类的可能性数组
    np.savetxt('../data/sem/all_logits_bilstm.txt', all_logits.reshape(-1, 5))

    accuracy = accuracy_score(all_y, np.array(all_preds).flatten())
    #micro_f1 = f1_score(all_y, np.array(all_preds).flatten(), average='micro')
    #weighted_f1=f1_score(all_y, np.array(all_preds).flatten(), average='weighted')
    macro_f1 = f1_score(all_y, np.array(all_preds).flatten(), average='macro')
    return accuracy, macro_f1, np.array(all_preds).flatten(), all_y

