# utils.py

import torch
from torchtext import data
from torchtext.vocab import Vectors
import spacy
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
import csv

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}
    
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
                #label_li.append(str(int(label)-1))
                label_li.append(label)
                #print(label_li)
        full_df = pd.DataFrame({"text": text_li, "label": label_li})
        return full_df
    
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
        
        # Creating Field for data
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text",TEXT),("label",LABEL)]
        
        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file)
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)
        
        test_df = self.get_pandas_df(test_file)
        test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, datafields)
        
        # If validation file exists, load it. Otherwise get validation data from training data
        if val_file:
            val_df = self.get_pandas_df(val_file)
            val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_examples, datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.8)

        #TEXT.build_vocab(train_data, vectors=Vectors(w2v_file))
        TEXT.build_vocab(train_data, val_data, test_data, vectors=Vectors(w2v_file))#

        self.word_embeddings = TEXT.vocab.vectors
        self.vocab = TEXT.vocab

        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,# 每个batch内的数据按照sork_key降序排列，为pack_padded_sequence做准备
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
    all_preds = []
    all_y = []
    for idx,batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text.cuda()
        else:
            x = batch.text
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    #print('预测样本数：'+str(len(all_y)))
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    macro_f1=f1_score(all_y, np.array(all_preds).flatten(), average='macro')
    return score,macro_f1

def evaluate_model_te(model, iterator):#有时间得到logits(roc曲线), tensoboard图
    all_preds = []
    all_y = []
    all_logits=[]
    for idx,batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text.cuda()
        else:
            x = batch.text
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
        all_logits=np.append(all_logits,y_pred.cpu().data)#每种分类的可能性数组
    np.savetxt('../data/sem/all_logits_cnn.txt', all_logits.reshape(-1, 5))

    accuracy = accuracy_score(all_y, np.array(all_preds).flatten())
    micro_f1 = f1_score(all_y, np.array(all_preds).flatten(), average='micro')
    weighted_f1=f1_score(all_y, np.array(all_preds).flatten(), average='weighted')
    macro_f1 = f1_score(all_y, np.array(all_preds).flatten(), average='macro')
    return accuracy, macro_f1, np.array(all_preds).flatten(), all_y,micro_f1,weighted_f1

