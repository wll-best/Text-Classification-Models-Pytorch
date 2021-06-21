# train.py

from utils import *
from model import *
from config import Config
import sys
import torch.optim as optim
from torch import nn
import torch
import random
import os

if __name__=='__main__':
    config = Config()
    w2v_file = sys.argv[1]  #0是.py
    train_file = sys.argv[2]#
    test_file = sys.argv[3]#
    dev_file = sys.argv[4]#
    seed=42
    #w2v_file = '../data/glove.840B.300d.txt'
    #为了复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    
    dataset = Dataset(config)
    dataset.load_data(w2v_file, train_file, test_file, dev_file)#
    
    # Create Model with specified optimizer and loss function
    ##############################################################
    #model = BiLSTM(config, len(dataset.vocab), dataset.word_embeddings)
    model = BiLSTM(config, len(dataset.vocab), dataset.word_embeddings,len(dataset.pos_vocab))
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)#SGD
    #NLLLoss = nn.NLLLoss()
    CELoss = nn.CrossEntropyLoss()
    model.add_optimizer(optimizer)
    #model.add_loss_op(NLLLoss)
    model.add_loss_op(CELoss)
    ##############################################################
    
    train_losses = []
    val_accuracies = []

    num_model =0
    num_bestacc = 0
    for ep in range(config.max_epochs):
        print ("Epoch: {}".format(ep))
        train_loss,val_accuracy= model.run_epoch(dataset.train_iterator, dataset.val_iterator, ep)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    # print('train_evaluate_model---')4974
    _,train_acc, _ = evaluate_model(model, dataset.train_iterator)
    # print('val_evaluate_model---')1658
    _,val_acc, _ = evaluate_model(model, dataset.val_iterator)
    # print('test_evaluate_model---')1658
    test_acc, macro_f1, all_preds, all_labels = evaluate_model_te(model,dataset.test_iterator)

   # 获取测试集的文本
    text_li = []
    with open(test_file, 'r', encoding='utf-8') as fi:
        next(fi)
        rowes = csv.reader(fi, delimiter='\t')
        for row in rowes:
            text = row[0]
            text_li.append(text)

    # dataframe保存带标签的预测文件ntest_label.tsv,格式：id,text,label,predict_label
    df = pd.DataFrame(columns=['text', 'label', 'predict_label'])
    df['text'] = text_li
    df['predict_label'] = all_preds
    df['label'] = all_labels
    df.to_csv('../data/sem/ntest_bilstm_label.tsv', sep='\t')

    print('Final Training Accuracy: {:.4f}'.format(train_acc))
    print('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print('Final Test Accuracy: {:.4f}'.format(test_acc))
    print('Final Test macro-f1: {:.4f}'.format(macro_f1))
    # print('Final Test micro-f1: {:.4f}'.format(micro_f1))
    # print('Final Test wei-f1: {:.4f}'.format(weighted_f1))
