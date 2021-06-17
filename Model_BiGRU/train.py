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
    model = BiGRU(config, len(dataset.vocab), dataset.word_embeddings,len(dataset.pos_vocab))
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
    
    for i in range(config.max_epochs):
        print ("Epoch: {}".format(i))
        train_loss,val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    train_acc,_ = evaluate_model(model, dataset.train_iterator)
    val_acc,_ = evaluate_model(model, dataset.val_iterator)
    test_acc,macro_f1 = evaluate_model(model, dataset.test_iterator)

    print ('Final Training Accuracy: {:.4f}'.format(train_acc))
    print ('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print ('Final Test Accuracy: {:.4f}'.format(test_acc))
    print('Final Test macro-f1: {:.4f}'.format(macro_f1))