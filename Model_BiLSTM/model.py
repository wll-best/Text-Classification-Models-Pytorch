# model.py

import torch
from torch import nn
import numpy as np
from utils import *

class BiLSTM(nn.Module):
    def __init__(self, config, vocab_size, word_embeddings, pos_vocab_size):#, pos_embeddings
        super(BiLSTM, self).__init__()
        self.config = config
        
        # Embedding Layer来解决 embedding lookup 问题
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)

        #加词性向量
        self.posembeddings = nn.Embedding(pos_vocab_size, self.config.pos_embed_size)
        #self.posembeddings.weight = nn.Parameter(pos_embeddings, requires_grad=False)

        #第一项后加pos
        self.lstm = nn.LSTM(input_size = self.config.embed_size+self.config.pos_embed_size,
                            hidden_size = self.config.hidden_size,
                            num_layers = self.config.hidden_layers,
                            dropout = self.config.dropout_keep,
                            bidirectional = self.config.bidirectional)
        
        self.dropout = nn.Dropout(self.config.dropout_keep)
        
        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.hidden_size * self.config.hidden_layers * (1+self.config.bidirectional),
            self.config.output_size
        )
        
        # Softmax non-linearity
        self.softmax = nn.Softmax()
        
    def forward(self, x, pos):#新增传参pos
        # x.shape = (max_sen_len, batch_size)
        embedded_sent = self.embeddings(x)
        # embedded_sent.shape = (max_sen_len=20, batch_size=64,embed_size=300)

        # 加词性向量
        embedded_pos=self.posembeddings(pos)
        #拼接--词向量与词性向量--torch.cat((a,a),1)
        embedded_all=torch.cat((embedded_sent,embedded_pos),2)

        #lstm_out, (h_n,c_n) = self.lstm(embedded_sent)
        lstm_out, (h_n, c_n) = self.lstm(embedded_all)#维度有变
        final_feature_map = self.dropout(h_n) # shape=(num_layers * num_directions, 64, hidden_size)
        
        # Convert input to (64, hidden_size * hidden_layers * num_directions) for linear layer
        final_feature_map = torch.cat([final_feature_map[i,:,:] for i in range(final_feature_map.shape[0])], dim=1)
        final_out = self.fc(final_feature_map)
        return self.softmax(final_out)
    
    def add_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def add_loss_op(self, loss_op):
        self.loss_op = loss_op
    
    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2
                
    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
        
        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config.max_epochs/3)) or (epoch == int(2*self.config.max_epochs/3)):
            self.reduce_lr()
            
        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text.cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
                pos=batch.pos.cuda()#新增pos
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
                pos = batch.pos#新增pos
            y_pred = self.__call__(x,pos)#新增pos
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
    
            if i % 50 == 0:#原来100
                print("Iter: {}".format(i+1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []
                
                # Evalute Accuracy on validation set
                val_accuracy,_ = evaluate_model(self, val_iterator)
                print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                self.train()
                
        return train_losses, val_accuracies