import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class EnglishModule(nn.Module):
    def __init__(self,word2vec,embedding_trainable=False):
        super(myModule,self).__init__()
        self.embed_dim = word2vec.shape[1]
        self.embed_num = word2vec.shape[0]
        self.hidden_size = 128
        self.hidden_layer = 3
        self.class_num = 2
        self.wv = word2vec
        self.embedding_trainable = embedding_trainable
        self.dropout = 0.25
        self.embedding = nn.Embedding(self.embed_num,self.embed_dim)
        self.biGRU = nn.GRU(self.embed_dim,self.hidden_size,self.hidden_layer,dropout=self.dropout,bidirectional=True)
        self.softmax_layer = nn.Sequential(
            nn.Linear(16+2*self.hidden_size+46,self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size,self.class_num),
        )
        self.position_layer = nn.Linear(46,self.embed_dim)
        self.conv1 = nn.Conv2d(1,32,kernel_size=(2,1),stride=(1,1))
        self.fc = nn.Linear(32,16)
        self.init_weight()
    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find('weight')!=-1:
                nn.init.xavier_normal(param.data)
            elif name.find('bias')!=-1:
                param.data.uniform_(-0.1,-0.1)
        self.embedding.weight.data.copy_(torch.from_numpy(self.wv))
        self.embedding.weight.requires_grad = self.embedding_trainable


    def init_hidden(self,batch_size):
        hidden = Variable(torch.zeros(self.hidden_layer*2,batch_size,self.hidden_size).cuda())                
        return hidden

    def forward(self,text,emo,pos,init_state):
        text_inputs = self.embedding(text).permute(1,0,2)
        outputs,_ = self.biGRU(text_inputs,init_state)
        last_outputs = outputs[-1].squeeze()
        bs,dims = last_outputs.size()#[batch,seq,2*hidden]

        pos_repre = self.position_layer(pos)  #[batch,39]->[batch,embed]

        emo_repre = Variable(torch.FloatTensor(bs,self.embed_dim).cuda())
        for batch_i,em in enumerate(emo):
            emo_inputs = Variable(torch.LongTensor(em).cuda())
            emo_embedding = self.embedding(emo_inputs)
            emo_embedding = emo_embedding.mean(dim=0)
            emo_repre[batch_i] = emo_embedding  #[batch,embed]

        #print(emo_repre.size())

        emo_pos = pos_repre*emo_repre  #[batch,embedding]
        emo_pos = emo_pos.unsqueeze(1)
        emo_pos = emo_pos.unsqueeze(3)
        e_p = F.max_pool2d(F.leaky_relu(self.conv1(emo_pos)),(self.embed_dim-2+1,1)).squeeze()
   
        e_p = self.fc(e_p)  #batch,16
        classification = torch.cat([e_p,last_outputs],1)
        class_p = torch.cat([classification,pos],1)
        logits = self.softmax_layer(class_p)
        probs = F.softmax(logits,1)
        return logits,probs


    
class ChineseModule(nn.Module):
    def __init__(self,word2vec,embedding_trainable=False):
        super(myModule,self).__init__()
        self.embed_dim = word2vec.shape[1]
        self.embed_num = word2vec.shape[0]
        self.hidden_size = 128
        self.hidden_layer = 3
        self.class_num = 2
        self.wv = word2vec
        self.embedding_trainable = embedding_trainable
        self.dropout = 0.25
        self.embedding = nn.Embedding(self.embed_num,self.embed_dim)
        self.biGRU = nn.GRU(self.embed_dim,self.hidden_size,self.hidden_layer,dropout=self.dropout,bidirectional=True)
        self.softmax_layer = nn.Sequential(
            nn.Linear(16+2*self.hidden_size+46,self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size,self.class_num),
        )
        self.position_layer = nn.Linear(46,self.embed_dim)
        self.conv1 = nn.Conv2d(1,32,kernel_size=(2,1),stride=(1,1))
        self.fc = nn.Linear(32,16)
        self.init_weight()
    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find('weight')!=-1:
                nn.init.xavier_normal(param.data)
            elif name.find('bias')!=-1:
                param.data.uniform_(-0.1,-0.1)
        self.embedding.weight.data.copy_(torch.from_numpy(self.wv))
        self.embedding.weight.requires_grad = self.embedding_trainable


    def init_hidden(self,batch_size):
        hidden = Variable(torch.zeros(self.hidden_layer*2,batch_size,self.hidden_size).cuda())
                  
        return hidden

    def forward(self,text,emo,pos,init_state):
        text_inputs = self.embedding(text).permute(1,0,2)
        outputs,_ = self.biGRU(text_inputs,init_state)
        last_outputs = outputs[-1].squeeze()
        bs,dims = last_outputs.size()#[batch,seq,2*hidden]

        pos_repre = self.position_layer(pos)  #[batch,39]->[batch,embed]

        emo_repre = Variable(torch.FloatTensor(bs,self.embed_dim).cuda())
        for batch_i,em in enumerate(emo):
            emo_inputs = Variable(torch.LongTensor(em).cuda())
            emo_embedding = self.embedding(emo_inputs)
            emo_embedding = emo_embedding.mean(dim=0)
            emo_repre[batch_i] = emo_embedding  #[batch,embed]

        #print(emo_repre.size())

        emo_pos = pos_repre*emo_repre  #[batch,embedding]
        emo_pos = emo_pos.unsqueeze(1)
        emo_pos = emo_pos.unsqueeze(3)
        e_p = F.max_pool2d(F.leaky_relu(self.conv1(emo_pos)),(self.embed_dim-2+1,1)).squeeze()
   
        e_p = self.fc(e_p)  #batch,16
        classification = torch.cat([e_p,last_outputs],1)
        class_p = torch.cat([classification,pos],1)
        logits = self.softmax_layer(class_p)
        probs = F.softmax(logits,1)
        return logits,probs









