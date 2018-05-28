import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
import os

class aspect_model(nn.Module):
    def __init__(self, word2vec, is_bi = 'True', embedding_trainable=False, loop_num=3):
        super(aspect_model, self).__init__()
        self.word2vec = word2vec
        self.is_bi = is_bi
        self.embedding_trainable = embedding_trainable
        self.loop_num = loop_num
        self.layers = 2
        self.num_label = 2
        self.vocab_size = word2vec.shape[0]
        self.embedding_dim = word2vec.shape[1]
        self.hidden_units =300
        self.hidden_size =128
        self.dropout = 0.25
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        if self.is_bi:
            self.is_bi_units = self.hidden_units * 2
        else:
            self.is_bi_units = self.hidden_units
        self.cell_position = nn.GRU(self.embedding_dim, self.hidden_units, self.layers, 
                               dropout = self.dropout, bidirectional=self.is_bi)
        self.cell_aspect = nn.GRU(self.embedding_dim, self.hidden_units, self.layers, 
                               dropout = self.dropout, bidirectional=self.is_bi)
        self.position_layer = nn.Linear(102, self.hidden_units*2)
        self.aspect2hidden = nn.Linear(self.embedding_dim, self.hidden_units*2)
        self.softmax_layer = nn.Sequential(
                nn.Linear(self.embedding_dim*2, self.hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.num_label)
            )
        self.init_weight()
    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find("weight") != -1:
                nn.init.xavier_normal(param.data)
            elif name.find("bias") != -1:
                param.data.uniform_(-0.1, 0.1)
            else:
                continue
        self.embedding.weight.data.copy_(torch.from_numpy(self.word2vec))
        self.embedding.weight.requires_grad = self.embedding_trainable
        
    def init_hidden(self, batch_size):
        if self.is_bi:
            hidden = Variable(torch.zeros(self.layers*2, batch_size, self.hidden_units).cuda())
        else:
            hidden = Variable(torch.zeros(self.layers, batch_size, self.hidden_units).cuda())
        return hidden
    def forward(self, text, aspects, pos, init_state):
        text_embedding = self.embedding(text)
        rnn_inputs = text_embedding.permute(1, 0, 2)   #seq_len * batch_size * embedding_dim
        bs, seq_len = text.size()
        pos_repre = self.position_layer(pos)
        pos_repre = pos_repre.unsqueeze(-1)
        
        aspect_repre = Variable(torch.FloatTensor(bs, self.embedding_dim).cuda())
        for batch_i, aspect in enumerate(aspects):
            aspect_inputs = Variable(torch.LongTensor(aspect).cuda())
            aspect_embedding = self.embedding(aspect_inputs)
            aspect_embedding = aspect_embedding.mean(dim=0)
            aspect_repre[batch_i] = aspect_embedding
        aspect2hidden_repre = self.aspect2hidden(aspect_repre).unsqueeze(-1)
        
        for loop_i in range(self.loop_num):
            outputs, _ = self.cell_position(rnn_inputs, init_state) #seq_len * batch_size * hidden_state
            outputs = outputs.permute(1, 0, 2).contiguous()      #batch_size * seq_len * hidden_state
            position_probs = torch.bmm(outputs, pos_repre)
            position_probs = position_probs.expand(bs, seq_len, self.embedding_dim)
            tmp_inputs = text_embedding * position_probs
            rnn_inputs = tmp_inputs.permute(1, 0, 2)
        final_pos_repre = tmp_inputs.sum(dim=1).squeeze(dim=1)
        
        rnn_inputs = text_embedding.permute(1, 0, 2)
        for loop_i in range(self.loop_num):
            outputs, _ = self.cell_aspect(rnn_inputs, init_state) #seq_len * batch_size * hidden_state
            outputs = outputs.permute(1, 0, 2).contiguous()      #batch_size * seq_len * hidden_state
            aspect_probs = torch.bmm(outputs, aspect2hidden_repre)
            aspect_probs = aspect_probs.expand(bs, seq_len, self.embedding_dim)
            tmp_inputs = text_embedding * aspect_probs
            rnn_inputs = tmp_inputs.permute(1, 0, 2)
        final_aspect_repre = tmp_inputs.sum(dim=1).squeeze(dim=1)
        classicication_repre = torch.cat([final_pos_repre, final_aspect_repre], 1)
        logits = self.softmax_layer(classicication_repre)
        probs = F.softmax(logits, 1)
        return logits, probs