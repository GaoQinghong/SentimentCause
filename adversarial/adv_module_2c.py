#encoding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function

class featureExtractor(nn.Module):
    def __init__(self,word2vecE,word2vecC,embedding_trainable=False):
        super(featureExtractor,self).__init__()
        self.embed_dim = word2vecE.shape[1]
        self.embed_numE = word2vecE.shape[0]
        self.embed_numC = word2vecC.shape[0]
        self.wvE = word2vecE
        self.wvC = word2vecC
        self.embedding_trainable = embedding_trainable
        self.dropout = 0.1
        self.hidden_units1 = 100
        self.hidden_units3 = 60
        self.hidden_units2 = 30
        self.embeddingE = nn.Embedding(self.embed_numE, self.embed_dim)
        self.embeddingC = nn.Embedding(self.embed_numC, self.embed_dim)
        self.fullConn_layer = nn.Sequential(
                    nn.Linear(self.embed_dim, self.hidden_units1),
                    nn.LeakyReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_units1, self.hidden_units3),
                    nn.LeakyReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_units3, self.hidden_units2),
                    nn.LeakyReLU(),
                    nn.Dropout(self.dropout)
                )
        self.init_weights()

    def init_weights(self):
        for name,param in self.named_parameters():
            if name.find('weight')!=-1:
                nn.init.xavier_normal(param.data)
            elif name.find('bias')!=-1:
                param.data.uniform_(-0.1,-0.1)
        self.embeddingE.weight.data.copy_(torch.from_numpy(self.wvE))
        self.embeddingC.weight.data.copy_(torch.from_numpy(self.wvC))
        self.embeddingE.weight.requires_grad = self.embedding_trainable
        self.embeddingC.weight.requires_grad = self.embedding_trainable
    def forward(self,textE=None,emoE=None,textC=None,emoC=None,textEC=None,emoEC=None,textCE=None,emoCE=None,train=None):
        if train=='False':
            bsC ,sC = textC.shape[0],textC.shape[1]
        
            textC_inputs = self.embeddingC(textC)         

           # emoE_inputs = torch.sum(self.embeddingE(emoE),1)
            emoC_inputs = torch.sum(self.embeddingC(emoC),1)

            #textE_inputs = textE_inputs.view(-1,self.embed_dim)
            textC_inputs = textC_inputs.view(-1,self.embed_dim)

            #outputsE = self.fullConn_layer(textE_inputs)
            outputsC = self.fullConn_layer(textC_inputs)

            #emoE_outputs = self.fullConn_layer(emoE_inputs)
            emoC_outputs = self.fullConn_layer(emoC_inputs)

            return outputsC.view(bsC,sC,self.hidden_units2),emoC_outputs.view(bsC,1,self.hidden_units2)
        if train=='True':
         
            bsE,sE = textE.shape[0],textE.shape[1]
            bsC ,sC = textC.shape[0],textC.shape[1]
            bsEC,sEC = textEC.shape[0],textEC.shape[1]
            bsCE ,sCE = textCE.shape[0],textCE.shape[1]

            textE_inputs = self.embeddingE(textE)
            textC_inputs = self.embeddingC(textC)   
            textEC_inputs = self.embeddingE(textEC)
            textCE_inputs = self.embeddingC(textCE) 
        
            
            emoE_inputs = torch.sum(self.embeddingE(emoE),1)
            emoC_inputs = torch.sum(self.embeddingC(emoC),1)
            emoEC_inputs = torch.sum(self.embeddingE(emoEC),1)
            emoCE_inputs = torch.sum(self.embeddingC(emoCE),1)
            
            textE_inputs = textE_inputs.view(-1,self.embed_dim)
            textC_inputs = textC_inputs.view(-1,self.embed_dim)
            textEC_inputs = textEC_inputs.view(-1,self.embed_dim)
            textCE_inputs = textCE_inputs.view(-1,self.embed_dim)
    
            outputsE = self.fullConn_layer(textE_inputs)  
            outputsC = self.fullConn_layer(textC_inputs)
            outputsEC = self.fullConn_layer(textEC_inputs)
            outputsCE = self.fullConn_layer(textCE_inputs)
            
            emoE_outputs = self.fullConn_layer(emoE_inputs)
           
            emoC_outputs = self.fullConn_layer(emoC_inputs)
      
            emoEC_outputs = self.fullConn_layer(emoEC_inputs)
           
            emoCE_outputs = self.fullConn_layer(emoCE_inputs)
           

            return outputsE.view(bsE,-1,self.hidden_units2),outputsC.view(bsC,-1,self.hidden_units2),outputsEC.view(bsEC,-1,self.hidden_units2),outputsCE.view(bsCE,-1,self.hidden_units2),emoE_outputs.view(bsE,1,self.hidden_units2),emoC_outputs.view(bsC,1,self.hidden_units2),emoEC_outputs.view(bsEC,1,self.hidden_units2),emoCE_outputs.view(bsCE,1,self.hidden_units2)
            


#     def forward(self,textE,emoE,textC,emoC):
        
#         bsE,sE = textE.shape[0],textE.shape[1]
#         bsC ,sC = textC.shape[0],textC.shape[1]

#         textE_inputs = self.embeddingE(textE)
#         textC_inputs = self.embeddingC(textC)         

#         emoE_inputs = torch.sum(self.embeddingE(emoE),1)
#         emoC_inputs = torch.sum(self.embeddingC(emoC),1)

#         textE_inputs = textE_inputs.view(-1,self.embed_dim)
#         textC_inputs = textC_inputs.view(-1,self.embed_dim)

#         outputsE = self.fullConn_layer(textE_inputs)
#         outputsC = self.fullConn_layer(textC_inputs)

#         emoE_outputs = self.fullConn_layer(emoE_inputs)
#         emoC_outputs = self.fullConn_layer(emoC_inputs)

#         return outputsE.view(bsE,sE,self.hidden_units2),outputsC.view(bsC,sC,self.hidden_units2),emoE_outputs.view(bsE,1,self.hidden_units2),emoC_outputs.view(bsC,1,self.hidden_units2)




class classifier(nn.Module):
    def __init__(self,is_bi = 'True', loop_num=3):
        super(classifier, self).__init__()
        self.is_bi = is_bi
        self.loop_num = loop_num
        self.layers = 2
        self.num_label = 2
        self.embedding_dim = 30
        self.hidden_units =300
        self.hidden_size =128
        self.dropout = 0.25
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
        
    def init_hidden(self, batch_size):
        if self.is_bi:
            hidden = Variable(torch.zeros(self.layers*2, batch_size, self.hidden_units).cuda())
        else:
            hidden = Variable(torch.zeros(self.layers, batch_size, self.hidden_units).cuda())
        return hidden
    def forward(self, textE=None, emoE=None, posE=None, init_state=None,textC=None, emoC=None, posC=None,train=None):
        if train=='True':
            text_inputs = torch.cat([textE,textC],0)
            #text_inputs = text_inputs#.permute(1,0,2)
            pos = torch.cat([posE,posC],0)
            emo = torch.cat([torch.sum(emoE,1),torch.sum(emoC,1)],0)
            
        if train=='False':
            
            text_inputs = textC#.permute(1,0,2)
            pos = posC
            emo = torch.sum(emoC,1)
            
        rnn_inputs = text_inputs.permute(1, 0, 2)   #seq_len * batch_size * embedding_dim
        bs, seq_len, _ = text_inputs.size()
        pos_repre = self.position_layer(pos)
        pos_repre = pos_repre.unsqueeze(-1)
        
        aspect_repre = emo
        aspect2hidden_repre = self.aspect2hidden(aspect_repre).unsqueeze(-1)
        
        for loop_i in range(self.loop_num):
            outputs, _ = self.cell_position(rnn_inputs, init_state) #seq_len * batch_size * hidden_state
            outputs = outputs.permute(1, 0, 2).contiguous()      #batch_size * seq_len * hidden_state
            position_probs = torch.bmm(outputs, pos_repre)
            position_probs = position_probs.expand(bs, seq_len, self.embedding_dim)
            tmp_inputs = text_inputs * position_probs
            rnn_inputs = tmp_inputs.permute(1, 0, 2)
        final_pos_repre = tmp_inputs.sum(dim=1).squeeze(dim=1)
        
        rnn_inputs = text_inputs.permute(1, 0, 2)
        for loop_i in range(self.loop_num):
            outputs, _ = self.cell_aspect(rnn_inputs, init_state) #seq_len * batch_size * hidden_state
            outputs = outputs.permute(1, 0, 2).contiguous()      #batch_size * seq_len * hidden_state
            aspect_probs = torch.bmm(outputs, aspect2hidden_repre)
            aspect_probs = aspect_probs.expand(bs, seq_len, self.embedding_dim)
            tmp_inputs = text_inputs * aspect_probs
            rnn_inputs = tmp_inputs.permute(1, 0, 2)
        final_aspect_repre = tmp_inputs.sum(dim=1).squeeze(dim=1)
        classicication_repre = torch.cat([final_pos_repre, final_aspect_repre], 1)
        logits = self.softmax_layer(classicication_repre)
        probs = F.softmax(logits, 1)
        return logits, probs



# class classifier(nn.Module):
#     def __init__(self):
#         super(classifier,self).__init__()
#         self.embed_dim = 30
#         self.hidden_size = 128
#         self.hidden_layer = 4
#         self.class_num = 2
#         self.dropout = 0.25
#         self.biGRU = nn.GRU(self.embed_dim,self.hidden_size,self.hidden_layer,dropout=self.dropout,bidirectional=True)
#         self.softmax_layer = nn.Sequential(
#             nn.Linear(16+2*self.hidden_size+102,self.hidden_size),
#             nn.LeakyReLU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(self.hidden_size,self.class_num),
#         )
#         self.position_layer = nn.Linear(102,self.embed_dim)
#         self.conv1 = nn.Conv2d(1,32,kernel_size=(2,1),stride=(1,1))
#         self.fc = nn.Linear(32,16)
#         self.init_weights()

#     def init_weights(self):
#         for name, param in self.named_parameters():
#             if name.find('weight')!=-1:
#                 nn.init.xavier_normal(param.data)
#             elif name.find('bias')!=-1:
#                 param.data.uniform_(-0.1,-0.1)

#     def init_hidden(self,batch_size):
      
#         hidden = Variable(torch.zeros(self.hidden_layer*2,batch_size,self.hidden_size).cuda())
#         return hidden

#     def forward(self,textE=None,emoE=None,posE=None,init_state=None,textC=None,emoC=None,posC=None,train=None):

#         if train=='True':
#             text_inputs = torch.cat([textE,textC],0)
#             text_inputs = text_inputs.permute(1,0,2)
#             pos = torch.cat([posE,posC],0)
#             emo = torch.cat([torch.sum(emoE,1),torch.sum(emoC,1)],0)
            
#         if train=='False':
            
#             text_inputs = textC.permute(1,0,2)
#             pos = posC
#             emo = torch.sum(emoC,1)
#         outputs,_ = self.biGRU(text_inputs,init_state)
#         last_outputs = outputs[-1].squeeze()
       
#         bs,dims = last_outputs.size()
        
#         pos_repre = self.position_layer(pos)  #[batch,39]->[batch,embed]
      
#         emo_pos = pos_repre*emo  #[batch,embedding]
#         emo_pos = emo_pos.unsqueeze(1)
#         emo_pos = emo_pos.unsqueeze(3)
#         e_p = self.conv1(emo_pos)
#         e_p = F.leaky_relu(e_p) 
#         e_p = F.max_pool2d(e_p,(self.embed_dim-2+1,1))
#         e_p = e_p.squeeze() 
#         e_p = self.fc(e_p)  #batch,16
     
#         classification = torch.cat([e_p,last_outputs],1)
#         class_p = torch.cat([classification,pos],1)
#         logits = self.softmax_layer(class_p)
#         probs = F.softmax(logits,1)

#         return logits,probs


        
# class classifierEN(nn.Module):
#     def __init__(self):
#         super(classifier,self).__init__()
#         self.embed_dim = 30
#         self.hidden_size = 128
#         self.hidden_layer = 4
#         self.class_num = 2
#         self.dropout = 0.25
#         self.biGRU = nn.GRU(self.embed_dim,self.hidden_size,self.hidden_layer,dropout=self.dropout,bidirectional=True)
#         self.softmax_layer = nn.Sequential(
#             nn.Linear(16+2*self.hidden_size+102,self.hidden_size),
#             nn.LeakyReLU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(self.hidden_size,self.class_num),
#         )
#         self.position_layer = nn.Linear(102,self.embed_dim)
#         self.conv1 = nn.Conv2d(1,32,kernel_size=(2,1),stride=(1,1))
#         self.fc = nn.Linear(32,16)
#         self.init_weights()

#     def init_weights(self):
#         for name, param in self.named_parameters():
#             if name.find('weight')!=-1:
#                 nn.init.xavier_normal(param.data)
#             elif name.find('bias')!=-1:
#                 param.data.uniform_(-0.1,-0.1)

#     def init_hidden(self,batch_size):
      
#         hidden = Variable(torch.zeros(self.hidden_layer*2,batch_size,self.hidden_size).cuda())
#         return hidden

#     def forward(self,textE=None,emoE=None,posE=None,init_state=None,textC=None,emoC=None,posC=None,train=None):

#         if train=='True':
#             text_inputs = torch.cat([textE,textC],0)
#             text_inputs = text_inputs.permute(1,0,2)
#             pos = torch.cat([posE,posC],0)
#             emo = torch.cat([torch.sum(emoE,1),torch.sum(emoC,1)],0)
            
#         if train=='False':
            
#             text_inputs = textC.permute(1,0,2)
#             pos = posC
#             emo = torch.sum(emoC,1)
#         outputs,_ = self.biGRU(text_inputs,init_state)
#         last_outputs = outputs[-1].squeeze()
       
#         bs,dims = last_outputs.size()
        
#         pos_repre = self.position_layer(pos)  #[batch,39]->[batch,embed]
      
#         emo_pos = pos_repre*emo  #[batch,embedding]
#         emo_pos = emo_pos.unsqueeze(1)
#         emo_pos = emo_pos.unsqueeze(3)
#         e_p = self.conv1(emo_pos)
#         e_p = F.leaky_relu(e_p) 
#         e_p = F.max_pool2d(e_p,(self.embed_dim-2+1,1))
#         e_p = e_p.squeeze() 
#         e_p = self.fc(e_p)  #batch,16
     
#         classification = torch.cat([e_p,last_outputs],1)
#         class_p = torch.cat([classification,pos],1)
#         logits = self.softmax_layer(class_p)
#         probs = F.softmax(logits,1)

#         return logits,probs
    
    
# class classifierCH(nn.Module):
#     def __init__(self):
#         super(classifier,self).__init__()
#         self.embed_dim = 30
#         self.hidden_size = 128
#         self.hidden_layer = 4
#         self.class_num = 2
#         self.dropout = 0.25
#         self.biGRU = nn.GRU(self.embed_dim,self.hidden_size,self.hidden_layer,dropout=self.dropout,bidirectional=True)
#         self.softmax_layer = nn.Sequential(
#             nn.Linear(16+2*self.hidden_size+102,self.hidden_size),
#             nn.LeakyReLU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(self.hidden_size,self.class_num),
#         )
#         self.position_layer = nn.Linear(102,self.embed_dim)
#         self.conv1 = nn.Conv2d(1,32,kernel_size=(2,1),stride=(1,1))
#         self.fc = nn.Linear(32,16)
#         self.init_weights()

#     def init_weights(self):
#         for name, param in self.named_parameters():
#             if name.find('weight')!=-1:
#                 nn.init.xavier_normal(param.data)
#             elif name.find('bias')!=-1:
#                 param.data.uniform_(-0.1,-0.1)

#     def init_hidden(self,batch_size):
      
#         hidden = Variable(torch.zeros(self.hidden_layer*2,batch_size,self.hidden_size).cuda())
#         return hidden

#     def forward(self,textE=None,emoE=None,posE=None,init_state=None,textC=None,emoC=None,posC=None,train=None):

#         if train=='True':
#             text_inputs = torch.cat([textE,textC],0)
#             text_inputs = text_inputs.permute(1,0,2)
#             pos = torch.cat([posE,posC],0)
#             emo = torch.cat([torch.sum(emoE,1),torch.sum(emoC,1)],0)
            
#         if train=='False':
            
#             text_inputs = textC.permute(1,0,2)
#             pos = posC
#             emo = torch.sum(emoC,1)
#         outputs,_ = self.biGRU(text_inputs,init_state)
#         last_outputs = outputs[-1].squeeze()
       
#         bs,dims = last_outputs.size()
        
#         pos_repre = self.position_layer(pos)  #[batch,39]->[batch,embed]
      
#         emo_pos = pos_repre*emo  #[batch,embedding]
#         emo_pos = emo_pos.unsqueeze(1)
#         emo_pos = emo_pos.unsqueeze(3)
#         e_p = self.conv1(emo_pos)
#         e_p = F.leaky_relu(e_p) 
#         e_p = F.max_pool2d(e_p,(self.embed_dim-2+1,1))
#         e_p = e_p.squeeze() 
#         e_p = self.fc(e_p)  #batch,16
     
#         classification = torch.cat([e_p,last_outputs],1)
#         class_p = torch.cat([classification,pos],1)
#         logits = self.softmax_layer(class_p)
#         probs = F.softmax(logits,1)

#         return logits,probs
        

# class classifier(nn.Module):
#     def __init__(self):
#         super(classifier,self).__init__()
#         self.embed_dim = 30
#         self.hidden_size = 128
#         self.hidden_layer = 4
#         self.class_num = 2
#         self.dropout = 0.25
#         self.biGRU = nn.GRU(self.embed_dim,self.hidden_size,self.hidden_layer,dropout=self.dropout,bidirectional=True)
#         self.softmax_layer = nn.Sequential(
#             nn.Linear(16+2*self.hidden_size+66,self.hidden_size),
#             nn.LeakyReLU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(self.hidden_size,self.class_num),
#         )
#         self.position_layer = nn.Linear(66,self.embed_dim)
#         self.conv1 = nn.Conv2d(1,32,kernel_size=(2,1),stride=(1,1))
#         self.fc = nn.Linear(32,16)
#         self.init_weights()

#     def init_weights(self):
#         for name, param in self.named_parameters():
#             if name.find('weight')!=-1:
#                 nn.init.xavier_normal(param.data)
#             elif name.find('bias')!=-1:
#                 param.data.uniform_(-0.1,-0.1)

#     def init_hidden(self,batch_size):
      
#         hidden = Variable(torch.zeros(self.hidden_layer*2,batch_size,self.hidden_size).cuda())
#         return hidden

#     def forward(self,textE=None,emoE=None,posE=None,init_state=None,textC=None,emoC=None,posC=None,train=None):

#         if train=='True':
#             text_inputs = textE.permute(1,0,2)
#             pos = posE
#             emo = torch.sum(emoE,1)
            
#         if train=='False':
            
#             text_inputs = textC.permute(1,0,2)
#             pos = posC
#             emo = torch.sum(emoC,1)
       
#         outputs,_ = self.biGRU(text_inputs,init_state)
     
#         last_outputs = outputs[-1].squeeze()
       
#         bs,dims = last_outputs.size()
        
#         pos_repre = self.position_layer(pos)  #[batch,39]->[batch,embed]
      

#         emo_pos = pos_repre*emo  #[batch,embedding]
#         emo_pos = emo_pos.unsqueeze(1)
#         emo_pos = emo_pos.unsqueeze(3)
      
#         e_p = self.conv1(emo_pos)
#         e_p = F.leaky_relu(e_p) 
#         e_p = F.max_pool2d(e_p,(self.embed_dim-2+1,1))
       
#         e_p = e_p.squeeze() 
#         e_p = self.fc(e_p)  #batch,16
     
#         classification = torch.cat([e_p,last_outputs],1)
#         class_p = torch.cat([classification,pos],1)
#         logits = self.softmax_layer(class_p)
#         probs = F.softmax(logits,1)

#         return logits,probs


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.embed_dim = 30
        self.dropout = 0.25
        self.hidden_units1 = 100
        self.hidden_units2 = 30
        self.class_num = 4

        self.fullConn_layer = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_units1),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_units1, self.hidden_units2),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_units2, self.class_num)
        )

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.find('weight') != -1:
                nn.init.xavier_normal(param.data)
            elif name.find('bias') != -1:
                param.data.uniform_(-0.1, -0.1)
    def forward(self,textE,emoE,textC,emoC,textEC,emoEC,textCE,emoCE,alpha):
       
        textE = GRL.apply(textE,alpha)
        textC = GRL.apply(textC,alpha)
        textEC = GRL.apply(textEC,alpha)
        textCE = GRL.apply(textCE,alpha)
        emoE = GRL.apply(emoE,alpha)
        emoC= GRL.apply(emoC,alpha)
        emoEC = GRL.apply(emoEC,alpha)
        emoCE = GRL.apply(emoCE,alpha)

        textE = torch.sum(textE,1) #[bs,seq,30]
        textC = torch.sum(textC,1)
        textEC = torch.sum(textEC,1)
        textCE = torch.sum(textCE,1)

        emoE = torch.sum(emoE,1) #[bs,seq,30]
        emoC = torch.sum(emoC,1)
        emoEC = torch.sum(emoEC,1)
        emoCE = torch.sum(emoCE,1)

        text_inputs = torch.cat([textE, textC,textEC,textCE], 0)
        emo = torch.cat([emoE, emoC,emoCE,emoEC], 0)
    
        inputs = text_inputs+emo
        logits = self.fullConn_layer(inputs)
        probs = F.softmax(logits, 1)

        return logits,probs

#     def forward(self,textE,emoE,textC,emoC):
#         text_inputs = torch.cat([textE, textC], 0)
#         emo = torch.cat([emoE, emoC], 0)
#         inputs = torch.sum(torch.cat([text_inputs,emo],1),1)
#         logits = self.fullConn_layer(inputs)
#         probs = F.softmax(logits, 1)

#         return logits,probs



class GRL(Function):
    @staticmethod
    def forward(ctx, x,alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None









