# -*- coding: utf-8 -*-
"""
Other Layers.
Reference: Bilateral Multi-Perspective Matching for Natural Language Sentences.
Sheng Liu
All rights reserved
Report bugs to ShengLiu shengliu@nyu.edu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .MultiPerspective import MultiPerspective

class ContextRepresentation(nn.Module):
    '''
    Second Layer, bidirectional rnn.
    Input size: batch_size * sequence_length * (word_embedding size + rnn_dim)
    output size: batch_size * sequence_length * (2 * hidden_size)
    '''
    def __init__(self, input_size, hidden_size = 100, rnn_unit = 'lstm', dropout = 0.1):
        super(ContextRepresentation, self).__init__()
        self.lstm_1 = nn.LSTM(input_size = input_size, hidden_size = 100, num_layers = 1, batch_first = True, dropout = dropout, bidirectional = True)
        self.lstm_2 = nn.LSTM(input_size = input_size, hidden_size = 100, num_layers = 1, batch_first = True, dropout = dropout, bidirectional = True)
        
        
    def forward(self, query,passage):
        out1,_ = self.lstm_1(query)
        out2,_ = self.lstm_2(passage)

        return out1,out2


class PredictionLayer(nn.Module):
    def __init__(self, pre_in, hidden_size, pre_out, dropout = 0.1):
        super(PredictionLayer, self).__init__()
        self.linear1 = nn.Linear(pre_in,hidden_size)
        self.linear2 = nn.Linear(hidden_size,pre_out)
        self.p = dropout
        

    def forward(self,x,boolean):
        
        out1 = F.dropout(F.relu(self.linear1(x)), p=self.p , training = boolean)
        out = F.relu(self.linear2(out1))
        return out



class BiMPM(nn.Module):
    def __init__(self, embedding_dim, perspective, hidden_size = 100, epsilon = 1e-6, num_classes = 2):
        super(BiMPM,self).__init__()
        self.hidden_size = hidden_size
        self.contex_rep = ContextRepresentation(embedding_dim, hidden_size)
        self.multiperspective = MultiPerspective(hidden_size, epsilon, perspective)
        self.aggregation = ContextRepresentation(4 * perspective, hidden_size)
        self.pre = PredictionLayer(4 * hidden_size, hidden_size, num_classes)





    def forward(self, query, passage, hidden_size, boolean):
        
        out1,out2 = self.contex_rep(query,passage)
        out3 = self.multiperspective(out1, out2)
        out4 = self.multiperspective(out2, out1)
        out3,out4 = self.aggregation(out3,out4)
        #timestep x batch x (2*hidden_size)
        pre_list = []
        pre_list.append(out3[-1,:,:hidden_size])
        pre_list.append(out3[0,:,hidden_size:])
        pre_list.append(out4[-1,:,:hidden_size])
        pre_list.append(out4[0,:,hidden_size:])
        pre1 = torch.cat(pre_list,1)
        # batch x (4*hidden_size)
        out = self.pre(pre1,boolean)

        return out