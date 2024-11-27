import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader,Dataset

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import os


class RecPreferenceModel(nn.Module):
    def __init__(self,embed_size_user,embed_size_cat,hidden_size,
                size_user,size_cat,size_poi,
                num_layers,size_tgt,dropout=0.1):
        super(RecPreferenceModel,self).__init__()
        
        self.embed_user=nn.Embedding(size_user,embed_size_user)
        self.embed_cat=nn.Embedding(size_cat,embed_size_cat)
        
        lstm_size=embed_size_user+embed_size_cat+2#delta_t，delta_d维度
        self.lstm=nn.LSTM(lstm_size,hidden_size,num_layers,batch_first=True,dropout=dropout)
        self.fc=nn.Linear(hidden_size,size_tgt)
    
    def forward(self,inputs_user,inputs_cat,inputs_delta_t,inputs_delta_d):
        '''
        inputs_cat、inputs_delta_t,inputs_delta_d是padding之后的
        '''
        inputs_user=self.embed_user(inputs_user).unsqueeze(1).repeat(1,inputs_cat.size(1),1)#(n_user,1)->(n_user,seq_len,1)
        inputs_cat=self.embed_cat(inputs_cat)#(n_user,seq_len,embed_cat)
        inputs_delta_t=inputs_delta_t.unsqueeze(2)#(n_users,seq_len,1)
        inputs_delta_d=inputs_delta_d.unsqueeze(2)#(n_users,seq_len,1)
        inputs=torch.cat((inputs_user,inputs_cat,inputs_delta_t,inputs_delta_d),2)
        output,_=self.lstm(inputs)
        out=self.fc(output)
        return out