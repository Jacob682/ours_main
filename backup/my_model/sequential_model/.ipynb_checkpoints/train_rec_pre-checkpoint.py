import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import os

from rec_preference import RecPreferenceModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dir_user='/home/jovyan/datasets/tsmc_nyc_3_groupby_user_chronological/user_id.pkl'
dir_padded_rec_cat_id='/home/jovyan/datasets/tsmc_nyc_7_rec_padding/padded_rec_cat_id.pkl'
dir_padded_rec_delta_t='/home/jovyan/datasets/tsmc_nyc_7_rec_padding/padded_rec_delta_t.pkl'
dir_padded_rec_delta_d='/home/jovyan/datasets/tsmc_nyc_7_rec_padding/padded_rec_delta_d.pkl'


embed_size_user=50
embed_size_cat=100
hidden_size=128
size_user=1079#len(user)+1=1078+1
size_cat=285
size_poi=3906
num_layers=1
rec_pre_model=RecPreferenceModel(embed_size_user,embed_size_cat,hidden_size,
size_user,size_cat,size_poi,num_layers)


with open(dir_user,'rb') as f:
    users=pickle.load(f)
users=torch.IntTensor(users)

with open(dir_padded_rec_cat_id,'rb') as f:
    padded_rec_cat_id=pickle.load(f)

with open(dir_padded_rec_delta_t,'rb') as f:
    padded_rec_delta_t=pickle.load(f)

with open(dir_padded_rec_delta_d,'rb') as f:
    padded_rec_delta_d=pickle.load(f)
    
out=rec_pre_model(users,padded_rec_cat_id,padded_rec_delta_t,padded_rec_delta_d)