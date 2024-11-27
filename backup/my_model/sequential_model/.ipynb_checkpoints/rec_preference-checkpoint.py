import torch
import torch.nn as nn
import torch.nn.functional as f

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt

class preference_sequencial(nn.Module):
    def __init__(self,embed_size_cat,hidden_size,vocab_cat):
        super(preference_sequencial,self).__init__()
        
        self.embed_cat=nn.Embedding(vocab_cat,embed_size_cat)