import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader,Dataset

import pandas as pd
import numpy as np
import pickle
import os
import sys
sys.path.append('/home/jovyan/code/my_code/my_model')
import warnings
warnings.filterwarnings('ignore')
from rec_preference_model import RecPreferenceModel
from process import Process_rec

dir_user='/home/jovyan/datasets/tsmc_nyc_3_groupby_user_chronological/user_id.pkl'
dir_padded_reindex_rec_cat_id='/home/jovyan/datasets/tsmc_nyc_7_rec_tgt_padding/padded_reindex_rec_cat_id.pkl'
dir_padded_rec_delta_t='/home/jovyan/datasets/tsmc_nyc_7_rec_tgt_padding/padded_rec_delta_t.pkl'
dir_padded_rec_delta_d='/home/jovyan/datasets/tsmc_nyc_7_rec_tgt_padding/padded_rec_delta_d.pkl'
dir_reindex_tgt=['/home/jovyan/datasets/tsmc_nyc_7_rec_tgt_padding/padded_tgt_reindex_rec_cat_id.pkl',
                 '/home/jovyan/datasets/tsmc_nyc_7_rec_tgt_padding/padded_tgt_reindex_rec_poi.pkl']
with open(dir_user,'rb') as f:
    users=pickle.load(f)
users=torch.IntTensor(users)

with open(dir_padded_reindex_rec_cat_id,'rb') as f:
    padded_reindex_rec_cat_id=pickle.load(f)

with open(dir_padded_rec_delta_t,'rb') as f:
    padded_rec_delta_t=pickle.load(f)

with open(dir_padded_rec_delta_d,'rb') as f:
    padded_rec_delta_d=pickle.load(f)


# reindex_tgt_name=['tgt_reindex_rec_cat_id','tgt_reindex_rec_poi']
with open(dir_reindex_tgt[0],'rb') as f:
    padded_tgt_reindex_rec_cat_id=pickle.load(f)
with open(dir_reindex_tgt[1],'rb') as f:
    padded_tgt_reindex_rec_poi=pickle.load(f)

embed_size_user=50
embed_size_cat=100
hidden_size=128
size_user=max(users)+1
size_cat=285
size_poi=3906
num_layers=1
batch_size=5
epoch_size=25
lr=0.001
#声明模型
rec_pre_model=RecPreferenceModel(embed_size_user,embed_size_cat,hidden_size,
size_user,size_cat,size_poi,num_layers,size_cat)

pre_data=Process_rec(users,padded_reindex_rec_cat_id,padded_rec_delta_t,padded_rec_delta_d,padded_tgt_reindex_rec_cat_id,batch_size)#,tgt_reindex_rec_cat_id)

#2.3定义评价矩阵
def Top_k_precision(indices, batch_y, k):
    '''
    indices:一个batch排序之后的下标，(batch_size,待预测长度),size_cat/size_poi，tensor
    batch_y:(batch_size)
    '''
    precision = 0
    for i in range(indices.size(0)):
        sort = indices[i]
        if batch_y[i] in sort[:k]:
            precision += 1
    return precision / indices.size(0)

#3对用户打卡切片得到真实预测的时间步长度
def Regain_batch_outputs_len(batch_step,batch_data):
    # 读rec打卡长度数据，tgt长度相等
    dir_lens='/home/jovyan/datasets/tsmc_nyc_7_rec_tgt_padding/lens.pkl'
    with open(dir_lens,'rb') as f:
        lens=pickle.load(f)
    # lens=torch.full((10,1),5)
    batch_data_list=[]
    batch_lens=lens[batch_step*(len(batch_data)):(batch_step+1)*(len(batch_data))]
    for i in range(len(batch_data)):
        batch_data_list.append(batch_data[i][:batch_lens[i]])
    return batch_data_list

def Regain_batch_tgt(batch_step,batch_data):
    '''可以是cat_id,poi_id'''
    dir_nonpadded_reindex_cat_id='/home/jovyan/datasets/tsmc_nyc_4_recent_target/tgt_reindex_cat_id_poi/tgt_reindex_rec_cat_id.pkl'
    dir_nonpadded_reindex_poi='/home/jovyan/datasets/tsmc_nyc_4_recent_target/tgt_reindex_cat_id_poi/tgt_reindex_rec_poi.pkl'
    dir_nonpadded_reindex=[dir_nonpadded_reindex_cat_id,dir_nonpadded_reindex_poi]
    var_name=['nonpadded_cat_id','nonpadded_poi']
    for d,v in zip(dir_nonpadded_reindex,var_name):
        with open(d,'rb') as f:
            exec(f'{v}={pickle.load(f)}',globals())
    batch_cat_id_list=nonpadded_cat_id[batch_step*(len(batch_data)):(batch_step+1)*(len(batch_data))]
    batch_poi_list=nonpadded_poi[batch_step*(len(batch_data)):(batch_step+1)*(len(batch_data))]
    return batch_cat_id_list,batch_poi_list


#2创建for循环，将数据送如模型
#2.1将模型放入cuda
rec_pre_model=rec_pre_model.cuda()
#2.2创建loss函数
loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(rec_pre_model.parameters(),lr)

#放入epoch训练
for epoch in range(epoch_size):
    rec_pre_model.train()
    epoch_loss=0.0
    
    top_k_1=0
    top_k_5=0
    top_k_10=0
    top_k_20=0
    data_len=len(pre_data)
    
    for batch_step,(batch_users,batch_cat_id,batch_delta_t,
              batch_delta_d,batch_tgt_cat_id) in enumerate(pre_data):
        '''
        batch_users：批中用户编号，未reindex
        '''
        #将batch放进模型
        rec_pre_model.zero_grad()
        cat_id_candidate=torch.arange(size_cat)#待预测对象,可改为poi_candidate

        # #将数据放到cuda
        batch_users=batch_users.cuda()
        batch_cat_id=batch_cat_id.cuda()
        batch_delta_t=batch_delta_t.cuda()
        batch_delta_d=batch_delta_d.cuda()

        outputs=rec_pre_model(batch_users,batch_cat_id,batch_delta_t,batch_delta_d)
        #计算一个批次的损失
        '''print(outputs.size())#(5,20,285)
        print(batch_tgt_cat_id.size())#(5,20)'''
        loss=0
        #3对每个用户计算真实的时间步
        batch_outputs_list=Regain_batch_outputs_len(batch_step,outputs)
        #3.1去得每个用户的nonpadded打卡cat_id
        batch_nonpadded_tgt_cat_id,_=Regain_batch_tgt(batch_step,batch_tgt_cat_id)
        for i in range(batch_users.size(0)):#i是每一个用户
            '''batch_output[i] (seq_len,candidate_size)  batch_out(batch_size,seq_len,candidate_size)
            batch_nonpadded_tgt_cat_id[i] (seq_len,1)  batch_nonpadded_tgt_cat_id(batch_size,seqlen,candidate_size)'''
            loss+=loss_function(batch_outputs_list[i],torch.tensor(batch_nonpadded_tgt_cat_id[i]).squeeze().cuda())

        loss.backward()#每个batch做一次反向传播
        optimizer.step()
        epoch_loss+=float(loss)#每个epoch的损失

        #准备评价数据
        outputs_evaluation=[sliced_tensor[-1,:] for sliced_tensor in batch_outputs_list]#batch_outputs_list[:,-1,:]
        tgt_evaluation=[sliced_tensor[-1] for sliced_tensor in batch_nonpadded_tgt_cat_id]#batch_nonpadded_tgt_cat_id[:,-1]

        out_p,indices=torch.sort(torch.stack(outputs_evaluation),dim=1,descending=True)
        count=float(len(batch_users))
        #计算评价指标
        top_k_1+=Top_k_precision(indices,tgt_evaluation,1)
        top_k_5+=Top_k_precision(indices,tgt_evaluation,5)
        top_k_10+=Top_k_precision(indices,tgt_evaluation,10)
        top_k_20+=Top_k_precision(indices,tgt_evaluation,20)
    print(
        'epoch:[{}/{}]\t'.format(epoch,epoch_size),
        'loss:{:.4f}\t'.format(epoch_loss),
        'top@1:{:4f}\t'.format(top_k_1/data_len),
        'top@5:{:4f}\t'.format(top_k_5/data_len),
        'top@10:{:4f}\t'.format(top_k_10/data_len),
        'top@20:{:4f}\t'.format(top_k_20/data_len)
    )

#4重新训练模型
#放入epoch训练
rec_pre_model_poi=RecPreferenceModel(embed_size_user,embed_size_cat,hidden_size,
size_user,size_cat,size_poi,num_layers,size_poi)
rec_pre_model_poi.cuda()
pre_data_poi=Process_rec(users,padded_reindex_rec_cat_id,
                                  padded_rec_delta_t,padded_rec_delta_d,
                                  padded_tgt_reindex_rec_poi,batch_size)
for epoch in range(epoch_size):
    rec_pre_model_poi.train()
    epoch_loss=0.0
    
    top_k_1=0
    top_k_5=0
    top_k_10=0
    top_k_20=0
    data_len=len(pre_data_poi)
    
    for batch_step,(batch_users,batch_cat_id,batch_delta_t,
              batch_delta_d,batch_tgt_poi) in enumerate(pre_data_poi):
        '''
        batch_users：批中用户编号，未reindex
        '''
        #将batch放进模型
        rec_pre_model.zero_grad()
        poi_candidate=torch.arange(size_poi)#待预测对象,可改为poi/candidate

        # #将数据放到cuda
        batch_users=batch_users.cuda()
        batch_cat_id=batch_cat_id.cuda()
        batch_delta_t=batch_delta_t.cuda()
        batch_delta_d=batch_delta_d.cuda()

        outputs=rec_pre_model_poi(batch_users,batch_cat_id,batch_delta_t,batch_delta_d)
        #计算一个批次的损失
        '''print(outputs.size())#(5,20,285)
        print(batch_tgt_cat_id.size())#(5,20)'''
        loss=0
        #3对每个用户计算真实的时间步
        batch_outputs_list=Regain_batch_outputs_len(batch_step,outputs)
        #3.1去得每个用户的nonpadded打卡cat_id
        batch_nonpadded_tgt_poi,_=Regain_batch_tgt(batch_step,batch_tgt_poi)
        for i in range(batch_users.size(0)):#i是每一个用户
            '''batch_output[i] (seq_len,candidate_size)  batch_out(batch_size,seq_len,candidate_size)
            batch_nonpadded_tgt_cat_id[i] (seq_len,1)  batch_nonpadded_tgt_cat_id(batch_size,seqlen,candidate_size)'''
            loss+=loss_function(batch_outputs_list[i],torch.tensor(batch_nonpadded_tgt_poi[i]).squeeze().cuda())

        loss.backward()#每个batch做一次反向传播
        optimizer.step()
        epoch_loss+=float(loss)#每个epoch的损失

        #准备评价数据
        outputs_evaluation=[sliced_tensor[-1,:] for sliced_tensor in batch_outputs_list]#batch_outputs_list[:,-1,:]
        tgt_evaluation=[sliced_tensor[-1] for sliced_tensor in batch_nonpadded_tgt_poi]#batch_nonpadded_tgt_cat_id[:,-1]

        out_p,indices=torch.sort(torch.stack(outputs_evaluation),dim=1,descending=True)
        count=float(len(batch_users))
        #计算评价指标
        top_k_1+=Top_k_precision(indices,tgt_evaluation,1)
        top_k_5+=Top_k_precision(indices,tgt_evaluation,5)
        top_k_10+=Top_k_precision(indices,tgt_evaluation,10)
        top_k_20+=Top_k_precision(indices,tgt_evaluation,20)
    print(
        'epoch:[{}/{}]\t'.format(epoch,epoch_size),
        'loss:{:.4f}\t'.format(epoch_loss),
        'top@1:{:4f}\t'.format(top_k_1/data_len),
        'top@5:{:4f}\t'.format(top_k_5/data_len),
        'top@10:{:4f}\t'.format(top_k_10/data_len),
        'top@20:{:4f}\t'.format(top_k_20/data_len)
    )