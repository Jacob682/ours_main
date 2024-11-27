import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

import pandas as pd
import numpy as np
import datetime
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

embed_size_user=50
embed_size_cat=100
embed_size_poi=300

hidden_size=128

size_cat=285
size_poi=3906

num_layers=1
num_rec=20
batch_size=1
epoch_size=25
lr=0.0001

class RecPreferenceModel(nn.Module):
    def __init__(self,embed_size_user,embed_size_cat,embed_size_poi,hidden_size,
                size_user,size_cat,size_poi,
                num_layers,num_rec,dropout=0.0):
        super(RecPreferenceModel,self).__init__()
        
        self.num_rec=num_rec
        self.embed_user=nn.Embedding(size_user,embed_size_user)
        self.embed_cat=nn.Embedding(size_cat,embed_size_cat)
        self.embed_poi=nn.Embedding(size_poi,embed_size_poi)
        
        lstm_size=embed_size_user+embed_size_cat+embed_size_poi+2#delta_t，delta_d维度
        self.lstm=nn.LSTM(lstm_size,hidden_size,num_layers,batch_first=True,dropout=dropout)
        self.fc=nn.Linear(hidden_size,size_poi)
    
    def forward(self,inputs_user,inputs_cat,inputs_poi,inputs_delta_t,inputs_delta_d,rec_lens):
        '''
        inputs_cat、inputs_delta_t,inputs_delta_d是padding之后的
        rec_lens:每个时间步rec的真实长度，滑动窗口tensor
        '''
        inputs_user=self.embed_user(inputs_user).unsqueeze(1).repeat(1,inputs_cat.size(1),1)#(n_user,embs)->(n_user,seq_len,embs)
        inputs_user=inputs_user.unsqueeze(2).expand(-1,-1,self.num_rec,-1)#(bs,sq,rec,embs)

        inputs_cat=self.embed_cat(inputs_cat)#(n_user,seq_len,rec,embed_cat)
        inputs_poi=self.embed_poi(inputs_poi)#(n_user,seq_len,rec,embed_poi)
        inputs_delta_t=inputs_delta_t.unsqueeze(-1)#(n_users,seq_len,rec,1)
        inputs_delta_d=inputs_delta_d.unsqueeze(-1)#(n_users,seq_len,rec,1)
        inputs=torch.cat((inputs_user,inputs_cat,inputs_poi,inputs_delta_t,inputs_delta_d),-1) 
        valid_inputs=inputs.view(inputs.size()[0]*inputs.size()[1],inputs.size()[2],inputs.size()[3])#(bs*seq,rec,embs)
        rec_lens=rec_lens.cpu().to(torch.int64)
        valid_inputs=pack_padded_sequence(valid_inputs,rec_lens.view(rec_lens.size()[0]*rec_lens.size()[1]),batch_first=True,enforce_sorted=False)
        output,(h,c)=self.lstm(valid_inputs)#取hidden，(bs*seqlen,hidden)
        c=c.view(rec_lens.size()[0],rec_lens.size()[1],-1)#(bs,sq,hidden)
        out=self.fc(c)#注掉之后相当于直接输出隐藏层,没注释掉之后就是还是tgt个输出,最后用second_importance在相应位置上做权重
        return out


#准备数据
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dir_user='/home/jovyan/datasets/tsmc_nyc_3_groupby_user_chronological/user_id.pkl'
with open(dir_user,'rb') as f:
    users=pickle.load(f)
users=torch.IntTensor(users)
size_user=max(users)+1
dir_train_rec_x='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/train_test_padding/train_rec_padded.pkl'
with open(dir_train_rec_x,'rb') as f:
    train_rec_x=pickle.load(f)
    train_rec_x_poi_int=train_rec_x['poi_int']
    train_rec_x_cat_int=train_rec_x['cat_int']
    train_rec_x_delta_t=train_rec_x['delta_t']
    train_rec_x_delta_d=train_rec_x['delta_d']

##1.2tgt_poi数据
dir_train_y='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/train_test_padding/train_padded_y.pkl'
with open(dir_train_y,'rb') as f:
    train_y=pickle.load(f)
    train_y_poi_int=train_y['poi_int']

#1.1读取second_importance数据，读取second和cat_id对应数据
dir_train_x='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/train_test_padding/train_padded_x.pkl'
with open(dir_train_x,'rb') as f:
    train_x=pickle.load(f)
    train_x_second_importance=train_x['users_timesteps_second_importance']
dir_dic_second_id='/home/jovyan/datasets/tsmc_nyc_6_distinguish_cat_poi/distin_dic_second_cat_id.pkl'
dir_dic_cat_id='/home/jovyan/datasets/tsmc_nyc_6_distinguish_cat_poi/distin_dic_cat_id.pkl'
dir_dic_id_second='/home/jovyan/datasets/level/foursquare_category_level/foursquare_category_level_v2/dic_id_second.pkl'

dir_dic_poi_reindex='/home/jovyan/datasets/tsmc_nyc_6_distinguish_cat_poi/dic_poi_id_reindex.pkl'
dir_dic_poi2cat='/home/jovyan/datasets/tsmc_nyc_6_distinguish_cat_poi/dic_poi2cat.pkl'


#1对每个时间步输出的结果(1,tgt_size)在想应位置上添加second_importance

with open(dir_dic_second_id,'rb') as f:
    dic_second_id=pickle.load(f)#二级目录（16进制）：重索引（int）

with open(dir_dic_id_second,'rb') as f:
    dic_id_second=pickle.load(f)#每个二级目录(hex)：目录(hex list)
with open(dir_dic_cat_id,'rb') as f:
    dic_cat_id=pickle.load(f)#目录(hex)：reindex id(int)

with open(dir_dic_poi_reindex,'rb') as f:
    dic_poi_reindex=pickle.load(f)
with open(dir_dic_poi2cat,'rb') as f:
    dic_poi2cat=pickle.load(f)

dir_train_rec_x_lens='/home/jovyan/datasets/tsmc_nyc_train_test/2_recent/train_rec_x_lens.pkl'
with open(dir_train_rec_x_lens,'rb') as f:
    train_rec_x_lens=pickle.load(f)#(us,seq)

dir_cat2_int_poi_int_matrix='/home/jovyan/datasets/level/foursquare_category_level/foursquare_category_level_v2/cat_int2poi_int_matrix/cat_second_poi_int_matrix.pkl'
#这个文件存的是(cat2_size,poi_size)的矩阵，表示poi_int归属于哪个cat2_int
with open(dir_cat2_int_poi_int_matrix,'rb') as f:
    cat2_int_poi_int_matric=pickle.load(f)

#声明模型
rec_pre_model=RecPreferenceModel(embed_size_user,embed_size_cat,embed_size_poi,hidden_size,
                size_user,size_cat,size_poi,
                num_layers,num_rec)


#1.将数据给dataloader
##1.1定义dataloader,已经padding之后再给dataloader，
##最后放入模型,模型里面embed
def Process_train_rec(users,train_rec_x_cat_int,train_rec_x_poi_int,
                train_rec_x_delta_t,train_rec_x_delta_d,
                train_y_poi_int,train_rec_x_lens,train_x_second_importance,
                batch_size):
    '''
    users:(n_user,1)
    padded_reindex_rec_cat_id:(n_user,20,embed_cat_size)
    padded_rec_delta_t/d:(n_user,20),不需要embed，最后在模型里concat
    '''
    class RecDataset(Dataset):
        def __init__(self,users,train_rec_x_cat_int,train_rec_x_poi_int,
                train_rec_x_delta_t,train_rec_x_delta_d,
                train_y_poi_int,train_rec_x_lens,train_x_second_importance):
            self.users=users
            self.train_rec_x_poi_int=train_rec_x_poi_int
            self.train_rec_x_cat_int=train_rec_x_cat_int
            self.train_rec_x_delta_t=train_rec_x_delta_t
            self.train_rec_x_delta_d=train_rec_x_delta_d
            self.train_y_poi_int=train_y_poi_int
            self.train_rec_x_lens=train_rec_x_lens
            self.train_x_second_importance=train_x_second_importance
        def __len__(self):
            return len(self.users)#长度相同为所有用户个数
        def __getitem__(self,index):#得到每一个用户的数据，非每个时间步，即每个用户的打卡数据
            user=self.users[index]
            poi_id=self.train_rec_x_poi_int[index]
            cat_id=self.train_rec_x_cat_int[index]
            delta_t=self.train_rec_x_delta_t[index]
            delta_d=self.train_rec_x_delta_d[index]
            y_poi_int=self.train_y_poi_int[index]
            rec_x_lens=self.train_rec_x_lens[index]
            second_importance=self.train_x_second_importance[index]
            return user,poi_id,cat_id,delta_t,delta_d,y_poi_int,rec_x_lens,second_importance
    #创建数据集实例
    rec_dataset=RecDataset(users,train_rec_x_cat_int,train_rec_x_poi_int,
                train_rec_x_delta_t,train_rec_x_delta_d,
                train_y_poi_int,train_rec_x_lens,train_x_second_importance)
    #创建dataloader
    dataloader=DataLoader(rec_dataset,batch_size,shuffle=True)
    return dataloader


##1.3将数据传给dataloader
##传cat数据试试
pre_data=Process_train_rec(users,train_rec_x_cat_int,train_rec_x_poi_int,
                train_rec_x_delta_t,train_rec_x_delta_d,
                train_y_poi_int,train_rec_x_lens,train_x_second_importance,batch_size)#,tgt_reindex_rec_cat_id)



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


#2创建for循环，将数据送入模型
#2.1将模型放入cuda
rec_pre_model=rec_pre_model.cuda()
#2.2创建loss函数
loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(rec_pre_model.parameters(),lr)



#3对用户打卡切片得到真实预测的时间步长度
def Regain_batch_outputs_len(batch_step,batch_data):
    # 读rec打卡长度数据，tgt长度相等
    dir_lens='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/train_test_padding/train_x_lens.pkl'
    with open(dir_lens,'rb') as f:
        lens=pickle.load(f)
    # lens=torch.full((10,1),5)
    batch_data_list=[]
    batch_lens=lens[batch_step*(len(batch_data)):(batch_step+1)*(len(batch_data))]
    for i in range(len(batch_data)):
        batch_data_list.append(batch_data[i][:batch_lens[i]])#对batch数据按真实长度切片，忽略了一个问题，batch数据是shuffle过的
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





def Checkin_cat2importance(outputs,dic_cat_id,dic_id_second,dic_second_id,second_importance,dic_poi_reindex,dic_poi2cat,b,batch_size):
    '''
    对tgt_size根据位置赋不同的权重,从而修改lstm_outputs
    outputs:(batch_size,seq_len,tgt_size)
    dic_cat_id:hex:reindex(int)
    dic_id_second:hex:hex list
    dic_second_id:hex: second_reindex(int)
    second_importance:(users,seq,second_size)
    b:第几个batch
    '''
    def Find_key_by_value_121(v,values,keys):
        '''hex:int'''
        index=values.index(v)
        return keys[index]

    def Find_key_by_value_12many(v,dic):
        for key,val_list in dic.items():
            if v in val_list:
                return key
        
    def Find_second_by_cat_id(cat_id,second_keys,dic_id_second):
        '''catid(hex)->second_id(hex)'''
        if cat_id in second_keys:
            return cat_id
        elif cat_id in second_values:
            return Find_key_by_value_12many(cat_id,dic_id_second)

    def Find_second_importance_by_secondhex(second_hex,dic_second_id,ui,step,second_importance,b,batch_size):
        '''second_hex->second_id(int)'''
        # second_values=list(dic_second_id.values())
        # second_keys=dic_second_id.keys()
        second_id=dic_second_id[second_hex]#second_hec2second_id
        importance=second_importance[b*(batch_size)+ui][step][second_id]#second_id2importance
        return importance
    
    cat_reindex_id=list(dic_cat_id.values())
    cat_id=list(dic_cat_id.keys())
    second_keys=list(dic_id_second.keys())
    second_values=set([x for sublist in dic_id_second.values() for x in sublist])

    poi_hex=list(dic_poi_reindex.keys())
    poi_int=list(dic_poi_reindex.values())
    poi2cat_values=list(dic_poi2cat.values())
    poi2cat_keys=list(dic_poi2cat.keys())

    len_outputs=len(outputs)
    matrix_importance=torch.zeros_like(outputs)#做乘法矩阵
    # outputs_with_importance=[[[0 for _ in step] for step in u] for u in outputs]
    
    for ui,u in enumerate(outputs):#修改每个用户
        for stepi,step in enumerate(u):#对用户每个时间步修改
            for ci,c in enumerate(step):#对每个预测位置修改，已经reindex
                if ci in poi_int:#当预测索引在所有目录重索引中时
                    position_poi_hex=Find_key_by_value_121(ci,poi_int,poi_hex)#poi_int->poi_hex
                    position_cat_hex=dic_poi2cat[position_poi_hex]#Find_key_by_value_121(position_poi_hex,poi2cat_values,poi2cat_keys)#poi_hex->cat_hex
                    # c_id=Find_key_by_value_121(ci,cat_reindex_id,cat_id)#取对应的index->cat_id
                    cat_second_id=Find_second_by_cat_id(position_cat_hex,second_keys,dic_id_second)#将对应id(hex)映射到二级目录(hex)
                    importance=Find_second_importance_by_secondhex(cat_second_id,dic_second_id,ui,stepi,second_importance,b,batch_size)#查找该二级目录重要性
                    matrix_importance[ui][stepi][ci]=importance#更新output对应位置值
    outputs_with_importance=torch.mul(outputs,matrix_importance)
    return outputs_with_importance



#1.用second_importance直接做权重
#放入epoch训练
for epoch in range(epoch_size):
    start=datetime.datetime.now()

    rec_pre_model.train()
    epoch_loss=0.0
    
    top_k_1=0
    top_k_5=0
    top_k_10=0
    top_k_20=0
    data_len=len(pre_data)
    
    for batch_step,(batch_users,batch_poi_id,batch_cat_id,batch_delta_t,
              batch_delta_d,batch_y_poi_int,batch_rec_x_lens,batch_x_second_importance) in enumerate(pre_data):
        '''
        batch_users：批中用户编号，未reindex
        '''
        #将batch放进模型
        rec_pre_model.zero_grad()
        poi_candidate=torch.arange(size_poi)# cat_id_candidate=torch.arange(size_cat)#待预测对象,可改为poi_candidate

        #将数据放到cuda
        batch_users=batch_users.cuda()
        batch_cat_id=batch_cat_id.cuda()
        batch_poi_id=batch_poi_id.cuda()
        batch_delta_t=batch_delta_t.cuda()
        batch_delta_d=batch_delta_d.cuda()
        batch_rec_x_lens=batch_rec_x_lens.cuda()
        batch_x_second_importance=batch_x_second_importance.cuda()

        outputs=rec_pre_model(batch_users,batch_cat_id,batch_poi_id,batch_delta_t,batch_delta_d,batch_rec_x_lens)#(batch_size,padded_seqlen,tgt_size)
        #计算一个批次的损失
        loss=0
        #乘上权重
        # batch_outputs_list_with_importance=Checkin_cat2importance(outputs,dic_cat_id,dic_id_second,dic_second_id,\
        #     train_x_second_importance,dic_poi_reindex,dic_poi2cat,batch_step,batch_size)
        batch_outputs_list_with_importance=torch.mul(outputs,torch.matmul(batch_x_second_importance,cat2_int_poi_int_matric.cuda()))
        #3对每个用户计算真实的时间步
        batch_outputs_list=Regain_batch_outputs_len(batch_step,batch_outputs_list_with_importance)
        #3.1去得每个用户的nonpadded打卡cat_id
        batch_nonpadded_tgt_cat_id,_=Regain_batch_tgt(batch_step,batch_y_poi_int)
        for i in range(batch_users.size(0)):#i是每一个用户
            '''batch_output[i] (seq_len,candidate_size)  batch_out(batch_size,seq_len,candidate_size)
            batch_nonpadded_tgt_cat_id[i] (seq_len,1)  batch_nonpadded_tgt_cat_id(batch_size,seqlen,candidate_size)'''
            loss+=loss_function( batch_outputs_list[i],torch.tensor(batch_nonpadded_tgt_cat_id[i]).squeeze().cuda())

        loss.backward()#每个batch做一次反向传播
        optimizer.step()
        epoch_loss+=float(loss)#每个epoch的损失

        #准备评价数据
        outputs_evaluation=[sliced_tensor[-1,:] for sliced_tensor in batch_outputs_list_with_importance]#batch_outputs_list[:,-1,:]
        tgt_evaluation=[sliced_tensor[-1] for sliced_tensor in batch_nonpadded_tgt_cat_id]#batch_nonpadded_tgt_cat_id[:,-1]

        out_p,indices=torch.sort(torch.stack(outputs_evaluation),dim=1,descending=True)
        count=float(len(batch_users))
        #计算评价指标
        top_k_1+=Top_k_precision(indices,tgt_evaluation,1)
        top_k_5+=Top_k_precision(indices,tgt_evaluation,5)
        top_k_10+=Top_k_precision(indices,tgt_evaluation,10)
        top_k_20+=Top_k_precision(indices,tgt_evaluation,20)
    end = datetime.datetime.now()
    total = (end - start).total_seconds()
    print("--  total: @ %.3fs = %.2fh" % (total, total / 3600.0))
    print(
        'epoch:[{}/{}]\t'.format(epoch,epoch_size),
        'loss:{:.4f}\t'.format(epoch_loss),
        'top@1:{:4f}\t'.format(top_k_1/data_len),
        'top@5:{:4f}\t'.format(top_k_5/data_len),
        'top@10:{:4f}\t'.format(top_k_10/data_len),
        'top@20:{:4f}\t'.format(top_k_20/data_len)
    )