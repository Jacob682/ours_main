import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import datetime
sys.path.append('/home/jovyan/code/my_code/my_model/')
from preference_model.model_part.preference_model import Preference_Model,Process_preference_model_train_state,MLP
from sequential_model.model_part.sequential_model import RecPreferenceModel,Process_train_rec
from process_data.process import Process_model_data
from utils.utils import to_cuda,compute_Log_Loss,label_Generate,top_K_Precision,mrr
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
run_name='ps_model-0.0001-NYC'
log=open('/home/jovyan/code/my_code/my_model/log/log_'+run_name+'txt','w')

batch_size=20
epoch_size=150
num_rec=20
num_layers=[1,2]
dropout=[0.0,0.1]
lr=0.001


embed_size_user=64
embed_size_cat=64
embed_size_poi=64
embed_size_days=8
embed_size_slot=32
embed_size_geohsh=32
hidden_size=128
mlp_units=[256,64,1]

dir_user='/home/jovyan/datasets/tsmc_nyc_3_groupby_user_chronological/user_id.pkl'
with open(dir_user,'rb') as f:
    users=pickle.load(f)
users=torch.IntTensor(users)
size_user=max(users)+1
size_cat=285
size_poi=3906
size_days=8
size_slots=25
size_geohshs=95



#1.1用padded数据
#sequential model data
dir_train_rec_x='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/train_test_padding/train_rec_padded.pkl'
with open(dir_train_rec_x,'rb') as f:
    train_rec_x=pickle.load(f)
    train_rec_x_poi_int=train_rec_x['poi_int']
    train_rec_x_cat_int=train_rec_x['cat_int']
    train_rec_x_delta_t=train_rec_x['delta_t']
    train_rec_x_delta_d=train_rec_x['delta_d']
dir_train_rec_x_lens='/home/jovyan/datasets/tsmc_nyc_train_test/2_recent/train_rec_x_lens.pkl'
with open(dir_train_rec_x_lens,'rb') as f:
    train_rec_x_lens=pickle.load(f)#(us,seq)
dir_train_x='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/train_test_padding/train_padded_x.pkl'
with open(dir_train_x,'rb') as f:
    train_x=pickle.load(f)
    train_x_second_importance=train_x['users_timesteps_second_importance']

dir_test_rec_x='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/train_test_padding/test_rec_padded.pkl'
with open(dir_test_rec_x,'rb') as f:
    test_rec_x=pickle.load(f)
    test_rec_x_poi_int=test_rec_x['poi_int']
    test_rec_x_cat_int=test_rec_x['cat_int']
    test_rec_x_delta_t=test_rec_x['delta_t']
    test_rec_x_delta_d=test_rec_x['delta_d']
dir_test_rec_x_lens='/home/jovyan/datasets/tsmc_nyc_train_test/2_recent/test_rec_x_lens.pkl'
with open(dir_test_rec_x_lens,'rb') as f:
    test_rec_x_lens=pickle.load(f)#(us,seq)
dir_test_x='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/train_test_padding/test_padded_x.pkl'
with open(dir_test_x,'rb') as f:
    test_x=pickle.load(f)
    test_x_second_importance=test_x['users_timesteps_second_importance']

#preference model data
dir_train_padded_x='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/train_test_padding/train_padded_x.pkl'#字典有poi_int,cat_int,day,hour,hs5_int,days_subs
dir_negs='/home/jovyan/datasets/tsmc_nyc_11_negative_sample/train_negative_sample/us_negs.pkl'#(us_negs_poi_int,us_negs_cat_int)
dir_train_padded_y='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/train_test_padding/train_padded_y.pkl'
dir_test_padded_y='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/train_test_padding/test_padded_y.pkl'
dir_test_padded_x='/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/train_test_padding/test_padded_x.pkl'#字典有poi_int,cat_int,day,hour,hs5_int,days_subs
dir_all_qs='/home/jovyan/datasets/tsmc_nyc_6_distinguish_cat_poi/cat_id_mapped_poi.pkl'

with open(dir_train_padded_x,'rb') as f:
    train_padded_x=pickle.load(f)
    train_x_poi_int=train_padded_x['poi_int']
    train_x_cat_int=train_padded_x['cat_int']
    train_x_day=train_padded_x['day']
    train_x_hour=train_padded_x['hour']
    train_x_hs5_int=train_padded_x['hs5_int']
    train_x_days_subs=train_padded_x['days_subs']
with open(dir_negs,'rb') as f:
    (negs_poi_int,negs_cat_int)=pickle.load(f)
with open(dir_all_qs,'rb') as f:
    poi_int2cat_int=pickle.load(f)
all_poi_int=torch.arange(size_poi)
all_cat_int=poi_int2cat_int[all_poi_int]
with open(dir_train_padded_y,'rb') as f:
    train_padded_y=pickle.load(f)
    train_y_poi_int=train_padded_y['poi_int']#用于做loss
    train_y_cat_int=train_padded_y['cat_int']#用于和负样本拼接
    train_y_day=train_padded_y['day']
    train_y_hour=train_padded_y['hour']
    train_y_hs5=train_padded_y['hs5_int']
with open(dir_test_padded_y,'rb') as f:
    test_padded_y=pickle.load(f)
    test_y_poi_int=test_padded_y['poi_int']#用于做loss
    test_y_cat_int=test_padded_y['cat_int']#用于和负样本拼接
    test_y_day=test_padded_y['day']
    test_y_hour=test_padded_y['hour']
    test_y_hs5=test_padded_y['hs5_int']
with open(dir_test_padded_x,'rb') as f:
    test_padded_x=pickle.load(f)
    test_x_poi_int=test_padded_x['poi_int']
    test_x_cat_int=test_padded_x['cat_int']
    test_x_day=test_padded_x['day']
    test_x_hour=test_padded_x['hour']
    test_x_hs5_int=test_padded_x['hs5_int']
    test_x_days_subs=test_padded_x['days_subs']



#用户真实时间步数据矩阵(user,maxlen)
with open('/home/jovyan/datasets/tsmc_nyc_8_all_tgt_padding/train_test_padding/train_x_lens_matrix.pkl','rb') as f:
    train_x_lens_mask=pickle.load(f)


class PS_Model(nn.Module):
    def __init__(self,embed_size_user,embed_size_cat,embed_size_poi,hidden_size,
                size_user,size_cat,size_poi,num_layers,num_rec,dropout,
                embed_size_days,embed_size_slot,embed_size_geohsh,
                size_days,size_slots,size_geohshs,mlp_units):
                '''
                num_layer:两个模块的数组[1,2]
                dropout:两个模块的数组[0.0,0.1]
                '''
                emb_size=embed_size_poi+embed_size_cat+embed_size_days+embed_size_slot+embed_size_geohsh
                mlp_size=hidden_size*2+emb_size
                super(PS_Model,self).__init__()
                #初始化两个模型
                self.preference_model=Preference_Model(embed_size_poi,embed_size_cat,embed_size_days,embed_size_slot,embed_size_geohsh,
                hidden_size,
                size_cat,size_poi,size_days,size_slots,size_geohshs,
                num_layers[1],dropout[1])#初始化偏好模型
                self.sequential_model=RecPreferenceModel(embed_size_user,embed_size_cat,embed_size_poi,hidden_size,
                                                        size_user,size_cat,size_poi,
                                                        num_layers[0],num_rec)
                self.mlp=MLP(mlp_units,mlp_size)
    def forward(self,batch_users,rec_cat_id,rec_poi_id,rec_delta_t,rec_delta_d,rec_x_lens,
                x_poi,x_cat,x_day,x_hour,x_hs5,x_days_subs,
                neg_poi,neg_cat,y_poi,y_cat,y_day,y_hour,y_hs5):
                seq_out=self.sequential_model(batch_users,rec_cat_id,rec_poi_id,rec_delta_t,rec_delta_d,rec_x_lens)#(bs,sq,hidden)
                pref_out=self.preference_model(x_poi,x_cat,x_day,x_hour,x_hs5,x_days_subs,
                                                neg_poi,neg_cat,y_poi,y_cat,y_day,y_hour,y_hs5)#(bs,sq,21,hidden+embs)
                seq_out=seq_out.unsqueeze(-2).expand(-1,-1,21,-1)
                concat_out=torch.cat((seq_out,pref_out),dim=-1)#(bs,sq,21,2*h+embs)
                out=self.mlp(concat_out)#(bs,sq,21),每个用户在每个时间步上的概率分布
                return out

ps_model=PS_Model(embed_size_user,embed_size_cat,embed_size_poi,hidden_size,
                size_user,size_cat,size_poi,num_layers,num_rec,dropout,
                embed_size_days,embed_size_slot,embed_size_geohsh,
                size_days,size_slots,size_geohshs,mlp_units)

model_train_data=Process_model_data(users,train_rec_x_cat_int,train_rec_x_poi_int,train_rec_x_delta_t,train_rec_x_delta_d,train_y_poi_int,train_rec_x_lens,train_x_second_importance,
                  train_x_poi_int,train_x_cat_int,train_x_day,train_x_hour,train_x_hs5_int,train_x_days_subs,
                  negs_poi_int,negs_cat_int,
                  train_y_cat_int,train_y_day,train_y_hour,train_y_hs5,
                  train_x_lens_mask,batch_size)
model_test_data=Process_model_data(users,test_rec_x_cat_int,test_rec_x_poi_int,test_rec_x_delta_t,test_rec_x_delta_d,test_y_poi_int,test_rec_x_lens,test_x_second_importance,
                all_poi_int,all_cat_int,
                test_x_poi_int,test_x_cat_int,test_x_day,test_x_hour,test_x_hs5_int,test_x_days_subs,
                test_y_cat_int,test_y_day,test_y_hour,test_y_hs5,
                train_x_lens_mask,batch_size)

loss_function=nn.BCELoss(reduction='none')
optimizer=torch.optim.Adam(ps_model.parameters(),lr)

ps_model=ps_model.cuda()


for epoch in range(epoch_size):
    start=datetime.datetime.now()
    ps_model.train()
    train_epoch_loss=0.0
    top_k_1,top_k_5,top_k_10,top_k_20,mrr_1,mrr_5,mrr_10,mrr_20=0,0,0,0,0,0,0,0
    data_len=len(model_train_data)
    for batch_step,(users,rec_cat,rec_poi,rec_delta_t,rec_delta_d,y_poi,rec_x_lens,second_importance,
                    x_poi,x_cat,x_day,x_hour,x_hs5,x_days_subs,
                    neg_poi,neg_cat,y_cat,y_day,y_hour,y_hs5,len_mask) in enumerate(model_train_data):
        ps_model.zero_grad()
        users,rec_cat,rec_poi,rec_delta_t,rec_delta_d,y_poi,rec_x_lens,second_importance,\
        x_poi,x_cat,x_day,x_hour,x_hs5,x_days_subs,\
        neg_poi,neg_cat,y_cat,y_day,y_hour,y_hs5,len_mask=to_cuda([users,rec_cat,rec_poi,rec_delta_t,rec_delta_d,y_poi,rec_x_lens,second_importance,
                                                        x_poi,x_cat,x_day,x_hour,x_hs5,x_days_subs,
                                                        neg_poi,neg_cat,y_cat,y_day,y_hour,y_hs5,len_mask])
        outputs=ps_model(users,rec_cat,rec_poi,rec_delta_t,rec_delta_d,rec_x_lens,
                        x_poi,x_cat,x_day,x_hour,x_hs5,x_days_subs,
                        neg_poi,neg_cat,y_poi,y_cat,y_day,y_hour,y_hs5)
        labels=label_Generate(outputs.size()).cuda()#(bs,sq,21),onehot

        shuffle_indice=torch.randperm(labels.size(-1))
        labels_shuffle,train_pre=labels[...,shuffle_indice],outputs[...,shuffle_indice]

        batch_avg_loss=compute_Log_Loss(loss_function,train_pre,labels_shuffle,len_mask)
        batch_avg_loss.backward()
        optimizer.step()
        train_epoch_loss+=float(batch_avg_loss)

        top_k_1+=top_K_Precision(train_pre,labels_shuffle,1)
        top_k_5+=top_K_Precision(train_pre,labels_shuffle,5)
        top_k_10+=top_K_Precision(train_pre,labels_shuffle,10)
        top_k_20+=top_K_Precision(train_pre,labels_shuffle,20)
        mrr_1+=mrr(train_pre,labels_shuffle,1)
        mrr_5+=mrr(train_pre,labels_shuffle,5)
        mrr_10+=mrr(train_pre,labels_shuffle,10)
        mrr_20+=mrr(train_pre,labels_shuffle,20)
    

    # train_epoch_loss+=train_epoch_loss
    end=datetime.datetime.now()
    total=(end-start).total_seconds()
    print("--  total: @ %.3fs = %.2fh" % (total, total / 3600.0))
    print(
        'epoch:[{}/{}]\t'.format(epoch,epoch_size),
        'loss:{:.4f}\t'.format(train_epoch_loss),
        'top@1:{:4f}\t'.format(top_k_1/data_len),
        'top@5:{:4f}\t'.format(top_k_5/data_len),
        'top@10:{:4f}\t'.format(top_k_10/data_len),
        'top@20:{:4f}\t'.format(top_k_20/data_len),
        'mrr_1:{:4f}\t'.format(mrr_1/data_len),
        'mrr_5:{:4f}\t'.format(mrr_5/data_len),
        'mrr_10:{:4f}\t'.format(mrr_10/data_len),
        'mrr_20:{:4f}\t'.format(mrr_20/data_len)
    )
    # savedir='/home/jovyan/code/my_code/my_model/checkpoint'+run_name
    # if not os.path.exists(savedir):
    #     os.makedirs(savedir)
    # savename=savedir+'/checkpoint'+'_'+str(epoch)+'.tar'
    # torch.save({'epoch':epoch+1,'state_dict':ps_model.state_dict(),},savename)
    # if epoch%1==0:
    #     ps_model=ps_model.eval()
    #     test_epoch_loss=0.0
    #     top_k_1,top_k_5,top_k_10,top_k_20,mrr_1,mrr_5,mrr_10,mrr_20=0,0,0,0,0,0,0,0
    #     for batch_step,(users,rec_cat,rec_poi,rec_delta_t,rec_delta_d,y_poi,rec_x_lens,second_importance,
    #                     x_poi,x_cat,x_day,x_hour,x_hs5,x_days_subs,
    #                     neg_poi,neg_cat,y_cat,y_day,y_hour,y_hs5,len_mask) in enumerate(model_test_data):#此处的neg是所有candidate
    #         users,rec_cat,rec_poi,rec_delta_t,rec_delta_d,y_poi,rec_x_lens,second_importance,\
    #         x_poi,x_cat,x_day,x_hour,x_hs5,x_days_subs,\
    #         neg_poi,neg_cat,y_cat,y_day,y_hour,y_hs5,len_mask=to_cuda([users,rec_cat,rec_poi,rec_delta_t[...,-1],rec_delta_d[...-1],y_poi[...,-1],rec_x_lens[...,-20:],second_importance,
    #                                                                     x_poi,x_cat,x_day,x_hour,x_hs5,x_days_subs,
    #                                                                     neg_poi,neg_cat,y_cat,y_day,y_hour,y_hs5,len_mask])
    #         outputs=ps_model(users,rec_cat,rec_poi,rec_delta_t,rec_delta_d,rec_x_lens,
    #                     x_poi,x_cat,x_day,x_hour,x_hs5,x_days_subs,
    #                     neg_poi,neg_cat,y_poi,y_cat,y_day,y_hour,y_hs5)
        



