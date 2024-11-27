import torch 
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import math
import sys
sys.path.append('code/my_code/my_model/utils/')
sys.path.append('code/my_code/my_model/stgn_model/')
sys.path.append('code/my_code/my_model/preference_model/model_part2/')
sys.path.append('/home/liuqiuyu/code/my_code/my_model/both_model/')
from two_model import Two_Model
from abla_two_model import Abl_Two_Model
from stgn_model.stgn import STGN_Model
from preference_model.model_part2.preference_model import Preference_Model,MLP,BahdanauAttention
from utils.utils import to_cuda,EarlyStop


class Preference_Stgn(nn.Module):
    def __init__(self,num_x,pref_embs,stgn_embs,mlp_units,drop_out,head_num,num_layers=1):
        '''
        #         0         1      2       3         4       5       6       7
        num_x=[num_ueser,num_poi,num_cat,num_month,num_hsh,num_rec,num_day,num_hours]
              [num_ueser,num_poi,num_cat,num_hsh,num_rec,num_day,num_hours]
        #          0   1   2  3 4  5  6
        pref_embs=[128,64,32,8,16,32]#(hidden,poi,cat,day,hour,hsh5)
        stgn_embs=[hidden_szie_stgn,embed_size_user,embed_size_poi,embed_size_cat,embed_size_month/hours,embed_size_hsh5]
        mlp_units=[,,1]
        drop_out=mlp_drop_out_rate
        '''
        super().__init__()
        
        tar_attn_dropout,hiera_attn_dropout,mlp_dropout=drop_out
        queries_emb_size=sum(pref_embs[1:])+sum(stgn_embs[2:])+pref_embs[3]
        mlp_size=pref_embs[0]*4+sum(pref_embs)+sum(stgn_embs[2:])+pref_embs[3]
        # mlp_size=sum(pref_embs)+stgn_embs[0]+pref_embs[0]#mlp输入的维度，两个模型的hidden+query的embs.pref有两个hidden pref_embs[0]加两次，query在sum中加了,concat
        # self.preference_model=Preference_Model(pref_embs,num_x,num_layers,head_num,tar_attn_dropout)
        seq_num_x=num_x[:3]+[num_x[6]]+[num_x[3]]#(num_ueser,num_poi,num_cat,num_hour,num_hsh)
        # self.sequential_model=STGN_Model(stgn_embs[0],seq_num_x[:5],stgn_embs[1:],num_x[4],mlp_units)
        # print(seq_num_x)
        self.two_model=Abl_Two_Model(pref_embs,num_x,num_layers,head_num,tar_attn_dropout,
                                 stgn_embs[0],seq_num_x[:5],stgn_embs[1:],num_x[4],mlp_units)
        self.hiera_attn=BahdanauAttention(pref_embs[0],queries_emb_size,pref_embs[0],hiera_attn_dropout)

        self.mlp=MLP(mlp_units,mlp_size,mlp_dropout)#mlp_units:中间维度，mlp_size:x的维度
        self.pc_w=nn.Parameter(torch.Tensor(1))
        self.mo_w=nn.Parameter(torch.Tensor(1))
        self.dense=nn.Linear(1,1)
        self.seq_dense_attn=nn.Linear(stgn_embs[0],pref_embs[0])
        self.acti_pc=torch.relu
    def init_weights(self):
        stdv=1.0/math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            weight.data.uniform_(-stdv,stdv)
    def forward(self,inputs,num_neg):
        '''
        inputs=[preference_inputs,stgn_inputs]
        pref_input=[ x_poi,x_cat,x_month,x_day,x_hour,x_hsh5,x_sub_month,x_sub_day,x_sub_hour,x_sub_hsh5,x_neg_poi,x_neg_cat,x_neg_second,\
                    y_cat,y_second,y_month,y_day,y_hour,y_hsh5,\
                    y_poi_int]
        pref_input=[x poi,x cat,x day,x hour,x hsh5,x sub day,x sub hour,x sub hsh5,x neg poi,x neg cat,y_cat,y_day,y hour,y_hsh5,\
                    y_poi int]
        stgn_inputs:=[u,x_rec_poi_int,x_rec_cat_int,x_rec_delta_t,x_rec_delta_d,x_rec_months/x_rec_hours,x_rec_hsh5,x_rec_len](batch,rec_num),所有训练个数：user*timestep
        num_negs=[21/3906]
        '''
        pref_inputs,stgn_inputs=inputs#second_importances(bs,187)
        # pref_out,shuffled_indices=self.preference_model(pref_inputs,num_neg)#shuffled_seconds(bs,neg+1)
        # seq_cat_out,seq_loc_out=self.sequential_model(stgn_inputs)#hidden_size=512
        pref_out,shuffled_indices,seq_cat_out,seq_loc_out=self.two_model(pref_inputs,num_neg,stgn_inputs)
        
        # seq_out=torch.cat((seq_cat_out,seq_loc_out),dim=-1)#对seq_cat_out,seq_loc_out concat，但也可以分别att
        #month pooling,concat
        # concat_out=torch.cat((seq_out.unsqueeze(1).expand(-1,num_neg,-1),pref_out),dim=-1)#(bs,embs)
        # model_out=self.mlp(concat_out)#(bs,21)
        
        #最后加一层att
        attn_day,attn_hour,attn_hsh5,queries=pref_out[0],pref_out[1],pref_out[2],pref_out[3]
        # stgn_dense=self.seq_dense_attn(seq_out.unsqueeze(1).expand(-1,num_neg,-1))#(pref_hidden),linear dense
        stgn_dense_cat=self.seq_dense_attn(seq_cat_out.unsqueeze(1).expand(-1,num_neg,-1))
        stgn_dense_loc=self.seq_dense_attn(seq_loc_out.unsqueeze(1).expand(-1,num_neg,-1))

            #att
        day_attn_out,_=self.hiera_attn(queries,attn_day,None,num_neg)
        hour_attn_out,_=self.hiera_attn(queries,attn_hour,None,num_neg)
        hsh5_attn_out,_=self.hiera_attn(queries,attn_hsh5,None,num_neg)
        stgn_cat_attn_out,_=self.hiera_attn(queries,stgn_dense_cat,None,num_neg)
        stgn_loc_attn_out,_=self.hiera_attn(queries,stgn_dense_loc,None,num_neg)
            #concat
        concat_out=torch.cat((queries,day_attn_out,hour_attn_out,hsh5_attn_out,stgn_cat_attn_out,stgn_loc_attn_out),dim=-1)#(pref_hidden_size*3+sum(pref_embs[1:]))
        model_out=self.mlp(concat_out)


        #加second_importance
        # neg_importance=torch.stack([second_importances[i,shuffled_seconds[i]] for i in range(shuffled_seconds.size(0))])#(bs,neg_num)
        # neg_importance=neg_importance.to(torch.float32)#(bs,neg_num,1)
        # pc_out=self.acti_pc(self.dense(neg_importance))#(b,neg_num,1)
        # out=torch.sigmoid(model_out.unsqueeze(-1)@self.mo_w+pc_out@self.pc_w)#(b,neg_num,1)
        # out=torch.sigmoid((model_out*neg_importance).unsqueeze(-1))
        # out=out.squeeze(-1)

        #pair_wise loss
        # u_embs=(self.sequential_model.embed_user(stgn_inputs[0])).unsqueeze(1).expand(-1,num_neg,-1)#(bs,num_neg+1,embs)
        # restored_indices=[torch.argsort(shuffled_indices[i]) for i in range(model_out.size(0))]
        # i_embs=torch.stack([model_out[i,restored_indices[i],:] for i in range(model_out.size(0))],dim=0)#(bs,num_neg+1,embs)
        # substracted=(u_embs@i_embs.transpose(1,2))[:,0,0]#(bs)
        # substract=(u_embs@i_embs.transpose(1,2))[:,0,1:]#(bs,neg_num)
        # out=torch.sigmoid((substracted.unsqueeze(-1).expand(-1,num_neg-1)-substract).unsqueeze(-1)).squeeze(-1)#(bs,20)
        
        return model_out,torch.stack(shuffled_indices)
    



