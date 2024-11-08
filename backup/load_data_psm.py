import time
import os
import pandas as pd
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
import datetime
from datetime import datetime,timedelta
import pickle
import pygeohash as pgh
import numpy as np
import json5
import pytz
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
from math import radians,sin,cos,sqrt,atan2
from geopy.distance import geodesic
import random
import sys
sys.path.append('./second_importance')
from second_importance import Compute_users_timesteps_importance
from ast import literal_eval

def exe_time(func):
    def new_func(*args, **args2):
        name = func.__name__
        start = datetime.now()
        print("-- {%s} start: @ %ss" % (name, start))
        back = func(*args, **args2)
        end = datetime.now()
        print("-- {%s} start: @ %ss" % (name, start))
        print("-- {%s} end:   @ %ss" % (name, end))
        total = (end - start).total_seconds()
        print("-- {%s} total: @ %.3fs = %.3fh" % (name, total, total / 3600.0))
        return back
    return new_func


def fun_load_data(rec_num,data_fr,neg_num,data_to,dir_mapping_dic,dir_second_importances,span_month,poi_size):
    print('\nOriginal data...')
    cols=['user_id', 'poi_hex', 'category_hex', 'category_name', 'latitude',
       'longitude', 'timezone_offset', 'utc_time', 'geohash5', 'datetime',
       'localtime', 'day', 'hour', 'month', 'category_int', 'poi_int',
       'hsh_int', 'second_hex', 'second_int', 'sub_day','sub_month','sub_hour','sub_hsh5']
    hist=pd.read_csv(data_fr,sep='\t',names=cols)
    hist['sub_day']=hist['sub_day'].apply(literal_eval)
    hist['sub_month']=hist['sub_month'].apply(literal_eval)
    hist['sub_hour']=hist['sub_hour'].apply(literal_eval)
    hist['sub_hsh5']=hist['sub_hsh5'].apply(literal_eval)

    gp=hist.groupby('user_id')
    uids,uids_extend_timesteps=[],[]
    poi_ints,cat_ints,cat_hexs,second_ints,second_hexs=[],[],[],[],[]
    days,hours,months,hsh5s,sub_days,sub_months,sub_hours,sub_hsh5s=[],[],[],[],[],[],[],[]
    localtimes,latis,longis=[],[],[]
    for user_id,hist_group in gp:
        uids.append(list(hist_group['user_id'])[0])#int64
        uids_extend_timesteps.append(list(hist_group['user_id']))
        poi_ints.append(list(hist_group['poi_int']))#int64 (user,timestep^)
        cat_ints.append(list(hist_group['category_int']))#int64
        cat_hexs.append(list(hist_group['category_hex']))
        second_hexs.append(list(hist_group['second_hex']))
        second_ints.append(list(hist_group['second_int']))#int64
        days.append(list(hist_group['day']))#int64
        hours.append(list(hist_group['hour']))#int64
        months.append(list(hist_group['month']))
        hsh5s.append(list(hist_group['hsh_int']))#int64
        sub_days.append(list(hist_group['sub_day']))#int64 list
        sub_months.append(list(hist_group['sub_month']))
        sub_hours.append(list(hist_group['sub_hour']))
        sub_hsh5s.append(list(hist_group['sub_hsh5']))
        localtimes.append(list(hist_group['localtime']))#str array
        latis.append(list(hist_group['latitude']))
        longis.append(list(hist_group['longitude']))
        assert len(poi_ints)==len(cat_ints)==len(second_ints)==len(days)==len(hours)==len(hsh5s)==len(localtimes)==len(latis)==len(longis)
    
    #做delta
    delta_ts=[]
    for u_localtime in localtimes:
        u_delta_t=[]
        u_localtime=pd.to_datetime(u_localtime).tolist()
        for i in range(len(u_localtime)-1):
            u_delta_t.append((u_localtime[i+1]-u_localtime[i]).total_seconds())
        delta_ts.append(u_delta_t)
    delta_ds=[]
    for u_lati,u_longi in zip(latis,longis):
        u_dist=[]
        u_loc=[(lati,longi) for lati,longi in zip(u_lati,u_longi)]
        for i in range(len(u_loc)-1):
            u_dist.append(geodesic(u_loc[i+1],u_loc[i]).kilometers*1000)
        delta_ds.append(u_dist)

    #做rec
    def fun_rec(lst):
        rec_datas=[]#所有用户，所有rec
        for u in lst:
            u_rec=[value[-rec_num:] if len(value)>=rec_num else value for \
                value in [u[:i+1] for i in range(len(u))]]
            rec_datas.append(u_rec)
        return rec_datas
    rec_poi_ints=fun_rec(poi_ints)
    rec_cat_ints=fun_rec(cat_ints)
    rec_delta_ts=fun_rec(delta_ts)
    rec_delta_ds=fun_rec(delta_ds)
    rec_hsh5s=fun_rec(hsh5s)
    rec_months=fun_rec(months)
    rec_hours=fun_rec(hours)

    #计算second_importance
    if os.path.exists(dir_second_importances):
        with open(dir_second_importances,'rb') as f:
            users_timesteps_importance=pickle.load(f)
            pass
    else:
        with open(dir_mapping_dic,'rb') as f:
            [cat_second_ids_dict,cat_second_ids_int2int,cat_hex2int_dic,second_hex2int_dic]=pickle.load(f)
        second_ints_all=[k for k in cat_second_ids_int2int.keys()]
        users_timesteps_importance=Compute_users_timesteps_importance(second_ints,months,second_ints_all,span_month)#(users,timestep,187)

    #计算lens,rec_lens
    # lens,用poi_ints(users,timestep^)
    def compute_lens(input_list):
        def recursive_min_dimension_lens(lst):
            if isinstance(lst[0],list):
                return [recursive_min_dimension_lens(sublist) for sublist in lst]
            else:
                return len(lst)
        return recursive_min_dimension_lens(input_list)
    lens=compute_lens(poi_ints)
    rec_lens=compute_lens(rec_poi_ints)

    #划分xy
    #不再需要具体localtime，lati，longi
    seqs_cols=['uids','uids_extend_timesteps','poi_ints','cat_ints','second_ints','months','days','hours',
            'hsh5s','sub_days','sub_months','sub_hours','sub_hsh5s','rec_poi_ints','rec_cat_ints','rec_delta_ts','rec_delta_ds','rec_months','rec_hours','rec_hsh5s','second_importances','lens','rec_lens']
    seqs=[uids,uids_extend_timesteps,poi_ints,cat_ints,second_ints,months,days,hours,#(user,seq),(user,seq,ren_num)
        hsh5s,sub_days,sub_months,sub_hours,sub_hsh5s,rec_poi_ints,rec_cat_ints,rec_delta_ts,rec_delta_ds,rec_months,rec_hours,rec_hsh5s,users_timesteps_importance,lens,rec_lens]
    dit_fea={k:v for k,v in zip(seqs_cols,seqs)}
    def subtract_one_from_list(input_list):
        def recursive_subtract(lst):
            if isinstance(lst[0], list):
                # 如果是多维列表，递归处理每个子列表
                return [recursive_subtract(sublist) for sublist in lst]
            else:
                # 对最小维度的元素执行减1操作
                return [element - 1 for element in lst]

        return recursive_subtract(input_list)
    def fun_make_xy(dit_fea):
        xy_uids_extend_timesteps=[row[:-1] for row in dit_fea['uids_extend_timesteps']]
        x_poi_ints=[row[:-1] for row in dit_fea['poi_ints']]#(users,seq)
        x_cat_ints=[row[:-1] for row in dit_fea['cat_ints']]
        x_second_ints=[row[:-1] for row in dit_fea['second_ints']]
        x_months=[row[:-1] for row in dit_fea['months']]
        x_days=[row[:-1] for row in dit_fea['days']]
        x_hours=[row[:-1] for row in dit_fea['hours']]
        x_hsh5s=[row[:-1] for row in dit_fea['hsh5s']]
        x_sub_days=[row[:-1] for row in dit_fea['sub_days']]#(users,seq,7)
        x_sub_months=[row[:-1] for row in dit_fea['sub_months']]
        x_sub_hours=[row[:-1] for row in dit_fea['sub_hours']]
        x_sub_hsh5s=[row[:-1] for row in dit_fea['sub_hsh5s']]
        x_rec_poi_ints=[row[:-1] for row in dit_fea['rec_poi_ints']]
        x_rec_cat_ints=[row[:-1] for row in dit_fea['rec_cat_ints']]
        x_rec_delta_ts=[row for row in dit_fea['rec_delta_ts']]
        x_rec_delta_ds=[row for row in dit_fea['rec_delta_ds']]
        x_rec_months=[row[:-1] for row in dit_fea['rec_months']]
        x_rec_hours=[row[:-1] for row in dit_fea['rec_hours']]
        x_rec_hsh5s=[row[:-1] for row in dit_fea['rec_hsh5s']]
        x_second_importances=[row [:-1] for row in dit_fea['second_importances']]
        xy_lens=subtract_one_from_list(lens)
        x_rec_lens=[row [:-1] for row in dit_fea['rec_lens']]
        x_seqs_cols=['x_uids','xy_uids_extend_timesteps','x_poi_ints','x_cat_ints','x_second_ints','x_months','x_days','x_hours','x_hsh5s','x_sub_days','x_sub_months','x_sub_hours','x_sub_hsh5s','x_rec_poi_ints',\
                'x_rec_cat_ints','x_rec_delta_ts','x_rec_delta_ds','x_rec_months','x_rec_hours','x_rec_hsh5s','x_second_importances','xy_lens','x_rec_lens']
        x_seqs=[uids,xy_uids_extend_timesteps,x_poi_ints,x_cat_ints,x_second_ints,x_months,x_days,x_hours,x_hsh5s,x_sub_days,x_sub_months,x_sub_hours,x_sub_hsh5s,x_rec_poi_ints,\
                x_rec_cat_ints,x_rec_delta_ts,x_rec_delta_ds,x_rec_months,x_rec_hours,x_rec_hsh5s,x_second_importances,xy_lens,x_rec_lens]
        dit_x={k:v for k,v in zip(x_seqs_cols,x_seqs)}

        #y只需要poi_int做标签，poi_info(poi_int,cat_int),context(month,day,hour,geohsh)做key
        y_poi_ints = [row[1:] for row in dit_fea['poi_ints']]
        y_cat_ints=[row[1:] for row in dit_fea['cat_ints']]
        y_second_ints=[row[1:] for row in dit_fea['second_ints']]
        y_months=[row[1:] for row in dit_fea['months']]
        y_days = [row[1:] for row in dit_fea['days']]
        y_hours = [row[1:] for row in dit_fea['hours']]
        y_hsh5s = [row[1:] for row in dit_fea['hsh5s']]
        # y_lens=[row[1:] for row in dit_fea['lens']]
        y_seqs=[uids,y_poi_ints,y_cat_ints,y_second_ints,y_months,y_days,y_hours,y_hsh5s]
        y_seqs_cols=['y_uids','y_poi_ints','y_cat_ints','y_second_ints','y_months','y_days','y_hours','y_hsh5s']
        dit_y={k:v for k,v in zip(y_seqs_cols,y_seqs)}
        return dit_x,dit_y
    dit_x,dit_y=fun_make_xy(dit_fea)



    #划分训练测试
    def make_timestep(*data):
        data_res=[]
        for d in data:
            res=[]
            for u in d:
                u_t=[]
                for i in range(len(u)):
                    u_t.append(u[:i+1])
                res.append(u_t)
            data_res.append(res)
        return data_res

    def fun_split_train(dit_x,dit_y):
        tra_xy_uids_extend_timesteps=[row[:-1] for row in dit_x['xy_uids_extend_timesteps']]

        tra_x_poi_ints=[row[:-1] for row in dit_x['x_poi_ints']]
        tra_x_cat_ints=[row[:-1] for row in dit_x['x_cat_ints']]
        tra_x_second_ints=[row[:-1] for row in dit_x['x_second_ints']]
        tra_x_months=[row[:-1] for row in dit_x['x_months']]
        tra_x_days=[row[:-1] for row in dit_x['x_days']]
        tra_x_hours=[row[:-1] for row in dit_x['x_hours']]
        tra_x_hsh5s=[row[:-1] for row in dit_x['x_hsh5s']]
        tra_x_sub_days=[row[:-1] for row in dit_x['x_sub_days']]
        tra_x_sub_months=[row[:-1] for row in dit_x['x_sub_months']]
        tra_x_sub_hours=[row[:-1] for row in dit_x['x_sub_hours']]
        tra_x_sub_hsh5s=[row[:-1] for row in dit_x['x_sub_hsh5s']]
        tra_x_second_importances=[row[:-1] for row in dit_x['x_second_importances']]
        tra_x_poi_ints,tra_x_cat_ints,tra_x_months,tra_x_days,tra_x_hours,tra_x_hsh5s,tra_x_sub_days,tra_x_sub_months,tra_x_sub_hours,tra_x_sub_hsh5s=\
        make_timestep(tra_x_poi_ints,tra_x_cat_ints,tra_x_months,tra_x_days,tra_x_hours,tra_x_hsh5s,tra_x_sub_days,tra_x_sub_months,tra_x_sub_hours,tra_x_sub_hsh5s)
        
        tra_x_rec_poi_ints=[row[:-1] for row in dit_x['x_rec_poi_ints']]
        tra_x_rec_cat_ints=[row[:-1] for row in dit_x['x_rec_cat_ints']]
        tra_x_rec_delta_ts=[row[:-1] for row in dit_x['x_rec_delta_ts']]
        tra_x_rec_delta_ds=[row[:-1] for row in dit_x['x_rec_delta_ds']]
        tra_x_rec_months=[row[:-1] for row in dit_x['x_rec_months']]
        tra_x_rec_hours=[row[:-1] for row in dit_x['x_rec_hours']]
        tra_x_rec_hsh5s=[row[:-1] for row in dit_x['x_rec_hsh5s']]
        
        tra_xy_lens=subtract_one_from_list(dit_x['xy_lens'])#(user)
        tra_x_rec_lens=[row[:-1] for row in dit_x['x_rec_lens']]
        tra_x_seqs_cols=['tra_x_uids','tra_xy_uids_extend_timesteps','tra_x_poi_ints','tra_x_cat_ints','tra_x_second_ints','tra_x_months','tra_x_days','tra_x_hours','tra_x_hsh5s','tra_x_sub_days','tra_x_sub_months','tra_x_sub_hours','tra_x_sub_hsh5s','tra_x_rec_poi_ints',\
                'tra_x_rec_cat_ints','tra_x_rec_delta_ts','tra_x_rec_delta_ds','tra_x_rec_months','tra_x_rec_hours','tra_x_rec_hsh5s','tra_x_second_importances','tra_xy_lens','tra_x_rec_lens']
        tra_x_seqs=[uids,tra_xy_uids_extend_timesteps,tra_x_poi_ints,tra_x_cat_ints,tra_x_second_ints,tra_x_months,tra_x_days,tra_x_hours,tra_x_hsh5s,tra_x_sub_days,tra_x_sub_months,tra_x_sub_hours,tra_x_sub_hsh5s,tra_x_rec_poi_ints,\
                tra_x_rec_cat_ints,tra_x_rec_delta_ts,tra_x_rec_delta_ds,tra_x_rec_months,tra_x_rec_hours,tra_x_rec_hsh5s,tra_x_second_importances,tra_xy_lens,tra_x_rec_lens]
        tra_dit_x={k:v for k,v in zip(tra_x_seqs_cols,tra_x_seqs)}

        tra_y_poi_ints = [row[:-1] for row in dit_y['y_poi_ints']]
        tra_y_cat_ints=[row[:-1] for row in dit_y['y_cat_ints']]
        tra_y_second_ints=[row[:-1] for row in dit_y['y_second_ints']]
        tra_y_months = [row[:-1] for row in dit_y['y_months']]
        tra_y_days = [row[:-1] for row in dit_y['y_days']]
        tra_y_hours = [row[:-1] for row in dit_y['y_hours']]
        tra_y_hsh5s = [row[:-1] for row in dit_y['y_hsh5s']]
        tra_y_seqs=[uids,tra_y_poi_ints,tra_y_cat_ints,tra_y_second_ints,tra_y_months,tra_y_days,tra_y_hours,tra_y_hsh5s]
        tra_y_seqs_cols=['tra_y_uids','tra_y_poi_ints','tra_y_cat_ints','tra_y_second_ints','tra_y_months','tra_y_days','tra_y_hours','tra_y_hsh5s']
        tra_dit_y={k:v for k,v in zip(tra_y_seqs_cols,tra_y_seqs)}
        return tra_dit_x,tra_dit_y

    def fun_split_test(dit_x,dit_y):
        tes_xy_uids_extend_timesteps=[[row[-1]] for row in dit_x['xy_uids_extend_timesteps']]
        tes_x_poi_ints=[[row] for row in dit_x['x_poi_ints']]
        tes_x_cat_ints=[[row] for row in dit_x['x_cat_ints']]
        tes_x_second_ints=[[row] for row in dit_x['x_second_ints']]
        tes_x_months=[[row] for row in dit_x['x_months']]
        tes_x_days=[[row] for row in dit_x['x_days']]
        tes_x_hours=[[row] for row in dit_x['x_hours']]
        tes_x_hsh5s=[[row] for row in dit_x['x_hsh5s']]
        tes_x_sub_days=[[row] for row in dit_x['x_sub_days']]
        tes_x_sub_months=[[row] for row in dit_x['x_sub_months']]
        tes_x_sub_hours=[[row] for row in dit_x['x_sub_hours']]
        tes_x_sub_hsh5s=[[row] for row in dit_x['x_sub_hsh5s']]
        tes_x_second_importances=[[row] for row in dit_x['x_second_importances']]
        # tes_x_poi_ints,tes_x_cat_ints,tes_x_days,tes_x_hours,tes_x_hsh5s,tes_x_sub_days=\
        # make_timestep(tes_x_poi_ints,tes_x_cat_ints,tes_x_days,tes_x_hours,tes_x_hsh5s,tes_x_sub_days)

        tes_x_rec_poi_ints=[[row[-1]] for row in dit_x['x_rec_poi_ints']]
        tes_x_rec_cat_ints=[[row[-1]] for row in dit_x['x_rec_cat_ints']]
        tes_x_rec_delta_ts=[[row[-1]] for row in dit_x['x_rec_delta_ts']]
        tes_x_rec_delta_ds=[[row[-1]] for row in dit_x['x_rec_delta_ds']]
        tes_x_rec_months=[[row[-1]] for row in dit_x['x_rec_months']]
        tes_x_rec_hours=[[row[-1]] for row in dit_x['x_rec_hours']]
        tes_x_rec_hsh5s=[[row[-1]] for row in dit_x['x_rec_hsh5s']]
        tes_x_rec_lens=[[row[-1]] for row in dit_x['x_rec_lens']]
        tes_xy_lens=dit_x['xy_lens']
        tes_x_seqs_cols=['tes_x_uids','tes_xy_uids_extend_timesteps','tes_x_poi_ints','tes_x_cat_ints','tes_x_second_ints','tes_x_months','tes_x_days','tes_x_hours','tes_x_hsh5s','tes_x_sub_days','tes_x_sub_months','tes_x_sub_hours','tes_x_sub_hsh5s','tes_x_rec_poi_ints',\
                'tes_x_rec_cat_ints','tes_x_rec_delta_ts','tes_x_rec_delta_ds','tes_x_rec_months','tes_x_rec_hours','tes_x_rec_hsh5s','tes_x_second_importances','tes_xy_lens','tes_x_rec_lens']
        tes_x_seqs=[uids,tes_xy_uids_extend_timesteps,tes_x_poi_ints,tes_x_cat_ints,tes_x_second_ints,tes_x_months,tes_x_days,tes_x_hours,tes_x_hsh5s,tes_x_sub_days,tes_x_sub_months,tes_x_sub_hours,tes_x_sub_hsh5s,tes_x_rec_poi_ints,\
                tes_x_rec_cat_ints,tes_x_rec_delta_ts,tes_x_rec_delta_ds,tes_x_rec_months,tes_x_rec_hours,tes_x_rec_hsh5s,tes_x_second_importances,tes_xy_lens,tes_x_rec_lens]
        tes_dit_x={k:v for k,v in zip(tes_x_seqs_cols,tes_x_seqs)}

        tes_y_poi_ints = [[row[-1]] for row in dit_y['y_poi_ints']]
        tes_y_cat_ints=[[row[-1]] for row in dit_y['y_cat_ints']]
        tes_y_second_ints=[[row[-1] for row in dit_y['y_second_ints']]]
        tes_y_months = [[row[-1]] for row in dit_y['y_months']]
        tes_y_days = [[row[-1]] for row in dit_y['y_days']]
        tes_y_hours = [[row[-1]] for row in dit_y['y_hours']]
        tes_y_hsh5s = [[row[-1]] for row in dit_y['y_hsh5s']]
        tes_y_seqs=[uids,tes_y_poi_ints,tes_y_cat_ints,tes_y_second_ints,tes_y_months,tes_y_days,tes_y_hours,tes_y_hsh5s]
        tes_y_seqs_cols=['tes_y_uids','tes_y_poi_ints','tes_y_cat_ints','tes_y_second_ints','tes_y_months','tes_y_days','tes_y_hours','tes_y_hsh5s']
        tes_dit_y={k:v for k,v in zip(tes_y_seqs_cols,tes_y_seqs)}
        return tes_dit_x,tes_dit_y

    tra_dit_x,tra_dit_y=fun_split_train(dit_x,dit_y)
    tes_dit_x,tes_dit_y=fun_split_test(dit_x,dit_y)
    

    #负采样
    #1.定义查找函数
    #1.1poiint2catint
    def fun_poiint2catint(hist):
            dit_poiint2catint={}
            pois=hist['poi_int']
            cats=hist['category_int']
            print("poi_int: {}".format(len(pois)))
            print("category_int : {}".format(len(cats)))
            for poi,cat in zip(pois,cats):
                dit_poiint2catint.update({poi:cat})
            print("category_int : {}".format(len(dit_poiint2catint)))
            return dit_poiint2catint
    #1.2poiint2secondint
    def fun_poiint2secondint(hist):
        dit_poiint2secondint={}
        pois=hist['poi_int']
        second_ints=hist['second_int']
        for poi ,second in zip(pois,second_ints):
            dit_poiint2secondint.update({poi:second})
        return dit_poiint2secondint
    #2.获取负采样
    def fun_negs(neg_num,hist,dit_fea):
        dit_poiint2catint=fun_poiint2catint(hist)
        dit_poiint2secondint=fun_poiint2secondint(hist)
        pois=dit_fea['poi_ints']#(user,seq)
        pois_step=tra_dit_x['tra_xy_lens']
        negs_candi=hist['poi_int']
        poi_num=negs_candi.max()
        neg_pois=[]
        neg_cats=[]#(us,seq,neg_num)
        neg_seconds=[]
        for ui,u in enumerate(pois_step):#u 每个用户的打卡列表
            u_neg_poi_ints=[]
            u_neg_cat_ints=[]
            u_neg_second_ints=[]
            for i,e in enumerate(range(u)):
                each_step_pois=[]
                each_step_cats=[]
                each_step_seconds=[]
                for k in range(neg_num):
                    j=random.randint(0,poi_num)#闭区间
                    while j in pois[ui]:
                        j=random.randint(0,poi_num)
                    each_step_pois+=[j]
                    j_cat=dit_poiint2catint[j]
                    j_second=dit_poiint2secondint[j]
                    each_step_cats+=[j_cat]
                    each_step_seconds+=[j_second]
                u_neg_poi_ints.append(each_step_pois)
                u_neg_cat_ints.append(each_step_cats)
                u_neg_second_ints.append(each_step_seconds)
            neg_pois.append(u_neg_poi_ints)#(us,timestep^,20)
            neg_cats.append(u_neg_cat_ints)
            neg_seconds.append(u_neg_second_ints)
        neg_seqs=[uids,neg_pois,neg_cats,neg_seconds]
        neg_cols=['tra_neg_uids','tra_neg_pois','tra_neg_cats','tra_neg_seconds'] 
        dit_neg={k:v for k,v in zip(neg_cols,neg_seqs)}
        return dit_neg,dit_poiint2catint,dit_poiint2secondint

    #3.将全部poi作为负采样
    def fun_negs_all(num_poi,hist,dit_fea):  
        dit_poiint2catint=fun_poiint2catint(hist)
        dit_poiint2secondint=fun_poiint2secondint(hist)
        print("each_step_pois : {}".format(num_poi))
        print("dit_poiint2catint : {}".format(len(dit_poiint2catint)))
        posi_poi=tra_dit_y['tra_y_poi_ints']#每个时间步的y（user，timestep*)
        neg_pois=[]
        neg_cats=[]
        neg_seconds=[]
        for ui,u in enumerate(posi_poi):#生成（user，timestep，3905）
            u_neg_poi_ints=[]
            u_neg_cat_ints=[]
            u_neg_second_ints=[]
            for i,y in enumerate(u):
                each_step_pois=list(range(num_poi))
                each_step_cats=[]
                each_step_seconds=[]
                each_step_pois.remove(y)
                #处理每个y对应的cat
                for p in each_step_pois:
                    each_step_cats+=[dit_poiint2catint[p]]
                    each_step_seconds+=[dit_poiint2secondint[p]]
                u_neg_poi_ints.append(each_step_pois)
                u_neg_cat_ints.append(each_step_cats)
                u_neg_second_ints.append(each_step_seconds)
            neg_pois.append(u_neg_poi_ints)
            neg_cats.append(u_neg_cat_ints)
            neg_seconds.append(u_neg_second_ints)
        neg_seqs=[uids,neg_pois,neg_cats,neg_seconds]
        neg_cols=['tra_neg_uids','tra_neg_pois','tra_neg_cats','tra_neg_seconds']
        dit_neg={k:v for k,v in zip(neg_cols,neg_seqs)}
        return dit_neg,dit_poiint2catint,dit_poiint2secondint

    #4.测试时的负采样
    def fun_tes_neg(tes_dit_y,poi_size,hist):
        '''
        tes_dit_y:(user,1),list
        从所有的候选poi中取数据，当作最后一个时间步的数据，所以poi和cat直接从range生成即可，second_ints同理
        只需要保存second_int即可，时间步在处理数据时按同样的dataload shuffle，并根据neg中second_ints找到对应种类的重要性值
        '''
        dit_poiint2catint=fun_poiint2catint(hist)
        dit_poiint2secondint=fun_poiint2secondint(hist)
        tes_neg_pois=[]
        tes_neg_cats=[]
        tes_neg_seconds=[]
        for u in tes_dit_y['tes_y_poi_ints']:
            x=u[0]#x为最后一个时间步的打卡
            p=[i for i in range(poi_size) if i!=x]#将非最后一个时间步打卡poi保存
            c=[dit_poiint2catint[i] for i in range(poi_size) if i!=x]
            s=[dit_poiint2secondint[i] for i in range(poi_size) if i!=x]
            tes_neg_pois.append([p])
            tes_neg_cats.append([c])
            tes_neg_seconds.append([s])
        tes_neg_seqs=[tes_neg_pois,tes_neg_cats,tes_neg_seconds]
        tes_neg_cols=['tes_neg_pois','tes_neg_cats','tes_neg_seconds']
        tes_dit_neg={k:v for k,v in zip(tes_neg_cols,tes_neg_seqs)}
        return tes_dit_neg
    
    # tra_dit_neg,dit_poiint2catint,dit_poiint2secondint=fun_negs(neg_num,hist,dit_fea)
    tra_dit_neg,dit_poiint2catint,dit_poiint2secondint=fun_negs_all(poi_size,hist,dit_fea)

    tes_dit_neg=fun_tes_neg(tes_dit_y,poi_size,hist)#poi_size=3906,数量，生成【0，3905】

    #保存tra_dit_xy,tes_dit_xy,dit_poiint2catint,dit_neg,dit_fea
    dit_data_seqs=[tra_dit_x,tra_dit_y,tes_dit_x,tes_dit_y,tra_dit_neg,tes_dit_neg,dit_fea,dit_poiint2catint,dit_poiint2secondint]
    dit_data_cols=['tra_dit_x','tra_dit_y','tes_dit_x','tes_dit_y','tra_dit_neg','tes_dit_neg','dit_fea','dit_poiint2catint','dit_poiint2secondint']
    dit_data={k:v for k,v in zip(dit_data_cols,dit_data_seqs)}
    with open(data_to,'wb') as f:
        pickle.dump(dit_data,f)


def fun_load_data_2(cols, rec_num, data_fr, neg_num, data_to, span_month, poi_size):
    print('\nOriginal data...')
    hist = pd.read_csv(data_fr, sep='\t', names=cols)
    hist['sub_day'] = hist['sub_day'].apply(literal_eval)
    hist['sub_month'] = hist['sub_month'].apply(literal_eval)
    hist['sub_hour'] = hist['sub_hour'].apply(literal_eval)
    hist['sub_hsh5'] = hist['sub_hsh5'].apply(literal_eval)

    gp = hist.groupby('user_id')
    uids, uids_extend_timesteps = [], []
    poi_ints, cat_ints, second_ints, second_hexs = [], [], [], []
    days, hours, months, hsh5s, sub_days, sub_months, sub_hours, sub_hsh5s = [], [], [], [], [], [], [], []
    localtimes, latis, longis = [], [], []
    for user_id, hist_group in gp:
        uids.append(list(hist_group['user_id'])[0])  # int64
        uids_extend_timesteps.append(list(hist_group['user_id']))
        poi_ints.append(list(hist_group['poi_int']))  # int64 (user,timestep^)
        cat_ints.append(list(hist_group['category_int']))  # int64
        # second_hexs.append(list(hist_group['second_hex']))
        # second_ints.append(list(hist_group['second_int']))  # int64
        days.append(list(hist_group['day']))  # int64
        hours.append(list(hist_group['hour']))  # int64
        months.append(list(hist_group['month']))
        hsh5s.append(list(hist_group['hsh_int']))  # int64
        sub_days.append(list(hist_group['sub_day']))  # int64 list
        sub_months.append(list(hist_group['sub_month']))
        sub_hours.append(list(hist_group['sub_hour']))
        sub_hsh5s.append(list(hist_group['sub_hsh5']))
        localtimes.append(list(hist_group['local_time']))  # str array
        latis.append(list(hist_group['latitude']))
        longis.append(list(hist_group['longitude']))
        # print("poi ints : {}".format(len(poi_ints)))
        # print("cat_ints : {}".format(len(cat_ints)))
        # print("cat_ints : {}".format(len(cat_ints)))
        # print("cat_ints : {}".format(len(cat_ints)))
        # print("cat_ints : {}".format(len(cat_ints)))
        # print("cat_ints : {}".format(len(cat_ints)))
        assert len(poi_ints) == len(cat_ints) == len(days) == len(hours) == len(hsh5s) == len(
            localtimes) == len(latis) == len(longis)

    # 做delta
    delta_ts = []
    for u_localtime in localtimes:
        u_delta_t = []
        u_localtime = pd.to_datetime(u_localtime).tolist()
        for i in range(len(u_localtime) - 1):
            u_delta_t.append((u_localtime[i + 1] - u_localtime[i]).total_seconds())
        delta_ts.append(u_delta_t)
    delta_ds = []
    for u_lati, u_longi in zip(latis, longis):
        u_dist = []
        u_loc = [(lati, longi) for lati, longi in zip(u_lati, u_longi)]
        for i in range(len(u_loc) - 1):
            u_dist.append(geodesic(u_loc[i + 1], u_loc[i]).kilometers * 1000)
        delta_ds.append(u_dist)

    # 做rec
    def fun_rec(lst):
        rec_datas = []  # 所有用户，所有rec
        for u in lst:
            u_rec = [value[-rec_num:] if len(value) >= rec_num else value for \
                     value in [u[:i + 1] for i in range(len(u))]]
            rec_datas.append(u_rec)
        return rec_datas

    rec_poi_ints = fun_rec(poi_ints)
    rec_cat_ints = fun_rec(cat_ints)
    rec_delta_ts = fun_rec(delta_ts)
    rec_delta_ds = fun_rec(delta_ds)
    rec_hsh5s = fun_rec(hsh5s)
    rec_months = fun_rec(months)
    rec_hours = fun_rec(hours)

    # 计算lens,rec_lens
    # lens,用poi_ints(users,timestep^)
    def compute_lens(input_list):
        def recursive_min_dimension_lens(lst):
            if isinstance(lst[0], list):
                return [recursive_min_dimension_lens(sublist) for sublist in lst]
            else:
                return len(lst)

        return recursive_min_dimension_lens(input_list)

    lens = compute_lens(poi_ints)
    rec_lens = compute_lens(rec_poi_ints)

    # 划分xy
    # 不再需要具体localtime，lati，longi
    seqs_cols = ['uids', 'uids_extend_timesteps', 'poi_ints', 'cat_ints', 'months', 'days', 'hours',
                 'hsh5s', 'sub_days', 'sub_months', 'sub_hours', 'sub_hsh5s', 'rec_poi_ints', 'rec_cat_ints',
                 'rec_delta_ts', 'rec_delta_ds', 'rec_months', 'rec_hours', 'rec_hsh5s',  'lens', 'rec_lens']
    seqs = [uids, uids_extend_timesteps, poi_ints, cat_ints, months, days, hours,
            # (user,seq),(user,seq,ren_num)
            hsh5s, sub_days, sub_months, sub_hours, sub_hsh5s, rec_poi_ints, rec_cat_ints, rec_delta_ts, rec_delta_ds,
            rec_months, rec_hours, rec_hsh5s, lens, rec_lens]
    dit_fea = {k: v for k, v in zip(seqs_cols, seqs)}

    def subtract_one_from_list(input_list):
        def recursive_subtract(lst):
            if isinstance(lst[0], list):
                # 如果是多维列表，递归处理每个子列表
                return [recursive_subtract(sublist) for sublist in lst]
            else:
                # 对最小维度的元素执行减1操作
                return [element - 1 for element in lst]

        return recursive_subtract(input_list)

    def fun_make_xy(dit_fea):
        xy_uids_extend_timesteps = [row[:-1] for row in dit_fea['uids_extend_timesteps']]
        x_poi_ints = [row[:-1] for row in dit_fea['poi_ints']]  # (users,seq)
        x_cat_ints = [row[:-1] for row in dit_fea['cat_ints']]
        # x_second_ints = [row[:-1] for row in dit_fea['second_ints']]
        x_months = [row[:-1] for row in dit_fea['months']]
        x_days = [row[:-1] for row in dit_fea['days']]
        x_hours = [row[:-1] for row in dit_fea['hours']]
        x_hsh5s = [row[:-1] for row in dit_fea['hsh5s']]
        x_sub_days = [row[:-1] for row in dit_fea['sub_days']]  # (users,seq,7)
        x_sub_months = [row[:-1] for row in dit_fea['sub_months']]
        x_sub_hours = [row[:-1] for row in dit_fea['sub_hours']]
        x_sub_hsh5s = [row[:-1] for row in dit_fea['sub_hsh5s']]
        x_rec_poi_ints = [row[:-1] for row in dit_fea['rec_poi_ints']]
        x_rec_cat_ints = [row[:-1] for row in dit_fea['rec_cat_ints']]
        x_rec_delta_ts = [row for row in dit_fea['rec_delta_ts']]
        x_rec_delta_ds = [row for row in dit_fea['rec_delta_ds']]
        x_rec_months = [row[:-1] for row in dit_fea['rec_months']]
        x_rec_hours = [row[:-1] for row in dit_fea['rec_hours']]
        x_rec_hsh5s = [row[:-1] for row in dit_fea['rec_hsh5s']]
        xy_lens = subtract_one_from_list(lens)
        x_rec_lens = [row[:-1] for row in dit_fea['rec_lens']]
        x_seqs_cols = ['x_uids', 'xy_uids_extend_timesteps', 'x_poi_ints', 'x_cat_ints', 'x_months',
                       'x_days', 'x_hours', 'x_hsh5s', 'x_sub_days', 'x_sub_months', 'x_sub_hours', 'x_sub_hsh5s',
                       'x_rec_poi_ints', \
                       'x_rec_cat_ints', 'x_rec_delta_ts', 'x_rec_delta_ds', 'x_rec_months', 'x_rec_hours',
                       'x_rec_hsh5s', 'xy_lens', 'x_rec_lens']
        x_seqs = [uids, xy_uids_extend_timesteps, x_poi_ints, x_cat_ints, x_months, x_days, x_hours,
                  x_hsh5s, x_sub_days, x_sub_months, x_sub_hours, x_sub_hsh5s, x_rec_poi_ints, \
                  x_rec_cat_ints, x_rec_delta_ts, x_rec_delta_ds, x_rec_months, x_rec_hours, x_rec_hsh5s,
                   xy_lens, x_rec_lens]
        dit_x = {k: v for k, v in zip(x_seqs_cols, x_seqs)}

        # y只需要poi_int做标签，poi_info(poi_int,cat_int),context(month,day,hour,geohsh)做key
        y_poi_ints = [row[1:] for row in dit_fea['poi_ints']]
        y_cat_ints = [row[1:] for row in dit_fea['cat_ints']]
        # y_second_ints = [row[1:] for row in dit_fea['second_ints']]
        y_months = [row[1:] for row in dit_fea['months']]
        y_days = [row[1:] for row in dit_fea['days']]
        y_hours = [row[1:] for row in dit_fea['hours']]
        y_hsh5s = [row[1:] for row in dit_fea['hsh5s']]
        # y_lens=[row[1:] for row in dit_fea['lens']]
        y_seqs = [uids, y_poi_ints, y_cat_ints, y_months, y_days, y_hours, y_hsh5s]
        y_seqs_cols = ['y_uids', 'y_poi_ints', 'y_cat_ints', 'y_months', 'y_days', 'y_hours',
                       'y_hsh5s']
        dit_y = {k: v for k, v in zip(y_seqs_cols, y_seqs)}
        return dit_x, dit_y

    dit_x, dit_y = fun_make_xy(dit_fea)

    # 划分训练测试
    def make_timestep(*data):
        data_res = []
        for d in data:
            res = []
            for u in d:
                u_t = []
                for i in range(len(u)):
                    u_t.append(u[:i + 1])
                res.append(u_t)
            data_res.append(res)
        return data_res

    def fun_split_train(dit_x, dit_y):
        tra_xy_uids_extend_timesteps = [row[:-1] for row in dit_x['xy_uids_extend_timesteps']]

        tra_x_poi_ints = [row[:-1] for row in dit_x['x_poi_ints']]
        tra_x_cat_ints = [row[:-1] for row in dit_x['x_cat_ints']]
        # tra_x_second_ints = [row[:-1] for row in dit_x['x_second_ints']]
        tra_x_months = [row[:-1] for row in dit_x['x_months']]
        tra_x_days = [row[:-1] for row in dit_x['x_days']]
        tra_x_hours = [row[:-1] for row in dit_x['x_hours']]
        tra_x_hsh5s = [row[:-1] for row in dit_x['x_hsh5s']]
        tra_x_sub_days = [row[:-1] for row in dit_x['x_sub_days']]
        tra_x_sub_months = [row[:-1] for row in dit_x['x_sub_months']]
        tra_x_sub_hours = [row[:-1] for row in dit_x['x_sub_hours']]
        tra_x_sub_hsh5s = [row[:-1] for row in dit_x['x_sub_hsh5s']]
        tra_x_poi_ints, tra_x_cat_ints, tra_x_months, tra_x_days, tra_x_hours, tra_x_hsh5s, tra_x_sub_days, tra_x_sub_months, tra_x_sub_hours, tra_x_sub_hsh5s = \
            make_timestep(tra_x_poi_ints, tra_x_cat_ints, tra_x_months, tra_x_days, tra_x_hours, tra_x_hsh5s,
                          tra_x_sub_days, tra_x_sub_months, tra_x_sub_hours, tra_x_sub_hsh5s)

        tra_x_rec_poi_ints = [row[:-1] for row in dit_x['x_rec_poi_ints']]
        tra_x_rec_cat_ints = [row[:-1] for row in dit_x['x_rec_cat_ints']]
        tra_x_rec_delta_ts = [row[:-1] for row in dit_x['x_rec_delta_ts']]
        tra_x_rec_delta_ds = [row[:-1] for row in dit_x['x_rec_delta_ds']]
        tra_x_rec_months = [row[:-1] for row in dit_x['x_rec_months']]
        tra_x_rec_hours = [row[:-1] for row in dit_x['x_rec_hours']]
        tra_x_rec_hsh5s = [row[:-1] for row in dit_x['x_rec_hsh5s']]

        tra_xy_lens = subtract_one_from_list(dit_x['xy_lens'])  # (user)
        tra_x_rec_lens = [row[:-1] for row in dit_x['x_rec_lens']]
        tra_x_seqs_cols = ['tra_x_uids', 'tra_xy_uids_extend_timesteps', 'tra_x_poi_ints', 'tra_x_cat_ints',
                           'tra_x_months', 'tra_x_days', 'tra_x_hours', 'tra_x_hsh5s',
                           'tra_x_sub_days', 'tra_x_sub_months', 'tra_x_sub_hours', 'tra_x_sub_hsh5s',
                           'tra_x_rec_poi_ints', \
                           'tra_x_rec_cat_ints', 'tra_x_rec_delta_ts', 'tra_x_rec_delta_ds', 'tra_x_rec_months',
                           'tra_x_rec_hours', 'tra_x_rec_hsh5s',  'tra_xy_lens',
                           'tra_x_rec_lens']
        tra_x_seqs = [uids, tra_xy_uids_extend_timesteps, tra_x_poi_ints, tra_x_cat_ints,
                      tra_x_months, tra_x_days, tra_x_hours, tra_x_hsh5s, tra_x_sub_days, tra_x_sub_months,
                      tra_x_sub_hours, tra_x_sub_hsh5s, tra_x_rec_poi_ints, \
                      tra_x_rec_cat_ints, tra_x_rec_delta_ts, tra_x_rec_delta_ds, tra_x_rec_months, tra_x_rec_hours,
                      tra_x_rec_hsh5s,  tra_xy_lens, tra_x_rec_lens]
        tra_dit_x = {k: v for k, v in zip(tra_x_seqs_cols, tra_x_seqs)}

        tra_y_poi_ints = [row[:-1] for row in dit_y['y_poi_ints']]
        tra_y_cat_ints = [row[:-1] for row in dit_y['y_cat_ints']]
        # tra_y_second_ints = [row[:-1] for row in dit_y['y_second_ints']]
        tra_y_months = [row[:-1] for row in dit_y['y_months']]
        tra_y_days = [row[:-1] for row in dit_y['y_days']]
        tra_y_hours = [row[:-1] for row in dit_y['y_hours']]
        tra_y_hsh5s = [row[:-1] for row in dit_y['y_hsh5s']]
        tra_y_seqs = [uids, tra_y_poi_ints, tra_y_cat_ints, tra_y_months, tra_y_days, tra_y_hours,
                      tra_y_hsh5s]
        tra_y_seqs_cols = ['tra_y_uids', 'tra_y_poi_ints', 'tra_y_cat_ints', 'tra_y_months',
                           'tra_y_days', 'tra_y_hours', 'tra_y_hsh5s']
        tra_dit_y = {k: v for k, v in zip(tra_y_seqs_cols, tra_y_seqs)}
        return tra_dit_x, tra_dit_y

    def fun_split_test(dit_x, dit_y):
        tes_xy_uids_extend_timesteps = [[row[-1]] for row in dit_x['xy_uids_extend_timesteps']]
        tes_x_poi_ints = [[row] for row in dit_x['x_poi_ints']]
        tes_x_cat_ints = [[row] for row in dit_x['x_cat_ints']]
        # tes_x_second_ints = [[row] for row in dit_x['x_second_ints']]
        tes_x_months = [[row] for row in dit_x['x_months']]
        tes_x_days = [[row] for row in dit_x['x_days']]
        tes_x_hours = [[row] for row in dit_x['x_hours']]
        tes_x_hsh5s = [[row] for row in dit_x['x_hsh5s']]
        tes_x_sub_days = [[row] for row in dit_x['x_sub_days']]
        tes_x_sub_months = [[row] for row in dit_x['x_sub_months']]
        tes_x_sub_hours = [[row] for row in dit_x['x_sub_hours']]
        tes_x_sub_hsh5s = [[row] for row in dit_x['x_sub_hsh5s']]
        # tes_x_poi_ints,tes_x_cat_ints,tes_x_days,tes_x_hours,tes_x_hsh5s,tes_x_sub_days=\
        # make_timestep(tes_x_poi_ints,tes_x_cat_ints,tes_x_days,tes_x_hours,tes_x_hsh5s,tes_x_sub_days)

        tes_x_rec_poi_ints = [[row[-1]] for row in dit_x['x_rec_poi_ints']]
        tes_x_rec_cat_ints = [[row[-1]] for row in dit_x['x_rec_cat_ints']]
        tes_x_rec_delta_ts = [[row[-1]] for row in dit_x['x_rec_delta_ts']]
        tes_x_rec_delta_ds = [[row[-1]] for row in dit_x['x_rec_delta_ds']]
        tes_x_rec_months = [[row[-1]] for row in dit_x['x_rec_months']]
        tes_x_rec_hours = [[row[-1]] for row in dit_x['x_rec_hours']]
        tes_x_rec_hsh5s = [[row[-1]] for row in dit_x['x_rec_hsh5s']]
        tes_x_rec_lens = [[row[-1]] for row in dit_x['x_rec_lens']]
        tes_xy_lens = dit_x['xy_lens']
        tes_x_seqs_cols = ['tes_x_uids', 'tes_xy_uids_extend_timesteps', 'tes_x_poi_ints', 'tes_x_cat_ints',
                           'tes_x_months', 'tes_x_days', 'tes_x_hours', 'tes_x_hsh5s',
                           'tes_x_sub_days', 'tes_x_sub_months', 'tes_x_sub_hours', 'tes_x_sub_hsh5s',
                           'tes_x_rec_poi_ints', \
                           'tes_x_rec_cat_ints', 'tes_x_rec_delta_ts', 'tes_x_rec_delta_ds', 'tes_x_rec_months',
                           'tes_x_rec_hours', 'tes_x_rec_hsh5s', 'tes_xy_lens',
                           'tes_x_rec_lens']
        tes_x_seqs = [uids, tes_xy_uids_extend_timesteps, tes_x_poi_ints, tes_x_cat_ints,
                      tes_x_months, tes_x_days, tes_x_hours, tes_x_hsh5s, tes_x_sub_days, tes_x_sub_months,
                      tes_x_sub_hours, tes_x_sub_hsh5s, tes_x_rec_poi_ints, \
                      tes_x_rec_cat_ints, tes_x_rec_delta_ts, tes_x_rec_delta_ds, tes_x_rec_months, tes_x_rec_hours,
                      tes_x_rec_hsh5s,  tes_xy_lens, tes_x_rec_lens]
        tes_dit_x = {k: v for k, v in zip(tes_x_seqs_cols, tes_x_seqs)}

        tes_y_poi_ints = [[row[-1]] for row in dit_y['y_poi_ints']]
        tes_y_cat_ints = [[row[-1]] for row in dit_y['y_cat_ints']]
        # tes_y_second_ints = [[row[-1] for row in dit_y['y_second_ints']]]
        tes_y_months = [[row[-1]] for row in dit_y['y_months']]
        tes_y_days = [[row[-1]] for row in dit_y['y_days']]
        tes_y_hours = [[row[-1]] for row in dit_y['y_hours']]
        tes_y_hsh5s = [[row[-1]] for row in dit_y['y_hsh5s']]
        tes_y_seqs = [uids, tes_y_poi_ints, tes_y_cat_ints, tes_y_months, tes_y_days, tes_y_hours,
                      tes_y_hsh5s]
        tes_y_seqs_cols = ['tes_y_uids', 'tes_y_poi_ints', 'tes_y_cat_ints', 'tes_y_months',
                           'tes_y_days', 'tes_y_hours', 'tes_y_hsh5s']
        tes_dit_y = {k: v for k, v in zip(tes_y_seqs_cols, tes_y_seqs)}
        return tes_dit_x, tes_dit_y

    tra_dit_x, tra_dit_y = fun_split_train(dit_x, dit_y)
    tes_dit_x, tes_dit_y = fun_split_test(dit_x, dit_y)

    # 负采样
    # 1.定义查找函数
    # 1.1poiint2catint
    def fun_poiint2catint(hist):
        dit_poiint2catint = {}
        pois = hist['poi_int']
        cats = hist['category_int']
        # print("poi_int: {}".format(len(pois)))
        # print("category_int : {}".format(len(cats)))
        for poi, cat in zip(pois, cats):
            dit_poiint2catint.update({poi: cat})
        # print("category_int : {}".format(len(dit_poiint2catint)))
        return dit_poiint2catint

    # 1.2poiint2secondint
    def fun_poiint2secondint(hist):
        dit_poiint2secondint = {}
        pois = hist['poi_int']
        # second_ints = hist['second_int']
        for poi, second in zip(pois, second_ints):
            dit_poiint2secondint.update({poi: second})
        return dit_poiint2secondint

    # 3.将全部poi作为负采样
    def fun_negs_all(num_poi, hist, dit_fea):
        dit_poiint2catint = fun_poiint2catint(hist)
        # dit_poiint2secondint = fun_poiint2secondint(hist)
        posi_poi = tra_dit_y['tra_y_poi_ints']  # 每个时间步的y（user，timestep*)
        neg_pois = []
        neg_cats = []
        neg_seconds = []
        for ui, u in enumerate(posi_poi):  # 生成（user，timestep，3905）
            u_neg_poi_ints = []
            u_neg_cat_ints = []
            u_neg_second_ints = []
            for i, y in enumerate(u):
                each_step_pois = list(range(num_poi))
                each_step_cats = []
                # each_step_seconds = []
                each_step_pois.remove(y)
                # print("each_step_pois : {}".format(num_poi))
                # print("dit_poiint2catint : {}".format(len(dit_poiint2catint)))
                # 处理每个y对应的cat
                for p in each_step_pois:
                    each_step_cats += [dit_poiint2catint[p]]
                    # each_step_seconds += [dit_poiint2secondint[p]]
                u_neg_poi_ints.append(each_step_pois)
                u_neg_cat_ints.append(each_step_cats)
                # u_neg_second_ints.append(each_step_seconds)
            neg_pois.append(u_neg_poi_ints)
            neg_cats.append(u_neg_cat_ints)
            # neg_seconds.append(u_neg_second_ints)
        neg_seqs = [uids, neg_pois, neg_cats]
        neg_cols = ['tra_neg_uids', 'tra_neg_pois', 'tra_neg_cats']
        dit_neg = {k: v for k, v in zip(neg_cols, neg_seqs)}
        return dit_neg, dit_poiint2catint

    # 4.测试时的负采样
    def fun_tes_neg(tes_dit_y, poi_size, hist):
        '''
        tes_dit_y:(user,1),list
        从所有的候选poi中取数据，当作最后一个时间步的数据，所以poi和cat直接从range生成即可，second_ints同理
        只需要保存second_int即可，时间步在处理数据时按同样的dataload shuffle，并根据neg中second_ints找到对应种类的重要性值
        '''
        dit_poiint2catint = fun_poiint2catint(hist)
        # dit_poiint2secondint = fun_poiint2secondint(hist)
        tes_neg_pois = []
        tes_neg_cats = []
        tes_neg_seconds = []
        for u in tes_dit_y['tes_y_poi_ints']:
            x = u[0]  # x为最后一个时间步的打卡
            p = [i for i in range(poi_size) if i != x]  # 将非最后一个时间步打卡poi保存
            c = [dit_poiint2catint[i] for i in range(poi_size) if i != x]
            # s = [dit_poiint2secondint[i] for i in range(poi_size) if i != x]
            tes_neg_pois.append([p])
            tes_neg_cats.append([c])
            # tes_neg_seconds.append([s])
        tes_neg_seqs = [tes_neg_pois, tes_neg_cats]
        tes_neg_cols = ['tes_neg_pois', 'tes_neg_cats']
        tes_dit_neg = {k: v for k, v in zip(tes_neg_cols, tes_neg_seqs)}
        return tes_dit_neg

    # tra_dit_neg,dit_poiint2catint,dit_poiint2secondint=fun_negs(neg_num,hist,dit_fea)
    tra_dit_neg, dit_poiint2catint = fun_negs_all(poi_size, hist, dit_fea)

    tes_dit_neg = fun_tes_neg(tes_dit_y, poi_size, hist)  # poi_size=170,数量，生成【0，170】

    # 保存tra_dit_xy,tes_dit_xy,dit_poiint2catint,dit_neg,dit_fea
    dit_data_seqs = [tra_dit_x, tra_dit_y, tes_dit_x, tes_dit_y, tra_dit_neg, tes_dit_neg, dit_fea, dit_poiint2catint,
                     ]
    dit_data_cols = ['tra_dit_x', 'tra_dit_y', 'tes_dit_x', 'tes_dit_y', 'tra_dit_neg', 'tes_dit_neg', 'dit_fea',
                     'dit_poiint2catint']
    dit_data = {k: v for k, v in zip(dit_data_cols, dit_data_seqs)}
    with open(data_to, 'wb') as f:
        pickle.dump(dit_data, f)


@exe_time
def load_data_main_nyc():
    rec_num=20
    #nyc
    data_to='/home/liuqiuyu/code/my_code/dataset/data_process.pkl'
    neg_num=20
    poi_size=3906
    data_fr='/home/liuqiuyu/code/my_code/dataset/dataset_TSMC2014_NYC_2_change_column.txt'
    dir_mapping_dic='/home/liuqiuyu/code/my_code/dataset/mapping_dic.pkl'
    dir_second_importances='/home/liuqiuyu/code/test_output/users_timesteps_importance.pkl'
    span_month=[4,5,6,7,8,9,10,11,12,1,2]
    fun_load_data(rec_num,data_fr,neg_num,data_to,dir_mapping_dic,dir_second_importances,span_month,poi_size)

@exe_time
def load_data_main_tky():
    #tky
    data_to='/home/liuqiuyu/code/my_code/dataset/data_process_tky.pkl'
    rec_num=20
    neg_num=7056
    poi_size=7057
    data_fr='/home/liuqiuyu/code/my_code/dataset/dataset_TSMC2014_TKY_2_change_column.txt'
    dir_mapping_dic='/home/liuqiuyu/code/my_code/dataset/mapping_dic_tky.pkl'
    dir_second_importances='/home/liuqiuyu/code/test_output/users_timesteps_importance.pkl'#因为不需要计算，所以不修改为tky
    span_month=[4,5,6,7,8,9,10,11,12,1,2]
    fun_load_data(rec_num,data_fr,neg_num,data_to,dir_mapping_dic,dir_second_importances,span_month,poi_size)


@exe_time
def load_data_main_cal():
    # cal
    cols = ['user_id', 'poi_hex', 'time_zone', 'latitude', 'longitude', 'category',
            'local_time', 'geohash5', 'datetime', 'day', 'hour', 'month', 'category_int',
            'poi_int', 'hsh_int', 'sub_day', 'sub_month', 'sub_hour', 'sub_hsh5']
    # data_fr = 'code/my_code/dataset/data/CAL/CAL_change_column.txt'
    # data_to = 'code/my_code/dataset/data/CAL/data_process_cal.pkl'
    data_fr = 'code/my_code/dataset/data/CAL/CAL_poi_change_column.txt'
    data_to = 'code/my_code/dataset/data/CAL/data_process_poi_cal.pkl'
    rec_num = 20
    neg_num = 7056
    poi_size = 170
    span_month = [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2]
    fun_load_data_2(cols, rec_num, data_fr, neg_num, data_to, span_month, poi_size)


@exe_time
def load_data_main_sin():
    # sin
    cols = ['user_id', 'poi_hex', 'time_zone', 'latitude', 'longitude', 'category',
            'local_time', 'geohash5', 'datetime', 'day', 'hour', 'month', 'category_int',
            'poi_int', 'hsh_int', 'sub_day', 'sub_month', 'sub_hour', 'sub_hsh5']
    # data_fr = 'code/my_code/dataset/data/SIN/SIN_change_column.txt'
    # data_to = 'code/my_code/dataset/data/SIN/data_process_sin.pkl'
    data_fr = 'code/my_code/dataset/data/SIN/SIN_user_change_column.txt'
    data_to = 'code/my_code/dataset/data/SIN/data_process_user_sin.pkl'
    # data_fr = 'code/my_code/dataset/data/SIN/SIN_poi_change_column.txt'
    # data_to = 'code/my_code/dataset/data/SIN/data_process_poi_sin.pkl'
    rec_num = 20
    neg_num = 7056
    poi_size = 3959
    span_month = [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2]
    fun_load_data_2(cols, rec_num, data_fr, neg_num, data_to, span_month, poi_size)


def count_feature():   
    # cols = ['user_id', 'poi_hex', 'time_zone', 'latitude', 'longitude', 'category',
    #         'local_time', 'geohash5', 'datetime', 'day', 'hour', 'month', 'category_int',
    #         'poi_int', 'hsh_int', 'sub_day', 'sub_month', 'sub_hour', 'sub_hsh5']
    # hist = pd.read_csv('/home/liuqiuyu/code/MCMG-main/dataset/TKY/TKY_checkin.csv', sep=',', names=cols)
    # cols = ['user_id','poi_hex','category_hex','category_name','latitude','longitude','timezone_offset','utc_time','r_latitude','r_longitude','RegionId']
    cols = ['user_id','poi_hex','category_hex','category','latitude','longitude','timezone_offset','utc_time']
    hist = pd.read_csv('/home/liuqiuyu/code/MCMG-main/dataset/TKY/TKY_checkin.csv', sep=',', names=cols)
    # cols = ['0', 'user_id', 'venue_id', 'time_zone', 'latitude', 'longitude', 'category', 'city', 'country_code',
    #         'time', 'date', 'local_time', 'l1_category', 'region_id', 'r_latitude', 'r_longitude',
    #         'checked_region', 'active_regionId_1', 'active_regionId_2', 'time_sort', 'local_time_true']
    # hist = pd.read_csv('/home/liuqiuyu/code/MCMG-main/dataset/SIN/SIN_checkin.csv', sep=',', names=cols)

    print(hist.head())
    print('去重后条目：', len(hist))
    print('去重后user数量：', len(hist.user_id.unique()))
    print("去重后poi数量：", len(hist.poi_hex.unique()))
    # print("去重后poi int数量：", len(hist.poi_int.unique()))
    print("去重后cat数量：", len(hist.category.unique()))
    print("去重后region数量：", len(hist.region_id.unique()))
    print("去重后local_time_true数量：", len(hist.local_time_true.unique()))
    # print("去重后hsh5数量：", len(hist.geohash5.unique()))
    # print("去重后month数量：", len(hist.month.unique()))
    # print("去重后day数量：", len(hist.day.unique()))
    # print("去重后hour数量：", len(hist.hour.unique()))
    # print("去重后r_latitude数量：", len(hist.r_latitude.unique()))
    # print(hist['r_latitude'].unique())
    # print("去重后r_longitude数量：", len(hist.r_longitude.unique()))
    # print(hist['r_longitude'].unique())

if '__main__'==__name__:
    count_feature()
    # load_data_main_cal()
    # load_data_main_sin()
    # load_data_main_nyc()
    # load_data_main_tky()