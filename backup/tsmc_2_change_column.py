import time
import os
import pandas as pd
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
import datetime
import pickle
import pygeohash as pgh
import numpy as np
import json5
import pytz
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
import sys
sys.path.append('/home/liuqiuyu/POI_OURS/utils')
from utils import exe_time


def fun_change_column(data_fr,data_to,dir_mapping_dic):
    data_dir=data_fr
    cols=['user_id', 'poi_hex', 'category_hex', 'category_name', 'latitude',
        'longitude', 'timezone_offset', 'utc_time']
    hist=pd.read_csv(data_dir,sep='\t',names=cols)


    hist['geohash5']=hist.apply(lambda x:pgh.encode(x.latitude,x.longitude,precision=5),axis=1)
    hist['datetime']=pd.to_datetime(hist.utc_time)
    time_zone=pytz.timezone('America/New_York')
    hist['localtime']=hist['datetime'].dt.tz_convert(time_zone)
    hist['day']=hist['localtime'].dt.dayofweek+1
    hist['hour']=hist['localtime'].dt.hour+1
    hist['month']=hist['localtime'].dt.month
    #将hex转换成int,并计算cat_hex2int字典
    hist.loc[hist.category_hex == '4e51a0c0bd41d3446defbb2e', 'category_hex'] = '4bf58dd8d48988d12d951735'
    hist['category_int']=hist['category_hex'].rank(method='dense').astype(int)-1
    hist['poi_int']=hist['poi_hex'].rank(method='dense').astype(int)-1
    hist['hsh_int']=hist['geohash5'].rank(method='dense').astype(int)-1
    cat_hex2int_dic=dict(zip(hist['category_hex'],hist['category_int']))

    #做second_cat_hex,second_hex_int
    hierachy=json5.load(open('/home/liuqiuyu/code/20200709_py37_ST-PIL_删掉无用的/20200709_py37_ST-PIL_删掉无用的/dataset/dataset_tsmc2014/category_hierarchy.json5','r'))
    cat_first_ids=list(hierachy['category_first'].keys())
    # 每层有两个字典，一个是该层的id：name。一个是下层的字典信息
    #1.层次化目录处理，找到每个目录的二级目录
    cat_second_ids_dict=defaultdict(list)
    cat_second_ids_dict={first_hex:[] for first_hex in cat_first_ids}#初始化二级目录字典，先将一级目录的id放进去作为键，[]作为值
    for i,cat_1st_dit in enumerate(hierachy['category_first2second']):#每个一级目录下层信息字典，即二层目录.
        cat_second_ids_dict.update({key:[] for key in cat_1st_dit['2nd_cate_ids']})#将二级目录作为键，[]作为值
        if 'category_second2third' in cat_1st_dit.keys():#如果有三级目录
            cat_2nd_dit_lst=cat_1st_dit['category_second2third']#将三级目录作为变量
            for cat_2nd_dit in cat_2nd_dit_lst:#每个三级目录
                if ('3rd_cate_ids' in cat_2nd_dit.keys()) and (cat_2nd_dit['3rd_cate_ids'] != []):#如果每个三级目录有内容,如果有内容将是某个二级目录的三级列表
                    cat_second_ids_dict[cat_2nd_dit['2nd_cate_id']].extend(cat_2nd_dit['3rd_cate_ids'])#将三级目录添加到二级目录的列表中
                if 'category_third2fourth' in cat_2nd_dit.keys():#如果每个三级目录有四级目录
                    cat_3rd_dit_lst=cat_2nd_dit['category_third2fourth']#获取四级目录迭代对象
                    for cat_3rd_dit in cat_3rd_dit_lst:#每个四级目录
                        if ('4th_cate_ids' in cat_3rd_dit.keys()) and (cat_3rd_dit['4th_cate_ids']!=[]):#如果每个四级目录有种类
                            cat_second_ids_dict[cat_2nd_dit['2nd_cate_id']].extend(cat_3rd_dit['4th_cate_ids'])#则加入到二级目录列表中
                        if 'category_fourth2fifth' in cat_3rd_dit.keys():#继续往下找，四级目录是否有五级目录
                            cat_4th_dit_lst=cat_3rd_dit['category_fourth2fifth']
                            for cat_4th_dit in cat_4th_dit_lst:#每个五级目录
                                if ('5th_cate_ids' in cat_4th_dit.keys()) and (cat_4th_dit['5th_cate_ids']!=[]):#如果五级目录有种类
                                    cat_second_ids_dict[cat_2nd_dit['2nd_cate_id']].extend(cat_4th_dit['5th_cate_ids'])

    #2.生成新的一列
    def generate_second_hex(value):
        for key,value_list in cat_second_ids_dict.items():
            if value==key:
                return key
            elif value in value_list:
                return key
        return None
    hist['second_hex']=hist['category_hex'].apply(generate_second_hex)
    hist['second_int']=hist['second_hex'].rank(method='dense').astype(int)-1
    second_hex2int_dic=dict(zip(hist['second_hex'],hist['second_int']))

    #将day填充为subdays
    def fun_split_days(day):
        #通过dataframe的一列，返回另一个值
        #day,一个星期的数字
        return [0]*(day-1)+[1]+[0]*(7-day)
    hist['sub_day']=hist['day'].apply(fun_split_days)
    #将month填充为submonth
    def fun_split_months(month):
        return [0]*(month-1)+[1]+[0]*(12-month)
    hist['sub_month']=hist['month'].apply(fun_split_months)
    #将hours填充成sub_hours
    def fun_split_hours(hours):
        return [0]*(hours-1)+[1]+[0]*(24-hours)
    hist['sub_hour']=hist['hour'].apply(fun_split_hours)
    #将hsh5填充成sub_hsh5s
    def fun_split_hsh5(hsh5):
        return [0]*(hsh5)+[1]+[0]*(95-hsh5)#hsh5从0开始
    hist['sub_hsh5']=hist['hsh_int'].apply(fun_split_hsh5)
    
    hist['user_id']=hist['user_id'].rank(method='dense').astype(int)-1
    hist=hist.sort_values(['user_id','localtime'])
    
    #其他准备
    #将cat_second_ids_dict(hex:hex),改为int:int
    cat_second_ids_int2int={second_hex2int_dic[key]:[second_hex2int_dic[val] 
                                                for val in value if val in second_hex2int_dic] 
                                                for key,value in cat_second_ids_dict.items() if key in second_hex2int_dic}
    
    
    #保存hist，和二级字典列表hex2hex，二级字典列表int2int,cat_hex2int_dic,second_hex2int_dic
    hist.to_csv(data_to,sep='\t',index=False,header=False)
    # hist.to_pickle(data_to)
    mapping_dic=[cat_second_ids_dict,cat_second_ids_int2int,cat_hex2int_dic,second_hex2int_dic]
    with open(dir_mapping_dic,'wb') as f:
        pickle.dump(mapping_dic,f)
    print(hist.columns)
    print(hist.head())




if '__main__'==__name__:
    # change_column_sin()
    # change_column_cal()
    