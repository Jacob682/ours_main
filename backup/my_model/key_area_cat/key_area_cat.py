import pickle
import numpy as np
import pandas as pd
import math
import warnings
warnings.filterwarnings('ignore')

#1获取第一个用户打卡的combined_second_cat_id信息
#1.1获取所有用户combined_second_cat_id
with open('/home/jovyan/datasets/tsmc_nyc_6_distinguish_cat_poi/reindex_second_cat_id.pkl','rb') as f:
    users_combined_second_cat_id=pickle.load(f)
#1.2取第100个用户
user_combined_second_cat_id=users_combined_second_cat_id[971]
revers_user_combined_second_cat_id=user_combined_second_cat_id[::-1]



'''以最后一个为测试，用之前的数据计算'''
#2.需要知道所有的种类
#2.1读所有的种类字典
with open('/home/jovyan/datasets/tsmc_nyc_6_distinguish_cat_poi/distin_dic_second_cat_id.pkl','rb') as f:
    distin_dic_second_cat_id=pickle.load(f)#combined_second_id:reindex
#2.2打卡中所出现second_cat_id个数
len_second_reindex=len(distin_dic_second_cat_id)#53
combined_second_reindex_id=list(distin_dic_second_cat_id.values())



#3.2将概率封装成函数
def Compute_probability(checkins,second_cat_id):
    '''
    len_p:离散概率的长度,int
    checkins：用户的打卡列表，这里指的是所有打卡列表/月周期的打卡列表,list
    second_cat_id:所有的second_cat_id,用于找离散概率的相应位置,list
    return:prob,np.array
    '''
    prob=[0]*len(second_cat_id)
    len_checkin=len(checkins)#用于做概率相除的分母
    for c in checkins:
        prob[second_cat_id.index(c)]+=1
    fre_uc=prob#该用户在所有种类上的频度
    prob=np.array(prob)/len_checkin
    return prob,fre_uc
#测试结果
len_user_checkin=len(user_combined_second_cat_id)
g_uc,_=Compute_probability(user_combined_second_cat_id,combined_second_reindex_id)




#3.4二维列表计算每个子列表的频率
def Compute_period_prob(len_period,checkins_2d,second_cat_id):
    '''
    len_period:时期个数，int
    checkins_2d:二维打卡数据。每个时期对应的打卡种类，
    （长度为时期个数，时期内打卡长度），若某时期无打卡，为空列表。2d list
    second_cat_id:所有second_cat_id,用于找离散概率的相应位置，list
    return:pro_2d:如果某个时期没有打卡，则这个打卡概率保持空列表。(时期个数，所有二级列表的长度53)2d list
    '''
    pro_2d=[[0]]*len_period
    #对每个时期计算
    for index,j in enumerate(checkins_2d):
        if j:#若j不为空列表，则计算该列表内的概率
            j_pro,_=Compute_probability(j,second_cat_id)
            #将这个时期的概率，更新到2d_pro
            pro_2d[index]=j_pro
        if not j:
            pro_2d[index]=j
    return pro_2d



#4将用户的打卡分到每个月，得到时期的打卡概率
#4.1读取用户打卡的每月信息
with open('/home/jovyan/datasets/tsmc_nyc_3_groupby_user_chronological/month.pkl','rb') as f:
    months=pickle.load(f)
'''
foursquare的时间跨度为2012.4-2013.2，没有月数字重复
共11个月数字
'''
#4.2根据该用户所有打卡，将其分到11个时期中
user_month=months[971]
reversed_user_month=user_month[::-1]
span_month=[4,5,6,7,8,9,10,11,12,1,2]
period_second=[[]]*len(span_month)
for i in span_month:#遍历数据集中的时期
    if i in user_month:#如果用户在i月打卡
        #那么将该月打卡的second_id放到对应月列表的位置
        start=user_month.index(i)#得到切片起始
        end=len_user_checkin-reversed_user_month.index(i)#省去-1，因为下面切片，切片是右开区间
        period_second[span_month.index(i)]=user_combined_second_cat_id[start:end]
#4.3计算每个时期的打卡频率
# print(period_second)
len_period=len(span_month)
x_ujc=Compute_period_prob(len_period,period_second,combined_second_reindex_id)




#5算法主体
#5.1
def Get_Cde_Guec(d,guc,combined_second_reindex_id):
    '''
    guc,combined_second_reindex_id:list,le=53
    return C_de:用户根据d划分的关键种类，G_uec:关键种类在用户所有打卡中的概率
    '''
    G_uec=[x for x in guc if x>=d]#先获得关键种类上的概率
    dc_mask=[1 if x>=d else 0 for x in guc]#对关键种类的位置取1，否则取0
    C_de=[val for val,m in zip(combined_second_reindex_id,dc_mask) if m]
    return G_uec,C_de,dc_mask
#5.1.1验证
G_uec,C_de,dc_mask=Get_Cde_Guec(0.00,g_uc,combined_second_reindex_id)



#5.2
def Get_xuec(d,x_ujc,dc_mask):
    '''
    获得在时期上关键种类的概率
    x_ujc:用户在j时期上关键种类的概率2d list,(j,53)
    dc_mask:整体上关键种类的位置mask
    return:x_ujec,在时期上关键种类的概率
    '''
    x_uec=[[val for val,m in zip(sub_ujc,dc_mask) if m] for sub_ujc in x_ujc]
    return x_uec
#5.2.1验证
G_uec,C_de,dc_mask=Get_Cde_Guec(0,g_uc,combined_second_reindex_id)
x_ujec=Get_xuec(0,x_ujc,dc_mask)




#5.3
def KL(g_uec,x_ujec):
    '''
    x_ujec:1d list,j时期上关键种类的概率
    '''
    KLresult=0
    len_guec=len(g_uec)
    for gi,xi in zip(g_uec,x_ujec):
        if  xi != 0:
            m_ui=xi*(math.log(xi/gi))
        else:
            m_ui=math.inf
        KLresult+=m_ui
    return KLresult


#5.-1算法主题
def Compute_du(user_cat_id,g_uc,x_ujc,combined_second_reindex_id,span_month):
    '''
    user_cat_id:用户的打卡种类列表 1d_list
    g_uc:用户在所有种类的概率 1d list,len:53,所有种类个数
    x_ujc:用户在j时期上关键种类的概率2d list,(j,53)
    combined_second_reindex_id:所有的二级目录id，顺序和用户的打卡概率相同
    span_month:时期的跨度列表
    '''
    KLsum=0
    min=math.inf
    J=len(span_month)#时期跨度的长度
    for i in range(100):
        d=i/100
        G_uec,C_de,dc_mask=Get_Cde_Guec(d,g_uc,combined_second_reindex_id)#5.1
        
        sum_g=sum(G_uec)
        norm_G_uec=[val/sum_g for val in G_uec]#对G_uec归一化
        
        x_uec=Get_xuec(d,x_ujc,dc_mask)#5.2得到用户在时期关键种类上的概率
        for j in range(J):
            x_ujec=x_uec[j]#用户在某时期的概率
            
            sum_j=sum(x_ujec)
            norm_x_ujec=[val/sum_j for val in x_ujec]#对j时期关键种类概率归一化
            KLsum+=KL(norm_G_uec,norm_x_ujec)#5.4
        if KLsum<=min:
            min=KLsum
            du=d
    return du
#测试5.-1
du=Compute_du(user_combined_second_cat_id,g_uc,x_ujc,combined_second_reindex_id,span_month)
print(du)




#6获取关键种类
'''
对第100个用户取关键种类
'''
def Get_essential_second(du,g_uc,combined_second_reindex_id):
    '''
    取关键种类和相应mask
    g_uc:用户在种类上的打卡概率 1d_list，顺序和combined_second_reindex_id相同
    '''
    user_essential_second_pro=[val for val in g_uc if val>=du]
    #计算对应的种类
    user_essential_mask=[1 if val>=du else 0 for val in g_uc]
    user_essential_second=[val for val,m in zip(combined_second_reindex_id,user_essential_mask) if m]
    user_non_essential_second=[val for val,m in zip(combined_second_reindex_id,user_essential_mask) if not m]
    return user_essential_second,user_non_essential_second,user_essential_mask
user_essential_second,user_non_essential_second,user_essential_mask=Get_essential_second(du,g_uc,combined_second_reindex_id)#53个种类，缩减到5个种类,列表顺序是所有二级种类列表的顺序，不是用户打卡的顺序，用户打卡有57个





#7计算两类权重
def Compute_all_second_importance(user_essential_second,user_non_essential_second,combined_second_reindex_id,g_uc,user_combined_second_cat_id):
    second_importance=[0]*len(combined_second_reindex_id)
    for e in user_essential_second:
        index=combined_second_reindex_id.index(e)
        second_importance[index]=g_uc[index]
    for ne in user_non_essential_second:
        index=combined_second_reindex_id.index(ne)
        _,fre_uc=Compute_probability(user_combined_second_cat_id,combined_second_reindex_id)
        g_uci=g_uc[index]
        ln=math.log(len(user_combined_second_cat_id)/(1+fre_uc[index]))
        numerator=g_uci*ln
        denominator=0
        for i in range(len(user_non_essential_second)):
            denominator_inside=g_uc[i]*(math.log(len(user_combined_second_cat_id)/(1+fre_uc[i])))
            denominator+=denominator_inside
        second_importance[index]=numerator/denominator
    return second_importance
# second_importance=Compute_all_second_importance(user_essential_second,user_non_essential_second,combined_second_reindex_id,g_uc,user_combined_second_cat_id)



#8对一个用户所有打卡都做种类权重生成
#8.1算法主题
def User_importance_second(user_checkins,combined_second_id,span_month,user_months):
    '''
    user_checkins:一个用户的所有打卡，1d list
    combined_second_id:数据集中出现的second_id,1d_list
    span_month:数据集出现的月份时间跨度
    user_month:用户打卡的月份
    return:second_importance该用户所有的关键种类和对应权重，2d list，(时间步，二级种类个数)
    '''
    len_timestep=len(user_checkins)
    len_period=len(span_month)
    second_importance=[]
    #8.1循环该用户的时间步
    for i in range(1,len_timestep+1):#从0开始切得空列表，并且切不到最后一个序列，左闭右开列表
        present_checkin=user_checkins[:i]#对用户切片，得到用户[:i]种类的切片
        present_month=user_months[:i]
        len_present=i
        g_uc_present,_=Compute_probability(present_checkin,combined_second_id)#计算整体种类的概率
        #计算时期的概率
        period_second=Split_period(present_month,span_month,len_present,present_checkin)#将当前打卡划分到月份中
        x_uc=Compute_period_prob(len_period,period_second,combined_second_id)#计算月份中的打卡概率
        #计算此时打卡下的du
        du_present=Compute_du(present_checkin,g_uc,x_uc,combined_second_id,span_month)
        #获取关键种类
        user_essential_second_present,user_non_essential_present,user_essential_mask=Get_essential_second(du_present,g_uc_present,combined_second_id)
        #计算两类权重
        second_importance_present=Compute_all_second_importance(user_essential_second_present,user_non_essential_present,combined_second_id,g_uc_present,user_combined_second_cat_id)
        
        second_importance.append(second_importance_present)#更新时间步结果
    return second_importance
#8.2定义将此时打卡之前的数据分到月份里
def Split_period(present_month,span_month,len_present,present_checkin):
    '''
    return:period_second 2d list,用户在月份中打卡的种类id（月份，在该月份中的打卡），若在该月没有打卡，则该位置取[]
    '''
    reversed_present_month=present_month[::-1]
    period_second=[[]]*len(span_month)
    for i in span_month:
        if i in present_month:
            start=present_month.index(i)
            end=len_present-reversed_present_month.index(i)
            period_second[span_month.index(i)]=present_checkin[start:end]
    return period_second

second_importance=User_importance_second(users_combined_second_cat_id[0],combined_second_reindex_id,span_month,months[0])
print(1)




#9对所有用户计算每个时间步的权重
#9.1读取所有用户的打卡数据second，month(users_combined_second_cat_id,user_month)
def Compute_users_timesteps_importance(users_combined_second_cat_id,users_month,combined_second_reindex_id,span_month):
    '''
    users_combined_second_cat_id:2d list,(users,checkin_cat)
    user_month:2d list,(users,checkin_month)
    combined_second_reindex_id:1d list,数据集中所有的种类
    return: 3d list,(users,timestep,second)
    '''
    users_timesteps_importance=[]
    for u_s,u_m in zip(users_combined_second_cat_id,users_month):#循环每个用户
        u_timestep_importance=User_importance_second(u_s,combined_second_reindex_id,span_month,u_m)
        users_timesteps_importance.append(u_timestep_importance)
    return users_timesteps_importance

users_timesteps_importance=Compute_users_timesteps_importance(users_combined_second_cat_id,months,combined_second_reindex_id,span_month)
# print(users_timesteps_importance)



#10 保存所有用户在所有timestep上对所有种类的重要程度
dir_users_timesteps_second_importance='/home/jovyan/datasets/tsmc_nyc_10_users_timesteps_second_importance/users_timesteps_second_importance.pkl'
with open(dir_users_timesteps_second_importance,'wb') as f:
    pickle.dump(users_timesteps_importance,f)