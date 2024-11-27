import pickle
import sys
with open('/home/jovyan/datasets/tsmc_nyc_10_users_timesteps_second_importance/users_timesteps_second_importance.pkl','rb') as f:
    users_timesteps_second_importance=pickle.load(f)

for i,d in enumerate(users_timesteps_second_importance):
    for sub_d in d:
        if sub_d is None:
            k=1
            print(i)
# print(j)
# with open('/home/jovyan/datasets/tsmc_nyc_10_users_timesteps_second_importance/users_timesteps_second_importance.txt','w') as f:

#     sys.stdout=f
#     print(users_timesteps_second_importance)
# sys.stdout=sys.__stdout__