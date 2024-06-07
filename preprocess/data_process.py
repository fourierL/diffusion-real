import numpy as np
import os

# 这是episode的总数，多个任务的数量需要一致，如果episode_num=18,3个任务，则每个任务收集6条轨迹
episode_num=6
# Important!!! 根据任务
num_task=1

path='rlbench_data/data_drawer_npy'
data_action=[]
data_state=[]
data_image=[]
episodes_end=[]
cls=[]
cnt=0
for i in range(episode_num):
    img = np.load(path+'/images'+f'/episode{i}_img.npy')
    img=img.transpose(0,3,1,2)
    state = np.load(path+'/states'+f'/episode{i}_pose.npy')
    # 对齐维度
    len_img=img.shape[0]
    # len_state=state.shape[0]
    # len_act=min(len_img,len_state)
    # img=img[:len_act]
    # state=state[:len_act]
    len_img=img.shape[0]
    len_state=state.shape[0]
    # print('img',len_img)
    # print('state',len_state)
    action = state[1:,:]
    action=np.concatenate((action,state[-1:,]),axis=0)

    data_state.append(state)
    data_image.append(img)
    data_action.append(action)

    cnt = cnt + len_img
    episodes_end.append(cnt)
    
    # modify here!!! add instr embedding, depend on class number
    # 几个任务就分为几类
    cls_cnt=episode_num//num_task
    for _ in range(len_img):
        cls.append(int(i//cls_cnt))
        
data_image=np.concatenate(data_image,axis=0)
data_state=np.concatenate(data_state,axis=0)
data_action=np.concatenate(data_action,axis=0)


np.save('rlbench_data/real_drawer/data_image_front.npy',data_image)
np.save('rlbench_data/real_drawer/data_state.npy',data_state)
np.save('rlbench_data/real_drawer/data_action.npy',data_action)
np.save('rlbench_data/real_drawer/episodes_end.npy',episodes_end)
np.save('rlbench_data/real_drawer/task_class.npy',cls)
