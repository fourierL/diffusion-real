import os
import re

import numpy as np

num_episodes=6
last_state_npy=np.random.randn(7)
last_image_npy=np.random.randn(3)
def natural_sort_key(s,_nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre,s)]

for i in range(num_episodes):
    folder_state_path=f'rlbench_data/data_drawer_npy/states/episode{i}'
    folder_image_path=f'rlbench_data/data_drawer_npy/images/episode{i}'
    file_names=[f for f in os.listdir(folder_state_path) if f.endswith('.npy')]

    file_names.sort(key=natural_sort_key)


    arrays_state=[]
    arrays_image=[]
    for file_name in file_names:
        file_state_path=os.path.join(folder_state_path,file_name)
        file_image_path=os.path.join(folder_image_path,file_name)
        cur_state_npy=np.load(file_state_path)
        cur_image_npy=np.load(file_image_path)
        if cur_state_npy[0]-last_state_npy[0]<0.007 and cur_state_npy[1]-last_state_npy[1]<0.007 and cur_state_npy[2]-last_state_npy[2]<0.007 and cur_state_npy[6]==last_state_npy[6]:
            pass
        else:
            arrays_state.append(cur_state_npy)
            arrays_image.append(cur_image_npy)
        last_state_npy=cur_state_npy
        last_image_npy=cur_image_npy
    # result=np.stack(arrays,axis=0)
    np.save(f'rlbench_data/data_drawer_npy/states/episode{i}_pose.npy',arrays_state)
    np.save(f'rlbench_data/data_drawer_npy/images/episode{i}_img.npy',arrays_image)

