import pickle

from core.environments import RLBenchEnv
from utils.keystep_detection import keypoint_discovery
from utils.coord_transforms import convert_gripper_pose_world_to_image
import collections
import numpy as np


def get_observation(task_str: str, variation: int, episode: int, env: RLBenchEnv):
    demo = env.get_demo(task_str, variation, episode)

    # key_frames = keypoint_discovery(demo)
    # key_frames.insert(0, 0)

    state_dict_ls = collections.defaultdict(list)
    for i in range(len(demo)):
        state_dict = env.get_observation(demo._observations[i])
        for k, v in state_dict.items():
            if len(v) > 0:
                # rgb: (N: num_of_cameras, H, W, C); gripper: (7+1, )
                state_dict_ls[k].append(v)

    for k, v in state_dict_ls.items():
        state_dict_ls[k] = np.stack(v, 0) # (T, N, H, W, C)

    action_ls = state_dict_ls['gripper'] # (T, 7+1)
    del state_dict_ls['gripper']

    return demo, state_dict_ls, action_ls

datapath='microsteps'

rlbench_env=RLBenchEnv(
    data_path=datapath,
    apply_rgb=True,
    apply_pc=False,
    apply_cameras=['front','wrist']
)


num_episodes=200
data_image_front=[]
data_image_wrist=[]
data_state=[]
data_action=[]
episodes_end=[]
cnt=0


#当前步的gripper-pose作为状态观测，下一步的joint-position作为预测
for i in range(num_episodes):
    demo,state_dict_ls,action_ls=get_observation('pick_and_lift',0,i,rlbench_env)
    image_front=state_dict_ls['rgb'][:,0,:,:,:][::2,:,:,:]
    image_wrist=state_dict_ls['rgb'][:,1,:,:,:][::2,:,:,:]

    image_front=image_front.transpose(0,3,1,2)
    image_wrist=image_wrist.transpose(0,3,1,2)
    # 当前的gripper位置作为观测
    state=action_ls[::2,:]
    # 下一步gripper位置作为动作
    action=state[1:,]
    action=np.concatenate((action,state[-1:,]),axis=0)

    data_image_front.append(image_front)
    data_image_wrist.append(image_wrist)
    data_state.append(state)
    data_action.append(action)

    # [start, end)
    cnt=cnt+image_front.shape[0]
    episodes_end.append(cnt)


data_image_front=np.concatenate(data_image_front,axis=0)
data_image_wrist=np.concatenate(data_image_wrist,axis=0)
data_state=np.concatenate(data_action,axis=0)
data_action=np.concatenate(data_action,axis=0)
np.save('100_trjs_pickandlift/data_image_front.npy',data_image_front)
np.save('100_trjs_pickandlift/data_image_wrist.npy',data_image_wrist)
np.save('100_trjs_pickandlift/data_state.npy',data_state)
np.save('100_trjs_pickandlift/data_action.npy',data_action)
np.save('100_trjs_pickandlift/episodes_end.npy',episodes_end)


pass