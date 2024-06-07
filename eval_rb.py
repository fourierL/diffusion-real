import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity,EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import FS10_V1
from rlbench.tasks import PickAndLift
from pyrep.const import RenderMode
import torch
import os
import gdown
import random
from model.view_fuse_net import FeatureFusionNN
from model.vision_encoder import get_resnet,replace_bn_with_gn,VisionEncoder
from model.conditional_unet1D import ConditionalUnet1D
import torch.nn as nn
import collections
import numpy as np
from tqdm.auto import tqdm
from dataset.rlbench_dataset import normalize_data,unnormalize_data,get_data_stats
from dataset.load_data import data_load,data_load_rb
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from skvideo.io import vwrite
import torchvision.transforms as transforms
from diffusers.training_utils import EMAModel


vision_encoder = VisionEncoder()
# vision_encoder = replace_bn_with_gn(vision_encoder)

# vision_encoder2=get_resnet('resnet18')
vision_encoder2=VisionEncoder()
# vision_encoder2=replace_bn_with_gn(vision_encoder2)


view_fuse_net = FeatureFusionNN(input_size=512,hidden_size=256)


# ResNet18 has output dim of 512
vision_feature_dim = 256
# agent_pos is 2 dimensional
lowdim_obs_dim = 8
# observation feature has 514 dims in total per step
obs_dim = vision_feature_dim + lowdim_obs_dim
action_dim = 8


batch_size=64
obs_horizon=2
pred_horizon=16
action_horizon=8
# num_epochs = 100


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# create dataset from file

# save training data statistics (min, max) for each dim
dataloader,stats=data_load_rb(pred_horizon=pred_horizon,obs_horizon=obs_horizon,batch_size=64,action_horizon=action_horizon)


# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

# the final arch has 2 parts
nets = nn.ModuleDict({
    # 'vision_encoder': vision_encoder,
    'vision_encoder2': vision_encoder2,
    # 'view_fuse_net': view_fuse_net,
    'noise_pred_net': noise_pred_net
})



device = torch.device('cuda')
nets.to(device)


np.random.seed(0)
random.seed(0)


load_pretrained = True
if load_pretrained:
  # ckpt_path = "pusht_vision_100ep.ckpt"
  ckpt_path = "ckpts/rst_epoch950_0.000.ckpt"
  state_dict = torch.load(ckpt_path, map_location='cuda')
  ema_nets = nets
  ema_nets.load_state_dict(state_dict)
  print('Pretrained weights loaded.')
else:
  print("Skipped pretrained weight loading.")

#@markdown ### **Inference**

# limit enviornment interaction to 200 steps before termination



obs_config = ObservationConfig()
obs_config.set_all(True)
obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
obs_config.overhead_camera.render_mode = RenderMode.OPENGL
obs_config.wrist_camera.render_mode = RenderMode.OPENGL
obs_config.front_camera.render_mode = RenderMode.OPENGL


env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=Discrete()),
    obs_config=ObservationConfig(),
    headless=False)
env.launch()
task=env.get_task(PickAndLift)


num_diffusion_iters = 200

noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

normalize = transforms.Normalize(mean=mean, std=std)
num_test=10
for i in range(num_test):
    desc, obs_info = task.reset()
    obs = dict()
    obs['state'] = np.concatenate([obs_info.gripper_pose, [obs_info.gripper_open]])
    obs['image1'] = obs_info.front_rgb.transpose(2, 0, 1)
    obs['image2'] = obs_info.wrist_rgb.transpose(2, 0, 1)
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)

    step_idx = 0
    max_steps = 200
    done = False

    while not done:
        B = 1
        # stack the last obs_horizon number of observations
        images_1 = np.stack([x['image1'] for x in obs_deque])
        images_2 = np.stack([x['image2'] for x in obs_deque])
        agent_poses = np.stack([x['state'] for x in obs_deque])

        # normalize observation
        nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
        # images are already normalized to [0,1]

        # nimages_1=normalize(images_1/255.0)
        # nimages_2=normalize(images_2/255.0)

        nimages_1 = normalize(torch.tensor(images_1 / 255.0)).detach().numpy()
        nimages_2 = normalize(torch.tensor(images_2 / 255.0)).detach().numpy()


        # nimages_1 = normalize_data(images_1, stats=stats['image1'])
        # nimages_2 = normalize_data(images_1, stats=stats['image2'])



        # device transfer
        nimages_1 = torch.from_numpy(nimages_1).to(device, dtype=torch.float32)
        nimages_2 = torch.from_numpy(nimages_2).to(device, dtype=torch.float32)
        # (2,3,96,96)
        nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
        # (2,2)

        # infer action
        with torch.no_grad():
            # get image features
            # image_features = ema_nets['vision_encoder'](nimages_1)
            image_features2 = ema_nets['vision_encoder2'](nimages_2)
            # image_features = ema_nets['view_fuse_net'](image_features, image_features2)
            # (2,512)

            # concat with low-dim observations
            obs_features = torch.cat([image_features2, nagent_poses], dim=-1)

            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, pred_horizon, action_dim), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                # predict noise
                noise_pred = ema_nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        naction = naction.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=stats['action'])

        # only take action_horizon number of actions
        start = obs_horizon - 1
        end = start + action_horizon
        action = action_pred[start:end, :]
        # (action_horizon, action_dim)

        # execute action_horizon number of steps
        # without replanning
        for i in range(len(action)):
            # stepping env
            # action[i][3] = 0.0
            # action[i][4] = 0.0
            # action[i][5] = 1.0
            # action[i][6] = 0.0

            obs_info, reward, done = task.step(action[i])
            # print(action[i])
            obs = dict()
            obs['state'] = np.concatenate([obs_info.gripper_pose, [obs_info.gripper_open]])
            obs['image1'] = obs_info.front_rgb.transpose(2, 0, 1)
            obs['image2'] = obs_info.wrist_rgb.transpose(2, 0, 1)
            # save observations
            obs_deque.append(obs)
            # and reward/vis
            # update progress bar
            step_idx += 1
            print(step_idx)
            # pbar.update(1)
            # pbar.set_postfix(reward=reward)
            if step_idx > max_steps:
                done = True
            if done:
                break




#
# desc,obs_info=task.reset()
# print(obs_info.gripper_pose)
# obs=dict()
# obs['state']=np.concatenate([obs_info.gripper_pose,[obs_info.gripper_open]])
# obs['image1']=obs_info.front_rgb.transpose(2,0,1)
# obs['image2']=obs_info.overhead_rgb.transpose(2,0,1)
#
# obs_deque = collections.deque(
#     [obs] * obs_horizon, maxlen=obs_horizon)
#
# num_diffusion_iters = 200
#
# noise_scheduler = DDPMScheduler(
#     num_train_timesteps=num_diffusion_iters,
#     # the choise of beta schedule has big impact on performance
#     # we found squared cosine works the best
#     beta_schedule='squaredcos_cap_v2',
#     # clip output to [-1,1] to improve stability
#     clip_sample=True,
#     # our network predicts noise (instead of denoised action)
#     prediction_type='epsilon'
# )

# step_idx=0
# max_steps = 400
# done=False
# # for i in range(2):
# with tqdm(total=max_steps, desc="Eval PickAndLift") as pbar:
#     while not done:
#         B = 1
#         # stack the last obs_horizon number of observations
#         images_1 = np.stack([x['image1'] for x in obs_deque])
#         images_2 = np.stack([x['image2'] for x in obs_deque])
#         agent_poses = np.stack([x['state'] for x in obs_deque])
#
#         # normalize observation
#         nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
#         # images are already normalized to [0,1]
#         nimages_1 = normalize_data(images_1,stats=stats['image1'])
#         nimages_2 = normalize_data(images_1,stats=stats['image2'])
#
#         # device transfer
#         nimages_1 = torch.from_numpy(nimages_1).to(device, dtype=torch.float32)
#         nimages_2 = torch.from_numpy(nimages_2).to(device, dtype=torch.float32)
#         # (2,3,96,96)
#         nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
#         # (2,2)
#
#         # infer action
#         with torch.no_grad():
#             # get image features
#             image_features = ema_nets['vision_encoder'](nimages_1)
        #     image_features2=ema_nets['vision_encoder2'](nimages_2)
        #     image_features=ema_nets['view_fuse_net'](image_features,image_features2)
        #     # (2,512)
        #
        #     # concat with low-dim observations
        #     obs_features = torch.cat([image_features, nagent_poses], dim=-1)
        #
        #     # reshape observation to (B,obs_horizon*obs_dim)
        #     obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
        #
        #     # initialize action from Guassian noise
        #     noisy_action = torch.randn(
        #         (B, pred_horizon, action_dim), device=device)
        #     naction = noisy_action
        #
        #     # init scheduler
        #     noise_scheduler.set_timesteps(num_diffusion_iters)
        #
        #     for k in noise_scheduler.timesteps:
        #         # predict noise
        #         noise_pred = ema_nets['noise_pred_net'](
        #             sample=naction,
        #             timestep=k,
        #             global_cond=obs_cond
        #         )
        #
        #         # inverse diffusion step (remove noise)
        #         naction = noise_scheduler.step(
        #             model_output=noise_pred,
        #             timestep=k,
        #             sample=naction
        #         ).prev_sample
        #
        # # unnormalize action
        # naction = naction.detach().to('cpu').numpy()
        # # (B, pred_horizon, action_dim)
        # naction = naction[0]
        # action_pred = unnormalize_data(naction, stats=stats['action'])
        #
        #
        # # only take action_horizon number of actions
        # start = obs_horizon - 1
        # end = start + action_horizon
        # action = action_pred[start:end,:]
        # # (action_horizon, action_dim)
        #
        # # execute action_horizon number of steps
        # # without replanning
        # for i in range(len(action)):
        #     # stepping env
        #     # action[i][3] = 0.0
        #     # action[i][4] = 0.0
        #     # action[i][5] = 1.0
        #     # action[i][6] = 0.0
        #
        #     obs_info, reward, done = task.step(action[i])
        #     # print(action[i])
        #     obs = dict()
        #     obs['state'] = np.concatenate([obs_info.gripper_pose, [obs_info.gripper_open]])
        #     obs['image1'] = obs_info.front_rgb.transpose(2, 0, 1)
        #     obs['image2'] = obs_info.overhead_rgb.transpose(2, 0, 1)
        #     # save observations
        #     obs_deque.append(obs)
        #     # and reward/vis
        #     # update progress bar
        #     step_idx += 1
        #     pbar.update(1)
        #     pbar.set_postfix(reward=reward)
        #     if step_idx > max_steps:
        #         done = True
        #     if done:
        #         break

pass






