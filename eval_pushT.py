import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity,EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import FS10_V1
from rlbench.tasks import PickAndLift
from pyrep.const import RenderMode
from env.pusht_env import PushTImageEnv
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
from omegaconf  import OmegaConf
from IPython.display import Video

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def launch_env():
    env=PushTImageEnv()

    return env


def eval(cfg,ckpt_path):
    obs_dim=cfg.obs_dim
    action_dim=cfg.action_dim
    obs_horizon=cfg.obs_horizon
    pred_horizon=cfg.pred_horizon
    action_horizon=cfg.action_horizon
    batch_size=cfg.batch_size
    num_epochs=cfg.num_epochs
    num_diffusion_iters=cfg.num_diffusion_iters
    num_eval=cfg.num_eval
    max_steps=cfg.max_steps

    # load stats
    dataloader, stats = data_load(pred_horizon=pred_horizon, obs_horizon=obs_horizon, batch_size=64, action_horizon=action_horizon)



    # define network
    vision_encoder = get_resnet('resnet18', weights='IMAGENET1K_V1')
    vision_encoder = replace_bn_with_gn(vision_encoder, features_per_group=16)

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon
    )
    nets = nn.ModuleDict({
        # 'vision_encoder': vision_encoder,
        'vision_encoder': vision_encoder,
        # 'view_fuse_net': view_fuse_net,
        'noise_pred_net': noise_pred_net
    })

    device = torch.device('cuda')
    nets.to(device)

    # load ckpts
    load_pretrained = True
    if load_pretrained:
        # ckpt_path = "pusht_vision_100ep.ckpt"
        ckpt_path = ckpt_path
        state_dict = torch.load(ckpt_path, map_location='cuda')
        ema_nets = nets
        ema_nets.load_state_dict(state_dict)
        print('Pretrained weights loaded.')
    else:
        print("Skipped pretrained weight loading.")


    # define scheduler
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
    # define normalizer
    normalize = transforms.Normalize(mean=mean, std=std)

    # launch env
    env=launch_env()

    for i in range(num_eval):
        obs, info = env.reset(seed=i)
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)

        imgs=[env.render(mode='rgb_array')]

        step_idx = 0
        max_steps = max_steps
        done = False

        while not done:
            B = 1
            # stack the last obs_horizon number of observations

            images = np.stack([x['image'] for x in obs_deque])
            agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

            # normalize observation
            nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
            # images are already normalized to [0,1]


            # normalize image based on ImageNet(std, mean)
            nimages = normalize(torch.tensor(images / 255.0)).detach().numpy()


            # device transfer
            nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)

            # (2,3,96,96)
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            # (2,2)

            # infer action
            with torch.no_grad():
                # get image features
                # image_features = ema_nets['vision_encoder'](nimages_1)
                image_features = ema_nets['vision_encoder'](nimages)
                # image_features = ema_nets['view_fuse_net'](image_features, image_features2)
                # (2,512)

                # concat with low-dim observations
                obs_features = torch.cat([image_features, nagent_poses], dim=-1)

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
            for j in range(len(action)):
                obs,reward,done,_,info = env.step(action[i])
                # print(action[i])
                # save observations
                obs_deque.append(obs)
                imgs.append(env.render(mode='rgb_array'))
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

        vwrite(f'vis/vis_{i}.mp4',imgs)

cfg=OmegaConf.load('config/cfg.yaml')
eval(cfg,'ckpts_pushT/ema/rst_epoch800_0.008.ckpt')





