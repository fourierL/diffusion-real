from pyrep.const import RenderMode
import torch
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

from urx.robot import Robot
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
initial_pose=[0.2813766, -0.00867719, 1.4709872, -0.0026369, 0.9923816,-0.00390227, 0.12311166, 1.]


class Ur5():
    def __init__(self,ip_addr):
        # 初始化机械臂和夹爪状态
        self.robot = Robot(ip_addr)
        self.gripper = Robotiq_Two_Finger_Gripper(self.robot)
        self.gripper_state=0.0

        # 初始化相机

    def reset(self):
        self.robot.movel(initial_pose)
        self.gripper.open_gripper()
        return self.get_obs()
    def step(self, action):
        self.robot.movel(action[:7])
        if action[7]>0.5:
            self.gripper.close_gripper()
            self.gripper_state=1.0
        else:
            self.gripper.open_gripper()
        return self.get_obs()
    def get_obs(self):
        # 相机拍摄
        # images=..camera
        # states=..getl()+gripper_state
        # return images, states
        pass




# 相机进程

# 获取机械臂位姿进程


def eval(cfg,ckpt_path,dataset_path):
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
    dataloader, stats = data_load_rb(pred_horizon=pred_horizon, obs_horizon=obs_horizon, batch_size=64, action_horizon=action_horizon,path=dataset_path)

    # if not cfg.rand:
    #     np.random.seed(0)
    #     random.seed(0)


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
    # task=launch_env(taskclass=task_class)
    env=Ur5(ip_addr='192.168.1.40')
    for i in range(num_eval):
        gripper_state=0.0
        images_front, states=env.reset()
        obs=dict()
        obs['state']=np.array(states)
        obs['images_front']=np.array(images_front).transpose(2,0,1)
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
        step_idx = 0
        max_steps = max_steps
        done = False

        while not done:
            B = 1
            # stack the last obs_horizon number of observations
            images_front = np.stack([x['images_front'] for x in obs_deque])
            state = np.stack([x['state'] for x in obs_deque])

            # normalize observation
            nagent_poses = normalize_data(state, stats=stats['state'])
            # images are already normalized to [0,1]


            # normalize image based on ImageNet(std, mean)
            nimages_front = normalize(torch.tensor(images_front / 255.0)).detach().numpy()


            # device transfer
            nimages_front = torch.from_numpy(nimages_front).to(device, dtype=torch.float32)

            # (2,3,96,96)
            nstate= torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            # (2,2)
            # infer action
            with torch.no_grad():
                # get image features
                # image_features = ema_nets['vision_encoder'](nimages_1)
                image_features = ema_nets['vision_encoder'](nimages_front)
                # image_features = ema_nets['view_fuse_net'](image_features, image_features2)
                # (2,512)

                # concat with low-dim observations
                obs_features = torch.cat([image_features, nstate], dim=-1)

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
            for j in range(len(action)):

                images,states = env.step(action[j])
                # images=np.array(images)
                # states=np.array(states)
                obs = dict()
                obs['state'] = np.array(states)
                obs['image_front'] = np.array(images).transpose(2, 0, 1)
                # save observations
                obs_deque.append(obs)
                step_idx+=1
                if step_idx>max_steps:
                    env.reset()



cfg=OmegaConf.load('config/cfg_real.yaml')
eval(cfg, ckpt_path='ckpts/real_pick/ema/rst_epoch1550_0.006.ckpt', dataset_path='rlbench_data/real')






