
import torch

from model.vision_encoder import get_resnet,replace_bn_with_gn,VisionEncoder
from model.conditional_unet1D import ConditionalUnet1D
from model.view_fuse_net import FeatureFusionNN
import torch.nn as nn
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import numpy as np
from dataset.load_data import data_load,data_load_rb


# IMPORTANT!
# replace all BatchNorm with GroupNorm to work with EMA
# performance will tank if you forget to do this!
# vision_encoder = get_resnet('resnet18')
# vision_encoder = VisionEncoder()
# vision_encoder = replace_bn_with_gn(vision_encoder)

# vision_encoder2=get_resnet('resnet18')
vision_encoder2=get_resnet('resnet18',weights='IMAGENET1K_V1')
vision_encoder2=replace_bn_with_gn(vision_encoder2,features_per_group=16)


# view_fuse_net = FeatureFusionNN(input_size=512,hidden_size=512)


# ResNet18 has output dim of 512
vision_feature_dim = 512
# agent_pos is 2 dimensional
lowdim_obs_dim = 8
# observation feature has 514 dims in total per step
obs_dim = vision_feature_dim + lowdim_obs_dim
action_dim = 8
batch_size=256
obs_horizon=2
pred_horzion=16
action_horizon=8
# dataloader,stats=data_load(pred_horizon=pred_horzion,obs_horizon=obs_horizon,batch_size=64)
dataloader,stats=data_load_rb(pred_horizon=pred_horzion,obs_horizon=obs_horizon,batch_size=batch_size,action_horizon=action_horizon)
num_epochs = 5001

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)


# the final arch has 2 parts
nets = nn.ModuleDict({
    'vision_encoder2': vision_encoder2,
    'noise_pred_net': noise_pred_net
})

device = torch.device('cuda')
nets.to(device)



# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(
    parameters=nets.parameters(),
    power=0.75)


ema_nets = nets

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=nets.parameters(),
    lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)


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




with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    rst_loss=[]
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        # with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
        for idx,nbatch in enumerate(dataloader):
            # for nbatch in tepoch:
                # data normalized in dataset
                # device transfer
            nimage1 = nbatch['image1'][:,:obs_horizon].to(torch.float32).to(device)
            nimage2 = nbatch['image2'][:,:obs_horizon].to(torch.float32).to(device)
            nagent_pos = nbatch['agent_pos'][:,:obs_horizon].to(device)
            naction = nbatch['action'].to(device)
            B = nagent_pos.shape[0]



            # encoder vision features
            # image_features1 = nets['vision_encoder'](
            #     nimage1.flatten(end_dim=1))

            image_features2= nets['vision_encoder2'](
                nimage2.flatten(end_dim=1)
            )

            # fuse features from 2 viewpoints

            # image_features=nets['view_fuse_net'](image_features1,image_features2)


            image_features = image_features2.reshape(
                *nimage1.shape[:2],-1)
            # (B,obs_horizon,D)

            # concatenate vision feature and low-dim obs
            obs_features = torch.cat([image_features, nagent_pos], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)
            # (B, obs_horizon * obs_dim)

            # sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            # add noise to the clean images according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = noise_scheduler.add_noise(
                naction, noise, timesteps)

            # predict the noise residual
            noise_pred = noise_pred_net(
                noisy_actions, timesteps, global_cond=obs_cond)

            # L2 loss
            loss = nn.functional.mse_loss(noise_pred, noise)

            # optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # step lr scheduler every batch
            # this is different from standard pytorch behavior
            lr_scheduler.step()

            # update Exponential Moving Average of the model weights
            ema.step(nets.parameters())

            # logging
            loss_cpu = loss.item()
            epoch_loss.append(loss_cpu)
                # tepoch.set_postfix(loss=loss_cpu)

        tglobal.set_postfix(loss=np.mean(epoch_loss))
        rst_loss.append(np.mean(epoch_loss))
        if epoch_idx%50==0:
            ema.copy_to(ema_nets.parameters())
            torch.save(nets.state_dict(), f'ckpts/rst_epoch{epoch_idx}_{np.mean(epoch_loss):.3f}.ckpt')
            torch.save(ema_nets.state_dict(),
                       f'ckpts/pickandlift/ema/rst_epoch{epoch_idx}_{np.mean(epoch_loss):.3f}.ckpt')
        # rst_loss=np.array(rst_loss)
        # np.save(f'rst/epoch_{epoch_idx}_loss.npy', rst_loss)
        # rst_loss.tolist()
        if epoch_idx%20==0 or epoch_idx==num_epochs-1:
            rst_loss=np.array(rst_loss)
            np.save(f'rst/epoch_{epoch_idx}_loss.npy',rst_loss)
            rst_loss=rst_loss.tolist()

# Weights of the EMA model
# is used for inference
# ema.copy_to(ema_nets.parameters())

# torch.save(nets.state_dict(),'rst.ckpt')
# torch.save(ema_nets)