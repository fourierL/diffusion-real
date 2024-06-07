from rlbench_dataset import RLBenchDataset
import torch
import sys
sys.path.append('/home/fourierl/Project/diffusion-policy')
dataset=RLBenchDataset(dataset_path='rlbench_data/real_drawer',pred_horizon=8,obs_horizon=2,action_horizon=4)

stats = dataset.stats

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)

for idx, nbatch in enumerate(dataloader):
    pass