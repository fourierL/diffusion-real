
import os
import gdown
from dataset.push_image_dataset import PushTImageDataset
import torch
from dataset.rlbench_dataset import RLBenchDataset



def data_load(dataset_path='pusht_cchi_v7_replay.zarr.zip',pred_horizon=16,obs_horizon=4,action_horizon=8,batch_size=64):
    if not os.path.isfile(dataset_path):
        id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
        gdown.download(id=id, output=dataset_path, quiet=False)

    dataset = PushTImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )
    # save training data statistics (min, max) for each dim
    stats = dataset.stats

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )
    return dataloader,stats

def data_load_rb(pred_horizon,obs_horizon,action_horizon,batch_size,path,instr_num_per_task):


    dataset = RLBenchDataset(
        dataset_path=path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        instr_num_per_task=instr_num_per_task
    )
    # save training data statistics (min, max) for each dim
    stats = dataset.stats

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )
    return dataloader,stats


# data_load_rb(16,2,8,128,'../rlbench_data/100_trjs_pickandlift')



#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16


# create dataset from file


