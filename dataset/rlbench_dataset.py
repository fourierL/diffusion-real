import numpy as np
import torch
import zarr
import torchvision.transforms as transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    # print(indices[:200,])

    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


class RLBenchDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 instr_num_per_task: int):

        # read from zarr dataset
        # dataset_root = zarr.open(dataset_path, 'r')
        #
        # # float32, [0,1], (N,96,96,3)
        # train_image_data = dataset_root['data']['img'][:]
        # train_image_data = np.moveaxis(train_image_data, -1,1)
        # # (N,3,96,96)
        #
        # # (N, D)
        # train_data = {
        #     # first two dims of state vector are agent (i.e. gripper) locations
        #     'agent_pos': dataset_root['data']['state'][:,:2],
        #     'action': dataset_root['data']['action'][:]
        # }
        # episode_ends = dataset_root['meta']['episode_ends'][:]

        train_image_data_front=np.load(dataset_path+'/data_image_front.npy')
        # train_image_data_wrist=np.load(dataset_path+'/data_image_wrist.npy')
        
        train_data= {
            'state': np.load(dataset_path+'/data_state.npy'),
            'action': np.load(dataset_path+'/data_action.npy')
        }
        episode_ends = np.load(dataset_path+'/episodes_end.npy')
        cls=np.load(dataset_path+'/task_class.npy')

        # print(episode_ends)


        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # normalized_train_data['image_wrist']=normalize(torch.tensor(train_image_data_wrist/255.0)).detach().numpy()
        normalized_train_data['image_front']=normalize(torch.tensor(train_image_data_front/255.0)).detach().numpy()
        normalized_train_data['task_cls']=cls
        # images are already normalized
        # stats['image1']=get_data_stats(train_image_data_1)
        # normalized_train_data['image1'] = normalize_data(train_image_data_1, stats['image1'])
        #
        #
        # stats['image2']=get_data_stats(train_image_data_2)
        # normalized_train_data['image2'] = normalize_data(train_image_data_2, stats['image2'])
        # print(normalized_train_data['action'][:150,])

        # normalized_train_data['image1'] = normalize(torch.tensor(train_image_data_1 / 255.0)).detach().numpy()
        # normalized_train_data['image2'] = normalize(torch.tensor(train_image_data_2 / 255.0)).detach().numpy()

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.instr_num_per_task=instr_num_per_task

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        # nsample['image_front'] = nsample['image_front'][:self.obs_horizon,:]
        # nsample['image_wrist'] = nsample['image_wrist'][:self.obs_horizon,:]
        nsample['image_front'] = nsample['image_front'][:self.obs_horizon,:]
        nsample['state'] = nsample['state'][:self.obs_horizon,:]
        nsample['task_cls']=nsample['task_cls'][:self.obs_horizon]
        
        
        # TODO:如果属于第n类，则调用相应的语言
        # index是每个任务对应语言的数量，最好统一每个任务的语言数量
        # 此处
        index=np.random.randint(0,self.instr_num_per_task)
        task_index=nsample['task_cls'][0]
        language=np.load(f'instr/task_{task_index}/instr_{index}.npy')
        nsample['instr']=language
        
        return nsample