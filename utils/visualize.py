import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
from PIL import Image

def plot_attention(
    attentions: torch.Tensor, rgbs: torch.Tensor, pcds: torch.Tensor, dest: Path, step_id: int
) -> plt.Figure:
    # dest.mkdir(exist_ok=True, parents=True)
    attentions = attentions.detach().cpu()
    # Squeeze out the first two dimensions and then split
    # Squeezing out the first two dimensions
    # Converting the tensor to numpy and splitting it into a list of numpy arrays
    attentions = [arr.numpy() for arr in attentions.squeeze(0).squeeze(0)]

    # rgbs = rgbs.detach().cpu()
    # pcds = pcds.detach().cpu()
    # ep_dir = dest.parent
    # ep_dir.mkdir(exist_ok=True, parents=True)
    name = dest.stem
    ext = dest.suffix

    # plt.figure(figsize=(10, 8))
    num_cameras = len(attentions)


    # for cam_id, (a, rgb, pcd) in enumerate(zip(attentions, rgbs, pcds)):
    #     # plt.subplot(num_cameras, 4, i * 4 + 1)
    #
    #     a=np.transpose(a,(1,2,0))
    #
    #     # 归一化atten_map
    #     a_normalized= a-np.min(a)
    #     a_normalized/=np.max(a_normalized)
    #     a_normalized=a_normalized.reshape((128,128))
    #
    #     heatmap=plt.cm.jet(a_normalized)
    #     heatmap=heatmap[...,:3]
    #
    #     heatmap=(heatmap*255).astype(np.uint8)
    #
    #     alpha=0.75
    #     overlayed_image=(heatmap*alpha+rgb*(1-alpha)).astype(np.uint8)
    #     image=Image.fromarray(overlayed_image)
    #     save_path=dest/f'camera_{cam_id}_atten'
    #     save_path.mkdir(exist_ok=True,parents=True)
    #     image.save(save_path/f'atten_{step_id}.png')
    #
    # return plt.gcf()

    a_normalized = attentions - np.min(attentions)
    a_normalized /= np.max(a_normalized)
    a_normalized = a_normalized.reshape((128, 128))
    heatmap = plt.cm.jet(a_normalized)
    heatmap = heatmap[..., :3]

    heatmap = (heatmap * 255).astype(np.uint8)

    alpha = 0.75

    overlayed_image = (heatmap * alpha + rgbs[2] * (1 - alpha)).astype(np.uint8)
    image = Image.fromarray(overlayed_image)
    save_path = dest / f'camera_2_atten'
    save_path.mkdir(exist_ok=True, parents=True)
    image.save(save_path / f'atten_{step_id}.png')

    np.save(save_path/f'attentions_{step_id}.npy',attentions)
    np.save(save_path/f"rgb0_{step_id}.npy",rgbs[0])
    np.save(save_path/f"rgb2_{step_id}",rgbs[2])

