from transformers import CLIPTokenizer, CLIPModel
import torch
import sys
import json
sys.path.append('/home/fourierl/Project/diffusion-policy')
import os
import numpy as np

# 选择一个预训练的CLIP模型
model_name = 'openai/clip-vit-base-patch32'
tokenizer = CLIPTokenizer.from_pretrained('clip_model/clip-vit-base-patch32')
model = CLIPModel.from_pretrained('clip_model/clip-vit-base-patch32')

# 读取JSON文件
with open('config/instrs.json', 'r') as file:
    data = json.load(file)

# 遍历每个任务
for task in data:
    task_id = task['task_id']
    instructions = task['instr']

    # 创建文件夹
    folder_name = f'instr/task_{task_id}'
    os.makedirs(folder_name, exist_ok=True)

    # 编码每个指令并保存为numpy文件
    for idx, instr in enumerate(instructions):
        # 编码指令
        encoded_input = tokenizer([instr], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            features = model.get_text_features(**encoded_input)
        
        # 将特征向量从Tensor转换为numpy数组
        features_numpy = features.cpu().numpy()

        # 保存为numpy文件
        npy_file_path = os.path.join(folder_name, f'instr_{idx}.npy')
        np.save(npy_file_path, features_numpy)
