import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusionNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FeatureFusionNN, self).__init__()
        self.fc1 = nn.Linear(input_size * 2, hidden_size)  # 输入层到隐藏层的全连接层
        self.fc2 = nn.Linear(hidden_size, input_size)      # 隐藏层到输出层的全连接层

    def forward(self, x1, x2):
        # 将两个输入特征向量拼接在一起
        x = torch.cat((x1, x2), dim=1)
        # 输入到隐藏层，应用激活函数ReLU
        x = F.relu(self.fc1(x))
        # 隐藏层到输出层
        x = self.fc2(x)
        return x