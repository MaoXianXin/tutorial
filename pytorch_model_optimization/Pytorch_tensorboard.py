# 导入函数库
import torch
from torch.utils.tensorboard import SummaryWriter  # 用于记录训练日志
import torch.nn as nn
import torch.nn.functional as F

# 声明一个writer实例，用于写入events文件
writer = SummaryWriter('./runs/network_visualization')

# 简单网络结构，2层卷积3层全连接
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 声明一个网络实例
model_ft = Net()

# 模拟输入，要和预训练模型的shape对应上
inputs = torch.ones([1, 3, 32, 32], dtype=torch.float32)

# 把模型写到硬盘上
writer.add_graph(model_ft, inputs)
writer.close()