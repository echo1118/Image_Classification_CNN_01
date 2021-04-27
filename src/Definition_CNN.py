# 第三步：定义神经网络
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 打印神经网络：
net = Net()
print(net)

# 第四步：定义损失函数和优化器
import torch.optim as optim

# 计算损失：
# 定义损失函数：
criterion = nn.CrossEntropyLoss()         # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
# 实现随机梯度下降    lr（float）:学习速率  momentum（float）:动量因子
