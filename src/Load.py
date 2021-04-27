# 第一步：加载数据
import torch
import torchvision
# transforms 定义了一系列数据转化形式，并对数据进行处理
import torchvision.transforms as transforms


# 定义归一化方法：
# transforms.Compose():Compose()类会将transforms列表里面的transform操作进行遍历。
# Compose里面的参数实际上就是个列表，而这个列表里面的元素就是你想要执行的transform操作。

transform = transforms.Compose(
    [transforms.ToTensor(),  # 传入数据转化成张量形式
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 定义归一化方法
     # transforms.Normalize((mean),(std)):用给定的均值和标准差对每个通道数据进行归一化：
     # 归一方式：  (input[channel]-mean[channel])/std[channel]
     ]
)

# 训练数据集：  CIFAR10数据集      原来：root='./data'
trainset = torchvision.datasets.CIFAR10(root='./data/', train=True, download=True, transform=transform)
# root（string）：数据集的根目录在哪里
# train（bool，optional）：如果为True，则创建数据集training.pt，否则创建数据集test.pt。
# download（bool，optional）：如果为true，则从Internet下载数据集并将其放在根目录中。如果已下载数据集，则不会再次下载。
# transform（callable ，optional）：一个函数/转换，它接收PIL图像并返回转换后的版本。
# target_transform（callable ，optional）：接收目标并对其进行转换的函数/转换。
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
# dataset(dataset):输入的数据类型
# batch_size（数据类型 int）:每次输入数据的行数，默认为1。
# shuffle（数据类型 bool）：洗牌。默认设置为False。在每次迭代训练时是否将数据洗牌，默认设置是False。将输入数据的顺序打乱，是为了使数据更有独立性，
# num_workers（数据类型 Int）：工作者数量，默认是0。使用多少个子进程来导入数据。

# 测试数据集：
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# # 用于查看已下载CIFAR10数据集的所在路径:
# import os.path
# path1=os.path.abspath('.')
# print(path1)
# path2=os.path.abspath('../..')
# print(path2)
