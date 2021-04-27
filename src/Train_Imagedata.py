# 第二步：训练图像数据
import torchvision
from src.Load import trainloader    # 引用Load文件中的trainloader变量
from src.Load import classes        # 引用Load文件中的classes变量

# 定义显示方法：
import numpy as np
import matplotlib.pyplot as plt


def imshow(img):
    # 输入数据：类型(torch.tensor[c,h,w]：[宽度，高度，颜色图层])
    img = img / 2 + 0.5  # 反归一处理
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # plt(属于numpy)显示要求图片格式转变为(torch.numpy[h,w,c])
    plt.show()


# 加载图像：
dataiter = iter(trainloader)  # 随机加载一个mini batch
images, labels = dataiter.next()

# 显示图像：
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
