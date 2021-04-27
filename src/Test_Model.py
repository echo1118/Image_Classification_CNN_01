import torch
import torchvision

from src.Train_Imagedata import imshow   # 引用Train_Imagedata文件中的imshow函数
from src.Load import testloader          # 引用Load文件中的testloader变量
from src.Load import classes             # 引用Load文件中的classes变量
from src.Definition_CNN import net       # 引用Definition_CNN文件中的net变量

# 准确率评估整个网络：
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# # 用  输出图形的类标签  来评价神经网络：
# dataiter = iter(testloader)
# images, labels = dataiter.next()
#
# imshow(torchvision.utils.make_grid(images))
# print('原始类: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
#
# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
#
# print('预测类: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# # 评估10类图像分别的准确度是多少：
# import torch
# from src.Load import testloader          # 引用Load文件中的testloader变量
# from src.Load import classes             # 引用Load文件中的classes变量
#
# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(4):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1
#
# for i in range(10):
#     print('Accuracy of %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))
