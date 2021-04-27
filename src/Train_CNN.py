# 第五步：训练神经网络
from src.Load import trainloader    # 引用Load文件中的trainloader变量
from src.Definition_CNN import net          # 引用Definition_CNN文件中的net变量
from src.Definition_CNN import criterion    # 引用Definition_CNN文件中的criterion变量
from src.Definition_CNN import optimizer    # 引用Definition_CNN文件中的optimizer变量


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data  # 数据包括图像与标签两部分

        # zero the parameter gradients
        optimizer.zero_grad()  # 梯度清零

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()     # 本次学习的梯度反向传递
        optimizer.step()    # 利用本次的梯度更新权值

        # print statistics（定期输出）
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


# 用  输出图形的类标签  来评价神经网络：
import torch
import torchvision
from src.Train_Imagedata import imshow   # 引用Train_Imagedata文件中的imshow函数
from src.Load import testloader          # 引用Load文件中的testloader变量
from src.Load import classes             # 引用Load文件中的classes变量
from src.Definition_CNN import net       # 引用Definition_CNN文件中的net变量
dataiter = iter(testloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print('原始类: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('预测类: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


# # 评估10类图像分别的准确度是多少
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


# 评估网络在整个数据集上的准确度
# import torch
# from src.Load import testloader          # 引用Load文件中的testloader变量
# correct = 0   # 测试数据中正确个数
# total = 0     # 总共测试数据数量
# with torch.no_grad():     # 不需要梯度
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
# # 统计正确数量和总共数量:
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))
