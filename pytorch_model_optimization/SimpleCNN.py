# 导入函数库
import torch
import torchvision
import torchvision.transforms as transforms  # 主要用于做数据转换
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # 里面主要包含优化器

# 训练的时候记得把数据和网络都传到GPU设备上
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用第一个GPU,通过cuda:0指定

# 设置transform, 用于对数据集进行转换
transform = transforms.Compose([
    transforms.ToTensor(), # convert a PIL image or numpy.ndarray to tensor
    # output[channel] = (input[channel] - mean[channel]) / std[channel]
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # 对应RGB三个通道，前者是mean,后者是std,数据缩放到[-1,1]
])

# 加载训练集和测试集，同时进行shuffle, batch_size, transform等设置
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  # train训练集设置为True
                                       download=True, transform=transform) # 对训练集进行transform
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, # batch_size一般取4的倍数
                                         shuffle=True, num_workers=2)  # shuffle训练集设置为True

testset = torchvision.datasets.CIFAR10(root='./data', train=False,  # train测试集设置为False
                                      download=True, transform=transform) # 对测试集进行transform
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=2)  # shuffle测试集设置为False

# 类别名称
classes = trainset.classes # 取出来的是一个类别列表

# 图片可视化函数
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5  # 因为一开始把数据缩放到了[-1,1]，所以这里需要还原到[0,1]，然后才能imshow
    npimg = img.numpy()  # 转换成numpy array
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 因为Pytorch接受的是CHW，所以需要转换成HWC进行imshow
    plt.show()

# 对训练集图片进行可视化
dataiter = iter(trainloader)
images, labels = dataiter.next() # images的shape=[4, 3, 32, 32], 3指的就是上面说的C

# 查看数据是否缩放到了[-1,1]
print('最大像素值', torch.max(images))  # 查看像素值的最大值，就是想验证下是否真的缩放到了[-1,1]
print('最小像素值', torch.min(images))

# 可视化训练集的第一个batch图片
# imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

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
net = Net()

# 把网络传送到GPU上
net.to(device)

# 查看网络处于eval还是training状态
print(net.training)   # 一般来说网络默认是training状态

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 一般作为分类的损失函数使用
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 通过net.parameters()把所有trainable的参数都传入优化器里

# 开始训练网络，epoch设置为2，并输出损失值
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)  # 数据也要传到GPU设备上

        # we need to set the gradients to zero before starting to do backpropragation because Pytorch accumulates
        # the gradients on subsequent backward passes
        optimizer.zero_grad()  # 对梯度进行清零

        outputs = net(inputs)  # 得到logits
        loss = criterion(outputs, labels)  # 计算损失值, outputs的shape=[4, 10]   labels的shape=[4]
        # computes dloss/dx for every parameter x which has requires_grad=True. there are accumulated into
        # x.grad for every parameter x
        loss.backward()  # 反向传播
        # performs a parameter update based on the current gradient(stored in .grad attribute of a parameter) and
        # the update rule
        optimizer.step()  # 梯度更新

        running_loss += loss.item()  # 此处的loss值是单一样本的损失值,而不是mini-batch的，因为criterion计算过程中取了均值
        if i % 2000 == 1999:  # 每2000个step输出一次损失值
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# 保存模型权重
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)  # 仅仅保存模型参数，torch.save(net, PATH)保存整个模型，包含优化器等
# optimizer.state_dict()可以查看优化器里的参数

# 可视化测试集的一个batch图片
dataiter = iter(testloader)
images, labels = dataiter.next()

# imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 声明一个类实例，并从.pth加载模型,也就是参数
net = Net()
net.load_state_dict(torch.load(PATH)) # 加载预训练过的模型，也就是参数，一般来说是模型结构+参数，在restore的时候，如果想训练还要优化器
net.to(device)  # 把网络传到GPU设备上

# 对一个batch的测试集图片进行预测
outputs = net(images.to(device))  # 此处要把待预测的图片传到GPU上
_, predicted = torch.max(outputs, 1)

print('Predicted:   ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# 在10000张测试集上的准确率,计算公式和下面这个理解很类似: Accuracy = (TP + TN) / (TP + TN + FP + FN)
correct = 0
total = 0
# the wrapper with torch.no_grad() temporarily set all requires_grad flag to false
with torch.no_grad():  # 推断的时候不需要计算梯度
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device) # 把数据传到GPU设备上
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)  # outputs.data得到输出的tensor值
        total += labels.size(0) # labels.size()的输出是torch.size([4]), 所以要加个索引0
        correct += (predicted == labels).sum().item() # 比如a = tensor([1.0]), 运行a.item()得到输出1.0

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# 在每个类别上的准确率,计算公式和下面这个理解很类似: Recall = TP / (TP + FN)
class_correct = list(0.0 for i in range(10))
class_total = list(0.0 for i in range(10))
with torch.no_grad():  # 推断的时候不需要计算梯度
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()  # c的值为True和False组成的列表
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()  # 我测试过，+上一个True等于+1, +上一个False等于+0
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))