数据处理流程一般是numpy->tensor

torchvision里面有[内置数据集](https://pytorch.org/docs/stable/torchvision/datasets.html#),比如mnist, cifar10, flickr等等

训练一个分类器的步骤:

1. Load and normalizing the CIFAR10 training and test datasets using torchvision
2. Define a Convolutional Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

# Pytorch分类教程

官方给的是这么一个[tutorial](https://pytorch.org/tutorials/_downloads/17a7c7cb80916fcdf921097825a0f562/cifar10_tutorial.ipynb),文件我已经下载到本地了,在notebooks文件夹下的**cifar10_tutorial.ipynb**,我要做的是在这上面进行改进,下面是我的一些概要:

1. 数据加载方面的I/O,对应于num_workers参数

2. 网络结构简单,2层卷积+3层全连接

3. 在CPU上训练模型的,未调用GPU,不过有个点要注意,简单的模型其实在GPU上加速不明显

4. 计算了每个类别的预测准确率,这个地方我觉得比较好

## 第一步-利用GPU训练模型

一些概要:

1. 要把网络传到GPU设备上
2. 要把训练用的数据也传到GPU设备上
3. 预测的时候使用的是CPU,先把模型保存成.pth,再进行加载预测

代码看notebooks文件夹下的**cifar10_tutorial _GPU.ipynb**

## 第二步-使用复杂网络VGG等

官方已经实现的一些在ImageNet上预训练的[模型](https://pytorch.org/docs/stable/torchvision/models.html#)

这里的话,我觉得可以结合transfer learning吧



model.training可以用于查看模型处于eval状态还是training状态

model处于eval还是training状态对模型的影响,下面这句话解释的不错:model.train() and model.eval() are flags that tell the model that you are training the model and testing mode respectively. This will make the model behave accordingly to techniques such us **dropout** that have different procedures in train and testing mode.

特征提取模块是否freeze

训练过程中使用learning rate scheduler

transfer learning的话,官方是替换掉最后一层的全连接,从而实现把原来的1000分类,换成2分类,而且还使用了预训练过的参数作为初始化,大大缩短训练时间

代码的话,在notebooks文件夹下的**transfer_learning_tutorial.ipynb**

## 显存优化



## 实验管理工具neptune.ai

