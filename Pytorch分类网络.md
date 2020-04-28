**该教程内容总结**

手敲了一遍代码，同时进行一些想法实验，对教程添加详细注释和补充，资料来源也给了相应链接，Pytorch分类，CPU版本，GPU版本，复杂网络迁移学习版本(存在2种方式，freeze or not feature extraction layer)，添加Accuracy和Recall等常见评价指标(可以引入confusion matrix)，对ResNet,DenseNet等复杂网络，利用Tensorboard进行可视化，相对于看代码来说，简洁明了，还有一部分如何计算Pytorch内存占用的内容，以上为目前Pytorch分类教程已完成的内容



数据处理流程一般是numpy->tensor

torchvision里面有[内置数据集](https://pytorch.org/docs/stable/torchvision/datasets.html#),比如mnist, cifar10, flickr等等

训练一个分类器的步骤:

1. Load and normalizing the CIFAR10 training and test datasets using torchvision
2. Define a Convolutional Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

# Pytorch分类教程

[网络如何搭建参考来源](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)

## 关于分类网络训练和推理的关键点

训练的时候,需要做数据预处理,把像素值缩放到[-1,1],对数据进行shuffle和batch
推理的时候,需要做数据预处理,把像素值缩放到[-1,1],对数据进行batch
上面两步属于data pipeline阶段,我们可以朝多线程方向优化速度**(这里需要补充,数据读取多线程还需要查资料)**,不同数据集,预处理方式可能不一样

接下来是网络结构定义,以及损失函数和优化器定义

然后是训练,这里要注意一个点,训练的时候我们一般采用GPU,所以需要把数据和网络都传到GPU设备上
在推理的时候,我们可以把数据和网络都放在CPU上,就是速度会慢些

官方给的是这么一个[tutorial](https://pytorch.org/tutorials/_downloads/17a7c7cb80916fcdf921097825a0f562/cifar10_tutorial.ipynb),文件我已经下载到本地了,在notebooks文件夹下的**Pytorch_cifar10_CPU_Accuracy_Recall.ipynb**,我要做的是在这上面进行改进,下面是我的一些概要:

1. 数据加载方面的I/O,对应于num_workers参数

2. 网络结构简单,2层卷积+3层全连接

3. 在CPU上训练模型的,未调用GPU,不过有个点要注意,简单的模型其实在GPU上加速不明显

4. 计算了每个类别的预测准确率,这个地方我觉得比较好

## 第一步-利用GPU训练模型

一些概要:

1. 要把网络传到GPU设备上
2. 要把训练用的数据也传到GPU设备上
3. 预测的时候使用的是CPU,先把模型保存成.pth,再进行加载预测

代码看notebooks文件夹下的**Pytorch_cifar10_GPU_VGG_Accuracy_Recall.ipynb**

## 第二步-使用复杂网络VGG等

官方已经实现的一些在ImageNet上预训练的[模型](https://pytorch.org/docs/stable/torchvision/models.html#)

这里的话,我觉得可以结合transfer learning吧



model.training可以用于查看模型处于eval状态还是training状态

model处于eval还是training状态对模型的影响,下面这句话解释的不错:model.train() and model.eval() are flags that tell the model that you are training the model and testing mode respectively. This will make the model behave accordingly to techniques such us **dropout** that have different procedures in train and testing mode.

特征提取模块是否freeze

训练过程中使用learning rate scheduler

transfer learning的话,官方是替换掉最后一层的全连接,从而实现把原来的1000分类,换成2分类,而且还使用了预训练过的参数作为初始化,大大缩短训练时间

代码的话,在notebooks文件夹下的**Pytorch_transfer_learning_freeze_layer_or_not_model_eval_or_training.ipynb**

## 显存优化

三篇参考文章:

[浅谈深度学习:如何计算模型以及中间变量的显存占用大小](https://oldpan.me/archives/how-to-calculate-gpu-memory)

[如何在Pytorch中精细化利用显存](https://oldpan.me/archives/how-to-use-memory-pytorch)

[再次浅谈Pytorch中的显存利用问题(附完善显存跟踪代码)](https://oldpan.me/archives/pytorch-gpu-memory-usage-track)

卷积核参数,output_feature_map,forward和backward过程中产生的中间变量,优化器,主要是这几个点占用显存



占用显存比较多空间的并不是我们输入图像,而是神经网络中的中间变量以及使用optimizer算法时产生的巨量的中间参数

中间变量在backward的时候会翻倍

比如及时清空中间变量,优化代码,减少batch

消费级显卡对单精度计算有优化,服务器级别显卡对双精度计算有优化

比较需要注意的点是模型准确率不能降太多

CPU调用和GPU调用优化问题

其实用实验管理工具就可以监测CPU, GPU利用率,GPU显存,RAM的实时使用情况



模型权重参数所占用数据量的计算:

这里拿vgg16来举例子,我们可以发现计算出来的数值为527.79M和下载过来的vgg16-397923af.pth大小是一致的

代码的话在notebooks文件夹下的**Pytorch_memory_compute.ipynb**

关于[Pytorch-Memory-Utils](https://github.com/Oldpan/Pytorch-Memory-Utils)的利用还得写**(这里需要补充,根据第三篇文章)**

## 利用Tensorboard对复杂网络进行可视化

代码在notebooks文件夹下的**Pytorch_tensorboard_summary.ipynb**

## 实验管理工具neptune.ai(已放弃)

[官网链接](https://neptune.ai/)

看个案例,然后自己差不多就可以改写了,下面是关键点,自己查一下

```
import neptune

neptune.init('xianxinmao/sandbox')
neptune.create_experiment()

neptune.send_metric('batch_loss', loss.data.cpu().numpy())
```

代码的话在notebooks文件夹下的**cifar10_tutorial _GPU_neptune.ipynb**

