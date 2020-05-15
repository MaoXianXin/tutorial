# 剪枝笔记

一个典型的network pruning过程, 有3个stage:

1. train a large, over-parameterized model
2. prune the trained large model according to a certain criterion
3. fine-tune the pruned model to regain the lost performance

# 探索Pytorch官网时发现的有意思的东西

[Pytorch Hub, SSD By NVIDIA](https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/),

[Albumentations](https://github.com/albumentations-team/albumentations),fast image augmentation library and easy to use wrapper around other libraries

[Detectron2](https://github.com/facebookresearch/detectron2),Detectron2 is FAIR's next_generation platform for object detection and segmentaion

[fast.ai](https://docs.fast.ai/)

[glow](https://github.com/pytorch/glow),compiler for neural network hardware accelerators

[horovod](https://github.com/horovod/horovod),Distributed training framework for Tensorflow, Keras, PyTorch, and Apache MXNet

[optuna](https://github.com/optuna/optuna),A hyperparameter optimization framework

# rethinking-network-pruning仓库 ICLR2019

we evaluate the following seven pruning methods

1. L1-norm based channel pruning
2. ThiNet
3. Regression based feature reconstruction
4. Network Slimming
5. Sparse Structure Selection
6. Soft filter pruning
7. Unstructured weight-level pruning

the first six is structured while the last one is unstructured(or sparse)

For CIFAR, our code is based on [pytorch-classification](https://github.com/bearpaw/pytorch-classification) and [network-slimming](https://github.com/Eric-mingjie/network-slimming)

对于pytorch-classification来说，跑过vgg19bn(92.75%), resnet_110(94.11%), resnetxt_29(95.66), densenet_100(95.23), preresnet_110(94.09)，基本上和作者跑出来的结果差不多

对于network-slimming来说，跑过vgg19, resnet164, densenet40，不过确实会出现下面的一种情况，剪枝率为60%的时候，resnet164出现错误

our experiment environment is python3.6 & PyTorch 0.3.1

[该算法的论文 Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519)

[YOLOv3-model-pruning](https://github.com/Lam1360/YOLOv3-model-pruning)

# network-slimming仓库 ICCV 2017

the code is based on [pytorch-slimming](https://github.com/foolwood/pytorch-slimming), we add support for ResNet and DenseNet

## Dependencies

torch 0.3.1      torchvision 0.2.0

搜一下train with sparsity是什么

Note: For results of pruning 60% of the channels for resnet164-cifar100, in this implementation, sometimes some layers are all pruned and there would be error. However, we also provide a mask implementation where we apply a mask to the scaling factor in BN layer. For [mask implementation](https://github.com/Eric-mingjie/network-slimming/tree/master/mask-impl), when pruning 60% of the channels in resnet164-cifar100, we can also train the pruned network