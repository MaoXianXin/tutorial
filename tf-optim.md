[当前模型剪枝有哪些可用的开源工具](https://zhuanlan.zhihu.com/p/97451755)

模型剪枝属于模型优化中的重要技术之一

工具有以下几个:

+ TensorFlow Model Optimization Toolkit
+ PaddleSlim
+ PocketFlow
+ distiller

官方提供的有post-training quantization, quantization aware training, and pruning

the starting point to use our training APIs is a Keras training script, which can be optionally initialized from a pre-trained Keras model to further fine tune

sparse models are easier to compress, and we can skip the zeroes during inference for latency improvements



1. 用keras重头开始训练一个分类模型
2. 用pruning API对模型进行剪枝训练
3. 比较压缩后的剪枝模型和未剪枝模型的大小
4. 将剪枝后的模型转化成TFlite格式，并验证准确率损失
5. Pruning结合post-training quantization技术



State-of-the-art deep learning techniques rely on over-parametrized models that are hard to deploy.(我们知道自从2012年的AlexNet在ImageNet上取得成功之后，网络就开始朝着越来越深和越来越宽的方向发展，也就是我们经常所说的网络变得越来越大，这是造成过参数化的原因) On the contrary, biological nerual networks are known to use efficient sparse connectivity(课程里其实有张图比较形象，举的例子是人的孩儿时代，青少年时代，成年时代，我们知道神经网络早期叫perceptron，后来发展到MLP，之后DNN，以及现在的CNN之类，神经网络的灵感来源是人的大脑，而我们人的成长，可以看做是对网络的不断修剪)