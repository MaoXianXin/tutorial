模型剪枝属于模型优化中的重要技术之一

[当前模型剪枝有哪些可用的开源工具](https://zhuanlan.zhihu.com/p/97451755)

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

代码在notebooks下的**TensorflowLitePruningWithWeightQuantizationKeras2TFlite.ipynb**

对于INT8量化的话，代码在notebooks下的**TensorflowLiteIntegerQuantizationKeras2TFlite.ipynb**



State-of-the-art deep learning techniques rely on over-parametrized models that are hard to deploy.(我们知道自从2012年的AlexNet在ImageNet上取得成功之后，网络就开始朝着越来越深和越来越宽的方向发展，也就是我们经常所说的网络变得越来越大，这是造成过参数化的原因) On the contrary, biological nerual networks are known to use efficient sparse connectivity(课程里其实有张图比较形象，举的例子是人的孩儿时代，青少年时代，成年时代，我们知道神经网络早期叫perceptron，后来发展到MLP，之后DNN，以及现在的CNN之类，神经网络的灵感来源是人的大脑，而我们人的成长，可以看做是对网络的不断修剪)



Tensorflow量化训练:

1. post-training quantization, 剪枝，稀疏编码，对模型存储体积进行压缩
2. quantization-aware training, forward FP32->INT8映射，backward FP32梯度更新，保存模型INT8，quantize/dequantize
3. 还有一种训练和推理都用INT8
4. 在训练过程中引入精度带来的误差，然后整个网络在训练过程中进行修正

模型大小不仅仅是内存容量问题，也是内存带宽问题

量化就是将神经网络的浮点算法转化为定点



花哨的研究往往是过于棘手或前提假设过强，以至于几乎无法引入工业界的软件栈
Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference揭示了量化感知训练的诸多细节
为什么量化是有效的(具有足够好的预测准确度)，尤其是将FP32转换为INT8时已经丢失了信息？直觉解释是神经网络被过度参数化，进而包含足够的冗余信息，裁剪这些冗余信息不会导致明显的准确度下降。相关证据表明对于给定的量化方法，FP32网络和INT8网络之间的准确度差距对于大型网络来说较小，因为大型网络过度参数化的程度更高

# 可能有用的Github上的一些东西

1. [Graph Transform Tool](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md)
2. [Convrter command line examples(tensorflow)](https://www.tensorflow.org/lite/convert/cmdline_examples)

# Tensorflow官网的一些东西

1. [Tensorflow Lite and Tensorflow operator compatibility](https://www.tensorflow.org/lite/guide/ops_compatibility)

# Tensorflow量化训练技巧

1. 在你定义好网络结构之后,加上下面这句话,即可量化训练 :
   tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=200)
2. 论文中提到,为了使量化训练有更好的精度,推荐使用 relu6 ,让输出限制在较小的范围内
3. tf.contrib.quantize.create_eval_graph()和tf.contrib.quantize.create_training_graph()不能同时出现在同一程序中，不然会出问题
4. 基于已经训好的网络去做模拟量化实验的，不基于预训练模型训不起来，可能还有坑要踩，而且在模拟量化训练过程中bn层参数固定，融合bn参数也是用已经训练好的移动均值和方差，而不是用每个batch的均值和方差
5. 重写后的 eval 图与训练图并非平凡地等价，这是因为量化操作会影响 batchnorm 这一步骤
6. 对于卷积层之后带batchnorm的网络，因为一般在实际使用阶段，为了优化速度，batchnorm的参数都会提前融合进卷积层的参数中，所以训练模拟量化的过程也要按照这个流程．首先把batchnorm的参数与卷积层的参数融合，然后再对这个参数做量化
7. 对于权值的量化分通道进行求缩放因子，然后对于激活值的量化整体求一个缩放因子，这样的效果最好

# 理论文章

1. [tensorflow的量化教程(2)](https://blog.csdn.net/u012101561/article/details/86321621)
2. [卷积神经网络训练模拟量化实践](https://my.oschina.net/Ldpe2G/blog/3000810)
3. [神经网络量化简介](https://jackwish.net/neural-network-quantization-introduction-chn.html)

# 实践文章

1. 用于coral TPU的预测 [Object detection and image classification with Google Coral USB Accelerator(pyImageSearch)](https://www.pyimagesearch.com/2019/05/13/object-detection-and-image-classification-with-google-coral-usb-accelerator/)
2. 基于tfslim的方式量化训练 [Quantizing neural networks to 8-bit using TensorFlow(armDevelop)](https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/quantizing-neural-networks-to-8-bit-using-tensorflow)
3. [【Tensorflow系列】使用Inception_resnet_v2训练自己的数据集并用Tensorboard监控(cnblogs)](https://www.cnblogs.com/andre-ma/p/8458172.html)

