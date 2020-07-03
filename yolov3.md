# 安装依赖

```
pip install -r requirements.txt
```

不过注意，此处使用的是CPU版本的tensorflow

# 训练

## 数据准备

这里说明一点，如果验证集为空，则会从训练集分割出80%的数据作为验证集

![Selection_071](pics/Selection_071.jpg)

​								从上图我们可以发现，训练所需的图片和标注都已经在data文件夹下面了

我们进入annotations文件夹看看，如下图所示

![Selection_072](pics/Selection_072.jpg)

我们进入images文件夹看看，如下图所示

![Selection_073](pics/Selection_073.jpg)

## 编辑配置文件

![Selection_074](pics/Selection_074.jpg)

因为采用的是raccoon dataset，而且是单一物体检测，所以labels=['raccoon']

然后的话，我们还需要改一下train_image_folder和train_annot_folder，这里的cache_name我建议每次训练前都删除一次

对于不同显卡，注意batch_size的设置

## 开启训练

```python train.py -c config.json```

## 预测

```python predict.py -c config.json -i /home/mao/Github/keras-yolo3/test.mp4```