# 训练步骤

1. 本文使用VOC格式进行训练
2. 训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotations中
3. 训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中
4. 在训练前利用voc2yolo4.py文件生成对应的txt
5. 再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes
6. 就会生成对应的2007_train.txt，每一行对应其图片位置及其真实框的位置
7. 在训练前需要修改model_data里面的voc_classes.txt文件，需要将classes改成你自己的classes
8. 运行train.py即可开始训练

## 克隆仓库

```
git clone https://github.com/bubbliiiing/yolov4-tf2
cd yolov4-tf2
```

## VOC格式数据

![Selection_075](pics/Selection_075.jpg)

![Selection_076](pics/Selection_076.jpg)

![Selection_077](pics/Selection_077.jpg)

## 利用voc2yolo4.py生成对应的txt

![Selection_078](pics/Selection_078.jpg)

​								上面的路径需要注意更改

![Selection_079](pics/Selection_079.jpg)

​															上图是生成的txt文件

## 运行voc_annotation.py

需要先更改voc_classes.txt

![Selection_080](pics/Selection_080.jpg)

​						因为我们使用的是raccoon dataset，是一个单一类别的检测

![Selection_081](pics/Selection_081.jpg)

​						上图是生成的2007_train.txt等文件

## 开始训练

```python train.py```

# 进行预测

```python predict.py```



# 云端训练

![Selection_027](yolov4_img/6.jpg)

![Selection_028](yolov4_img/7.jpg)

![Selection_022](yolov4_img/1.jpg)

![Selection_023](yolov4_img/2.jpg)

![Selection_024](yolov4_img/3.jpg)

![Selection_025](yolov4_img/4.jpg)

## 第一种训练方式(长时间训练容易断网)

![Selection_026](yolov4_img/5.jpg)

## 第二种训练方式(适合长时间训练断网也没事)

![Selection_035](yolov4_img/8.jpg)

![Selection_036](yolov4_img/9.jpg)

# 预测

![Selection_064](yolov4_img/10.jpg)

```
运行   python predict.py
```

![Selection_065](yolov4_img/11.jpg)

# 代码解读

## 预测部分

![Selection_091](yolov4_img/12.jpg)

![Selection_092](yolov4_img/13.jpg)

![Selection_093](yolov4_img/16.jpg)

