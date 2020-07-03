# 克隆仓库

```git clone https://github.com/zzh8829/yolov3-tf2```

然后进入仓库:	```cd yolov3-tf2```

# 下载数据集

```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -O ./data/voc2012_raw.tar
mkdir -p ./data/voc2012_raw
tar -xf ./data/voc2012_raw.tar -C ./data/voc2012_raw
ls ./data/voc2012_raw/VOCdevkit/VOC2012 # Explore the dataset
```

# 把数据集制作成TFRcord

```
python tools/voc2012.py \
  --data_dir './data/voc2012_raw/VOCdevkit/VOC2012' \
  --split train \
  --output_file ./data/voc2012_train.tfrecord

python tools/voc2012.py \
  --data_dir './data/voc2012_raw/VOCdevkit/VOC2012' \
  --split val \
  --output_file ./data/voc2012_val.tfrecord
```

# 开始训练

这里采用迁移学习的训练方式:

## Convert pre-trained Darknet weights

```
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf
```

## start train

```
python train.py \
	--dataset ./data/voc2012_train.tfrecord \
	--val_dataset ./data/voc2012_val.tfrecord \
	--classes ./data/voc2012.names \
	--num_classes 20 \
	--mode fit --transfer darknet \
	--batch_size 16 \
	--epochs 10 \
	--weights ./checkpoints/yolov3.tf \
	--weights_num_classes 80 
```

# 进行预测

```
# detect from images
python detect.py \
	--classes ./data/voc2012.names \
	--num_classes 20 \
	--weights ./checkpoints/yolov3_train_10.tf \
	--image ./data/street.jpg
```

