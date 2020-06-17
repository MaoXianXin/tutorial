# import packages
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds # 这个是指Tensorflow Datasets
import mlflow
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', default=224, type=int)
parser.add_argument('--batch_size', default=192, type=int)
parser.add_argument('--shuffle_buffer_size', default=192, type=int)
parser.add_argument('--dataset_name', default='food101', type=str)
parser.add_argument('--split', default=['train', 'validation'], type=list)
parser.add_argument('--data_dir', default='./tensorflow_datasets', type=str)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--classes', default=101, type=int)
parser.add_argument('--weights_path', default='./models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', type=str)
parser.add_argument('--save_keras_file', default='./test.h5', type=str)
args = parser.parse_args()

# 如果出现显存不够的错误,把这个代码加上
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 一些参数设置
layers = tf.keras.layers
models = tf.keras.models

# 定义ResNet50用于food101分类
def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def ResNet50(input_shape=(args.img_size, args.img_size, 3),
             classes=args.classes):
    img_input = layers.Input(shape=input_shape)  # 输入节点

    bn_axis = 3
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input  # inputs是输入节点
    outputs = x  # outputs是输出节点
    model = models.Model(inputs, outputs, name='resnet50')  # 生成一个Model, 需要指定输入和输出

    model.load_weights(args.weights_path, by_name=True) # 注意by_name参数很有用，把layer和layer name对应上了

    return model

# 进行数据增强
def convert(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
    image = tf.image.resize_with_crop_or_pad(image, args.img_size+32, args.img_size+32)
    image = tf.image.random_crop(image, size=[args.img_size, args.img_size, 3]) # Random crop back to 224x224
    return image, label

def augment(image, label):
    image, label = convert(image, label)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.3) # Random brightness
    return image, label

def main():
    # 数据读取并预处理，此处使用tfds的方式构建data pipeline
    (raw_train, raw_validation), metadata = tfds.load(
        args.dataset_name, # 数据集名称，这个是food101分类数据集，共101个类别
        split=args.split, # 这里的raw_validation和split的'validation'对应，raw_train和split的'train'对应
        with_info=True, # 这个参数和metadata对应
        as_supervised=True, # 这个参数的作用是返回tuple形式的(input, label),举个例子，raw_train=tuple(input, label)
        data_dir=args.data_dir
    )

    # 可以体验下这里是否加prefetch(tf.data.experimental.AUTOTUNE)和cache()的区别，对训练速度，以及CPU负载有影响
    train_batches = raw_train.shuffle(args.shuffle_buffer_size).map(augment).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_batches = raw_validation.map(convert).batch(args.batch_size)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = ResNet50()
        # 进行模型compile
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

    start = time.time()
    # logging parameters
    with mlflow.start_run():
        mlflow.log_param('IMG_SIZE', str(args.img_size))
        mlflow.log_param('BATCH_SIZE', str(args.batch_size))
        mlflow.log_param('DATASET_NAME', args.dataset_name)
        mlflow.log_param('LEARNING_RATE', str(args.learning_rate))
        mlflow.log_param('EPOCHS', str(args.epochs))
        mlflow.log_param('CLASSES', str(args.classes))
        mlflow.log_param('OPTIM', 'Adam')
        # 开启模型训练
        model.fit(
            train_batches,
            epochs=args.epochs
        )
        end = time.time()
        mlflow.log_metric('elapsedTime', end - start)
        _, trainAcc = model.evaluate(train_batches, verbose=1)
        mlflow.log_metric('trainAcc', trainAcc)
        # Baseline的test acc
        _, baseline_model_accuracy = model.evaluate(test_batches, verbose=1)
        print('Baseline test accuracy: ', baseline_model_accuracy)
        mlflow.log_metric('testAcc', baseline_model_accuracy)

    # 模型训练后预测展示
    get_label_name = metadata.features['label'].int2str

    for image, label in raw_validation.take(5):
        image, label = convert(image, label)
        predict = np.argmax(model.predict(np.expand_dims(image, axis=0)))
        print(get_label_name(label), ' is ', get_label_name(predict))

    # 保存模型
    tf.keras.models.save_model(model, args.save_keras_file, include_optimizer=False)
    print('Saved baseline model to: ', args.save_keras_file)

if __name__ == '__main__':
    main()