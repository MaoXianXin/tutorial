# 导入函数库
import tensorflow as tf
import tensorflow_datasets as tfds

# 如果出现显存不够的错误，把这个代码加上
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

IMG_SIZE = 32
BATCH_SIZE = 4 * 2
SHUFFLE_BUFFER_SIZE = 12 * 2
DATASET_NAME = 'rock_paper_scissors'
SPLIT = ['test', 'train']
DATA_DIR = './tensorflow_datasets'
LEARNING_RATE = 1e-4
EPOCHS = 50
CLASSES = 3

# 定义LeNet5用于手势识别
def LeNet5(input_shape=(IMG_SIZE, IMG_SIZE, 3),
           classes=CLASSES):
    img_input = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (5, 5),
                      activation='relu',
                      padding='valid',
                      name='conv1')(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(64, (5, 5),
                      activation='relu',
                      padding='valid',
                      name='conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu', name='fc2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    outputs = x
    model = models.Model(inputs, outputs, name='lenet5')
    return model

# 进行数据增强
def convert(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image, label

def augment(image, label):
    image, label = convert(image, label)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    return image, label

# 数据读取并预处理，此处使用tfds的方式构建data pipeline
(raw_test, raw_train), metadata = tfds.load(
    DATASET_NAME,
    split=SPLIT,
    with_info=True,
    as_supervised=True,
    data_dir=DATA_DIR
)

# 可以体验下这里是否加prefetch(tf.data.experimental.AUTOTUNE)和cache()的区别，对训练速度，以及CPU负载有影响
train_batches = raw_train.shuffle(SHUFFLE_BUFFER_SIZE).map(augment).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
test_batches = raw_test.map(convert).batch(BATCH_SIZE)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # 声明一个LeNet5模型实例
    model = LeNet5()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

# 进行模型训练
model.fit(train_batches,
         epochs=EPOCHS
         )

# Baseline的test acc,并保存模型
_, baseline_model_accuracy = model.evaluate(test_batches, verbose=1)
print('Baseline test accuracy: ', baseline_model_accuracy)

keras_file = './test.h5'
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
print('Saved baseline model to: ', keras_file)