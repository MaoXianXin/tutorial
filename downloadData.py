# 导入函数库
import tensorflow_datasets as tfds # 这个是指Tensorflow Datasets

SHUFFLE_BUFFER_SIZE = 48 * 1
DATASET_NAME = 'food101'
SPLIT = ['train', 'validation']
DATA_DIR = './tensorflow_datasets'

# 数据读取并预处理，此处使用tfds的方式构建data pipeline
(raw_train, raw_validation), metadata = tfds.load(
    DATASET_NAME, # 数据集名称，这个是手势识别分类数据集，共3个类别
    split=SPLIT, # 这里的raw_test和split的'test'对应，raw_train和split的'train'对应
    with_info=True, # 这个参数和metadata对应
    as_supervised=True, # 这个参数的作用是返回tuple形式的(input, label),举个例子，raw_test=tuple(input, label)
    data_dir=DATA_DIR
)