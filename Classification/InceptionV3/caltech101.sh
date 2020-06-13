#!/bin/bash

python distributeInceptionV3Classification.py \
  --img_size 224 \
  --batch_size 64 \
  --shuffle_buffer_size 64 \
  --learning_rate 1e-4 \
  --epochs 30 \
  --classes 102 \
  --weights_path './models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5' \
  --save_keras_file './test.h5'

python distributeInceptionV3Classification.py \
  --img_size 224 \
  --batch_size 128 \
  --shuffle_buffer_size 128 \
  --learning_rate 1e-4 \
  --epochs 30 \
  --classes 102 \
  --weights_path './models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5' \
  --save_keras_file './test.h5'

python distributeInceptionV3Classification.py \
  --img_size 224 \
  --batch_size 256 \
  --shuffle_buffer_size 256 \
  --learning_rate 1e-4 \
  --epochs 30 \
  --classes 102 \
  --weights_path './models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5' \
  --save_keras_file './test.h5'

python InceptionV3Classification.py \
  --img_size 224 \
  --batch_size 32 \
  --shuffle_buffer_size 32 \
  --learning_rate 1e-4 \
  --epochs 30 \
  --classes 102 \
  --weights_path './models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5' \
  --save_keras_file './test.h5'

python InceptionV3Classification.py \
  --img_size 224 \
  --batch_size 64 \
  --shuffle_buffer_size 64 \
  --learning_rate 1e-4 \
  --epochs 30 \
  --classes 102 \
  --weights_path './models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5' \
  --save_keras_file './test.h5'

python InceptionV3Classification.py \
  --img_size 224 \
  --batch_size 128 \
  --shuffle_buffer_size 128 \
  --learning_rate 1e-4 \
  --epochs 30 \
  --classes 102 \
  --weights_path './models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5' \
  --save_keras_file './test.h5'
