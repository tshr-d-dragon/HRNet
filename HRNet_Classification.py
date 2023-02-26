import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import keras
# import keras.backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, Dense, Input, Add, GlobalAveragePooling2D
from keras.models import Model
from HRNet import HRNet, Strided_Conv_Block


print(tf.__version__)       # 2.11.0
print(keras.__version__)    # 2.11.0


def HRNet_Classification(input_shape=(1024,1024,3), n_classes=4, n_filters=32):

  inputs = Input(input_shape)

  stage1, stage2, stage3, stage4 = HRNet(inputs, n_filters=32)

  stage1 = Conv2D(filters=4*n_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(stage1)
  stage1 = BatchNormalization()(stage1)
  stage1 = Activation('relu')(stage1)

  stage2 = Conv2D(filters=8*n_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(stage2)
  stage2 = BatchNormalization()(stage2)
  stage2 = Activation('relu')(stage2)

  stage3 = Conv2D(filters=16*n_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(stage3)
  stage3 = BatchNormalization()(stage3)
  stage3 = Activation('relu')(stage3)

  stage4 = Conv2D(filters=32*n_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(stage4)
  stage4 = BatchNormalization()(stage4)
  stage4 = Activation('relu')(stage4)

  y1 = Strided_Conv_Block(stage1, 8*n_filters, n_stride=2)
  y = Add()([stage2, y1])

  y2 = Strided_Conv_Block(y1, 16*n_filters, n_stride=2)
  y = Add()([stage3, y2])

  y3 = Strided_Conv_Block(y2, 32*n_filters, n_stride=2)
  y = Add()([stage4, y3])

  y = Conv2D(filters=64*n_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(y)
  y = BatchNormalization()(y)
  y = Activation('relu')(y)

  y = Conv2D(filters=64*n_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(y)
  y = BatchNormalization()(y)
  y = Activation('relu')(y)

  y = GlobalAveragePooling2D()(y)
  y = Dense(n_classes, use_bias=False)(y)

  if n_classes == 1:
    y = Activation('sigmoid')(y)
  else:
    y = Activation('softmax')(y)
  # print(y.shape)

  model = Model(inputs=inputs, outputs=y)

  return model


def main():

  model = HRNet_Classification(input_shape=(1024,1024,3), n_classes=4, n_filters=32)
  # model.summary()  


if __name__== '__main__':

  main()