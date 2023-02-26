import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import keras
# import keras.backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, Input, UpSampling2D, Concatenate
from keras.models import Model
from HRNet import HRNet, Strided_Conv_Block


print(tf.__version__)       # 2.11.0
print(keras.__version__)    # 2.11.0


def HRNet_Segmentation(input_shape, n_classes, n_filters):

  inputs = Input(input_shape)

  stage1, stage2, stage3, stage4 = HRNet(inputs, n_filters=n_filters)

  stage2 = UpSampling2D(size=2, interpolation='bilinear')(stage2)
  stage3 = UpSampling2D(size=4, interpolation='bilinear')(stage3)
  stage4 = UpSampling2D(size=8, interpolation='bilinear')(stage4)
  
  y = Concatenate()([stage1, stage2, stage3, stage4])

  y = Conv2D(filters=2*n_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(y)
  y = BatchNormalization()(y)
  y = Activation('relu')(y)

  y = Conv2D(filters=n_classes, kernel_size=1, strides=1, padding='same', use_bias=False)(y)

  if n_classes == 1:
    y = Activation('sigmoid')(y)
  else:
    y = Activation('softmax')(y)
  # print(y.shape)

  model = Model(inputs=inputs, outputs=y)

  return model


def main():

  model = HRNet_Segmentation(input_shape=(512,512,3), n_classes=10, n_filters=16)
  # model.summary()  


if __name__== '__main__':

  main()