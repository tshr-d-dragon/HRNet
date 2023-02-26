import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import tensorflow as tf
# import keras
# import keras.backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate


def Conv_Block(inputs, n_filters):

  y = Conv2D(filters=n_filters, kernel_size=3, strides=1, padding='same', use_bias=False)(inputs)
  y = BatchNormalization()(y)
  y = Activation('relu')(y)

  return y

def Strided_Conv_Block(inputs, n_filters, n_stride):

  y = Conv2D(filters=n_filters, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
  y = BatchNormalization()(y)
  y = Activation('relu')(y)

  return y

def Multi_Resolution_Parallel_Convolution(inputs, n_filters):

  y = Conv_Block(inputs, n_filters)
  y = Conv_Block(y, n_filters)
  y = Conv_Block(y, n_filters)
  y = Conv_Block(y, n_filters)

  return y

def Multi_Resolution_Fusion1(stage1_in, stage2_in, stage3_in, n_filters):

  stage2_i = Strided_Conv_Block(stage1_in, 2*n_filters, n_stride=2)
  stage2_ii = Strided_Conv_Block(stage2_i, 4*n_filters, n_stride=2)
  stage3 = Concatenate()([stage3_in, stage2_ii])
  stage3 = Conv2D(filters=4*n_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(stage3)
  stage3 = BatchNormalization()(stage3)
  stage3 = Activation('relu')(stage3)


  stage2 = Concatenate()([stage2_in, stage2_i])
  stage2 = Conv2D(filters=2*n_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(stage2)
  stage2 = BatchNormalization()(stage2)
  stage2 = Activation('relu')(stage2)


  stage1_i = UpSampling2D(size=2, interpolation='bilinear')(stage2_in)
  stage1 = Concatenate()([stage1_in, stage1_i])
  stage1 = Conv2D(filters=n_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(stage1)
  stage1 = BatchNormalization()(stage1)
  stage1 = Activation('relu')(stage1)

  return stage1, stage2, stage3

def Multi_Resolution_Fusion2(stage1_in, stage2_in, stage3_in, stage4_in, n_filters):

  stage4_i = Strided_Conv_Block(stage1_in, 2*n_filters, n_stride=2)
  stage4_ii = Strided_Conv_Block(stage4_i, 4*n_filters, n_stride=2)
  stage4_iii = Strided_Conv_Block(stage4_ii, 8*n_filters, n_stride=2)

  stage4_iii_i = Strided_Conv_Block(stage2_in, 4*n_filters, n_stride=2)
  stage4_iii_ii = Strided_Conv_Block(stage4_iii_i, 8*n_filters, n_stride=2)

  stage4 = Concatenate()([stage4_in, stage4_iii, stage4_iii_ii])
  stage4 = Conv2D(filters=8*n_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(stage4)
  stage4 = BatchNormalization()(stage4)
  stage4 = Activation('relu')(stage4)


  stage3 = Concatenate()([stage3_in, stage4_ii, stage4_iii_i])
  stage3 = Conv2D(filters=4*n_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(stage3)
  stage3 = BatchNormalization()(stage3)
  stage3 = Activation('relu')(stage3)


  stage2_i = UpSampling2D(size=2, interpolation='bilinear')(stage3_in)
  stage2 = Concatenate()([stage2_in, stage4_i, stage2_i])
  stage2 = Conv2D(filters=2*n_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(stage2)
  stage2 = BatchNormalization()(stage2)
  stage2 = Activation('relu')(stage2)


  stage1_i = UpSampling2D(size=2, interpolation='bilinear')(stage2_in)
  stage1_ii = UpSampling2D(size=4, interpolation='bilinear')(stage3_in)
  stage1 = Concatenate()([stage1_in, stage1_i, stage1_ii])
  stage1 = Conv2D(filters=n_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(stage1)
  stage1 = BatchNormalization()(stage1)
  stage1 = Activation('relu')(stage1)

  return stage1, stage2, stage3, stage4

def Multi_Resolution_Fusion3(stage1_in, stage2_in, stage3_in, stage4_in, n_filters):

  stage4_i = Strided_Conv_Block(stage1_in, 2*n_filters, n_stride=2)
  stage4_ii = Strided_Conv_Block(stage4_i, 4*n_filters, n_stride=2)
  stage4_iii = Strided_Conv_Block(stage4_ii, 8*n_filters, n_stride=2)

  stage4_iii_i = Strided_Conv_Block(stage2_in, 4*n_filters, n_stride=2)
  stage4_iii_ii = Strided_Conv_Block(stage4_iii_i, 8*n_filters, n_stride=2)

  stage4_iii_ii_i = Strided_Conv_Block(stage3_in, 8*n_filters, n_stride=2)

  stage4 = Concatenate()([stage4_in, stage4_iii, stage4_iii_ii, stage4_iii_ii_i])
  stage4 = Conv2D(filters=8*n_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(stage4)
  stage4 = BatchNormalization()(stage4)
  stage4 = Activation('relu')(stage4)


  stage3_i = UpSampling2D(size=2, interpolation='bilinear')(stage4_in)
  stage3 = Concatenate()([stage3_in, stage4_ii, stage4_iii_i, stage3_i])
  stage3 = Conv2D(filters=4*n_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(stage3)
  stage3 = BatchNormalization()(stage3)
  stage3 = Activation('relu')(stage3)


  stage2_i = UpSampling2D(size=2, interpolation='bilinear')(stage3_in)
  stage2_ii = UpSampling2D(size=4, interpolation='bilinear')(stage4_in)
  stage2 = Concatenate()([stage2_in, stage4_i, stage2_i, stage2_ii])
  stage2 = Conv2D(filters=2*n_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(stage2)
  stage2 = BatchNormalization()(stage2)
  stage2 = Activation('relu')(stage2)


  stage1_i = UpSampling2D(size=2, interpolation='bilinear')(stage2_in)
  stage1_ii = UpSampling2D(size=4, interpolation='bilinear')(stage3_in)
  stage1_iii = UpSampling2D(size=8, interpolation='bilinear')(stage4_in)
  stage1 = Concatenate()([stage1_in, stage1_i, stage1_ii, stage1_iii])
  stage1 = Conv2D(filters=n_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(stage1)
  stage1 = BatchNormalization()(stage1)
  stage1 = Activation('relu')(stage1)

  return stage1, stage2, stage3, stage4

def HRNet(inputs, n_filters):

  # inputs = Input(input_shape)

  stage1 = Multi_Resolution_Parallel_Convolution(inputs, n_filters)
  stage2 = Strided_Conv_Block(stage1, 2*n_filters, n_stride=2)

  stage1 = Conv_Block(stage1, n_filters)
  stage1 = Multi_Resolution_Parallel_Convolution(stage1, n_filters) 
  stage2 = Multi_Resolution_Parallel_Convolution(stage2, 2*n_filters)
  stage3 = Strided_Conv_Block(stage2, 4*n_filters, n_stride=2)

  stage1, stage2, stage3 = Multi_Resolution_Fusion1(stage1, stage2, stage3, n_filters)
  stage1 = Multi_Resolution_Parallel_Convolution(stage1, n_filters)
  stage2 = Multi_Resolution_Parallel_Convolution(stage2, 2*n_filters)
  stage3 = Multi_Resolution_Parallel_Convolution(stage3, 4*n_filters)
  stage4 = Strided_Conv_Block(stage3, 8*n_filters, n_stride=2)

  stage1, stage2, stage3, stage4 = Multi_Resolution_Fusion2(stage1, stage2, stage3, stage4, n_filters)
  stage1 = Multi_Resolution_Parallel_Convolution(stage1, n_filters)
  stage2 = Multi_Resolution_Parallel_Convolution(stage2, 2*n_filters)
  stage3 = Multi_Resolution_Parallel_Convolution(stage3, 4*n_filters)
  stage4 = Multi_Resolution_Parallel_Convolution(stage4, 8*n_filters)
  stage1, stage2, stage3, stage4 = Multi_Resolution_Fusion3(stage1, stage2, stage3, stage4, n_filters)
  # print(stage1.shape, stage2.shape, stage3.shape, stage4.shape)
  
  return stage1, stage2, stage3, stage4