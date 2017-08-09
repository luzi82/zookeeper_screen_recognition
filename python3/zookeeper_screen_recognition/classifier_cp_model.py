from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.models import Sequential
from keras import regularizers
from . import _util
import cv2
import numpy as np

INPUT_WIDTH  = 6
INPUT_HEIGHT = 1
CROP_X0 = 22
CROP_X1 = CROP_X0 + 50
CROP_Y0 = 22
CROP_Y1 = CROP_Y0 + 4

PHI = _util.PHI

INPUT_SHAPE = (INPUT_HEIGHT,INPUT_WIDTH,3)

def preprocess_img(img):
    img = img[CROP_Y0:CROP_Y1,CROP_X0:CROP_X1,:]
    img = cv2.resize(img,dsize=(INPUT_WIDTH,INPUT_HEIGHT),interpolation=cv2.INTER_AREA)
    assert(img.shape==INPUT_SHAPE)
    return img

def create_model(label_count):
    model = Sequential()
    model.add(Conv2D(filters=4, kernel_size=1, padding='valid', activation='elu', input_shape=INPUT_SHAPE))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(8, activation='elu'))
    model.add(Dense(label_count))
    model.add(Activation('softmax'))
    return model
