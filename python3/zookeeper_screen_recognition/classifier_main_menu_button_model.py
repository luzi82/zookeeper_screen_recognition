from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.models import Sequential
from keras import regularizers
from . import _util
import cv2
import numpy as np

INPUT_WIDTH  = 18
INPUT_HEIGHT = 18
CROP_X0 = 31
CROP_X1 = CROP_X0 + 58
CROP_Y0 = 33
CROP_Y1 = CROP_Y0 + 35

PHI = _util.PHI

INPUT_SHAPE = (INPUT_WIDTH,INPUT_HEIGHT,5)

_XY_LAYER = _util.xy_layer(INPUT_WIDTH, INPUT_HEIGHT)

def preprocess_img(img):
    img = img[CROP_Y0:CROP_Y1,CROP_X0:CROP_X1,:]
    img = cv2.resize(img,dsize=(INPUT_WIDTH,INPUT_HEIGHT),interpolation=cv2.INTER_AREA)
    img = np.append(img,_XY_LAYER,axis=2)
    assert(img.shape==(INPUT_HEIGHT,INPUT_WIDTH,5))
    return img

def create_model(label_count):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=1, padding='valid', activation='elu', input_shape=INPUT_SHAPE))
    model.add(Conv2D(filters=16, kernel_size=3, padding='valid', activation='elu'))
    model.add(Conv2D(filters=16, kernel_size=1, padding='valid', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=16, kernel_size=1, padding='valid', activation='elu'))
    model.add(Conv2D(filters=16, kernel_size=3, padding='valid', activation='elu'))
    model.add(Conv2D(filters=16, kernel_size=1, padding='valid', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=16, kernel_size=1, padding='valid', activation='elu'))
    model.add(Conv2D(filters=16, kernel_size=3, padding='valid', activation='elu'))
    model.add(Conv2D(filters=16, kernel_size=1, padding='valid', activation='elu'))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(2-PHI))
    model.add(Dense(16, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(2-PHI))
    model.add(Dense(16, activation='elu', activity_regularizer=regularizers.l1(0.01/16)))
    model.add(BatchNormalization())
    model.add(Dropout(2-PHI))
    model.add(Dense(label_count, kernel_regularizer=regularizers.l1(0.01/(label_count*16))))
    model.add(Activation('softmax'))
    return model
