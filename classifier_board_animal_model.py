from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.models import Sequential
from keras import regularizers
import _util
import cv2
import numpy as np

ICON_WIDTH  = 18
ICON_HEIGHT = 18
ICON_COUNT = 8
BOARD_WIDTH = ICON_WIDTH * ICON_COUNT
BOARD_HEIGHT = ICON_HEIGHT * ICON_COUNT
ORI_HEIGHT = 1136
ORI_CROP_Y = 332

PHI = _util.PHI

INPUT_SHAPE = (ICON_HEIGHT,ICON_WIDTH,5)

_XY_LAYER = _util.xy_layer(ICON_WIDTH, ICON_HEIGHT)
_XY_LAYER = np.tile(_XY_LAYER,ICON_COUNT*ICON_COUNT)
_XY_LAYER = np.reshape(_XY_LAYER,(ICON_COUNT*ICON_COUNT,ICON_WIDTH, ICON_HEIGHT,2))

def preprocess_img(img):
    y0 = int(img.shape[0]*ORI_CROP_Y/ORI_HEIGHT)
    y1 = y0 + img.shape[1]
    img = img[y0:y1,:,:]
    #img = cv2.resize(img,dsize=(BOARD_WIDTH,BOARD_HEIGHT),interpolation=cv2.INTER_AREA)
    assert(img.shape==(BOARD_HEIGHT,BOARD_WIDTH,3))
    img_list = [img[i*ICON_HEIGHT:(i+1)*ICON_HEIGHT,j*ICON_WIDTH:(j+1)*ICON_WIDTH,:]for i in range(ICON_COUNT) for j in range(ICON_COUNT)]
    img_list = np.array(img_list)
    img_list = np.append(img_list,_XY_LAYER,axis=3)
    assert(img_list.shape==(ICON_COUNT*ICON_COUNT,ICON_HEIGHT,ICON_WIDTH,5))
    return img_list

def create_model(label_count):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=1, padding='valid', activation='elu', input_shape=INPUT_SHAPE))
    model.add(Conv2D(filters=32, kernel_size=2, padding='valid', activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=1, padding='valid', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=1, padding='valid', activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=3, padding='valid', activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=1, padding='valid', activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=2, padding='valid', activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=1, padding='valid', activation='elu'))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(2-PHI))
    model.add(Dense(32, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(2-PHI))
    model.add(Dense(32, activation='elu', activity_regularizer=regularizers.l1(0.01/32)))
    model.add(BatchNormalization())
    model.add(Dropout(2-PHI))
    model.add(Dense(label_count, kernel_regularizer=regularizers.l1(0.01/(label_count*32))))
    model.add(Activation('softmax'))
    return model
