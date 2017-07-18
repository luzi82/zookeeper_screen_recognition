from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.models import Sequential
from keras import regularizers
from . import _util
import cv2
import numpy as np

SCREEN_WIDTH = 120
SCREEN_HEIGHT = 213
WIDTH = Math.floor(SCREEN_WIDTH/16)*2 # ensure even
HEIGHT = Math.floor(SCREEN_HEIGHT/2)
CROP_X0 = (SCREEN_WIDTH-WIDTH)/2
CROP_X1 = CROP_X0+WIDTH
CROP_Y0 = SCREEN_HEIGHT-HEIGHT

_XY1_LAYER = _util.xy1_layer(WIDTH, HEIGHT)

def preprocess_img(img):
    img = img[CROP_X0:CROP_X1,CROP_Y0:,:]
    img = cv2.resize(img,dsize=(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA)
    img = np.append(img,_XY1_LAYER,axis=2)
    assert(img.shape == (HEIGHT, WIDTH, 6))
    return img

def create_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=1, padding='same', activation='elu', input_shape=(HEIGHT,WIDTH,6)))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=1, padding='same', activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=1, padding='same', activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=1, padding='same', activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=(1,WIDTH), padding='valid', activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=1, padding='same', activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=(3,1), padding='same', activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=1, padding='same', activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=(3,1), padding='same', activation='elu'))
    model.add(Conv2D(filters=1, kernel_size=1, padding='same', activation='elu',
        activity_regularizer=regularizers.l1(0.01/(2*HEIGHT))
    ))
    model.add(Flatten())
    model.add(Activation('tanh'))
    return model
