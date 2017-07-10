from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.models import Sequential
from keras import regularizers

WIDTH = 18
HEIGHT = 6
CROP_X0 = 54
CROP_X1 = 90
CROP_Y0 = 128

_XY1_LAYER = _util.xy1_layer(WIDTH, HEIGHT)

def preprocess_img(img):
    img = img[CROP_X0:CROP_X1,CROP_Y0:,:]
    img = cv2.resize(img,dsize=(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA)
    img = np.append(img,_XY1_LAYER,axis=2)
    return img

def create_model(label_count):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=1, padding='same', activation='elu', input_shape=(HEIGHT,WIDTH,5)))
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
