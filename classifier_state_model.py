from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras import regularizers

WIDTH = 36
HEIGHT = 64
PHI = (1+5**0.5)/2

def create_model(label_count):
    model = Sequential()
    model.add(Conv2D(
        filters=32, kernel_size=3, padding='valid', activation='elu', input_shape=(HEIGHT,WIDTH,5),
        kernel_regularizer = regularizers.l2(0.01),
        bias_regularizer = regularizers.l2(0.01)
    ))
    model.add(Conv2D(filters=32, kernel_size=1, padding='valid', activation='elu',
        kernel_regularizer = regularizers.l2(0.01),
        bias_regularizer = regularizers.l2(0.01)
    ))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding='valid', activation='elu',
        kernel_regularizer = regularizers.l2(0.01),
        bias_regularizer = regularizers.l2(0.01)
    ))
    model.add(Conv2D(filters=64, kernel_size=1, padding='valid', activation='elu',
        kernel_regularizer = regularizers.l2(0.01),
        bias_regularizer = regularizers.l2(0.01)
    ))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=128, kernel_size=(2,3), padding='valid', activation='elu',
        kernel_regularizer = regularizers.l2(0.01),
        bias_regularizer = regularizers.l2(0.01)
    ))
    model.add(Conv2D(filters=128, kernel_size=1, padding='valid', activation='elu',
        kernel_regularizer = regularizers.l2(0.01),
        bias_regularizer = regularizers.l2(0.01)
    ))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=256, kernel_size=2, padding='valid', activation='elu',
        kernel_regularizer = regularizers.l2(0.01),
        bias_regularizer = regularizers.l2(0.01)
    ))
    model.add(Conv2D(filters=256, kernel_size=1, padding='valid', activation='elu',
        kernel_regularizer = regularizers.l2(0.01),
        bias_regularizer = regularizers.l2(0.01)
    ))
    model.add(MaxPooling2D(pool_size=2))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(2-PHI))
    model.add(Dense(256, activation='elu',
        kernel_regularizer = regularizers.l2(0.01),
        bias_regularizer = regularizers.l2(0.01),
    ))
    model.add(Dropout(2-PHI))
    model.add(Dense(256, activation='elu',
        kernel_regularizer = regularizers.l2(0.01),
        bias_regularizer = regularizers.l2(0.01),
    ))
    model.add(Dropout(2-PHI))
    model.add(Dense(256, activation='elu',
        kernel_regularizer = regularizers.l2(0.01),
        bias_regularizer = regularizers.l2(0.01),
        activity_regularizer = regularizers.l2(0.01)
    ))
    model.add(Dropout(2-PHI))
    model.add(Dense(label_count, activation='softmax'))
    return model

