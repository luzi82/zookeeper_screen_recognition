import os
import json
import sys
import random
import cv2
import numpy as np
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

WIDTH = 36
HEIGHT = 64
PHI = (1+5**0.5)/2

def sample_list_to_data_set(sample_list, label_count):
    fn_list = [ sample['fn'] for sample in sample_list ]
    img_list = load_img_list(fn_list, WIDTH, HEIGHT)
    label_idx_list = np.array([ sample['label_idx'] for sample in sample_list ])
    label_onehot_list = np_utils.to_categorical(label_idx_list, label_count)
    return img_list, label_onehot_list

def load_img_list(fn_list,width,height):
    img_list = [ load_img(fn, width, height) for fn in fn_list ]
    return np.array(img_list).astype('float32')/255

def load_img(fn, width, height):
    img = cv2.imread(fn)
    img = cv2.resize(img,dsize=(width,height),interpolation=cv2.INTER_AREA)
    return img

if __name__ == '__main__':

    label_state_path = os.path.join('label','state')
    label_name_list = os.listdir(label_state_path)
    label_name_list = filter(lambda v:os.path.isfile(os.path.join(label_state_path,v)),label_name_list)
    label_name_list = filter(lambda v:v.endswith('.txt'),label_name_list)
    label_name_list = [ i[:-4] for i in label_name_list]
    label_name_list = sorted(label_name_list)
    
    label_count = len(label_name_list)

    label_name_to_idx_dict = { label_name_list[i]:i for i in range(label_count) }

    sample_list = []
    for label_name, label_idx in label_name_to_idx_dict.items():
        img_fn_list_fn = os.path.join(label_state_path,'{}.txt'.format(label_name))
        with open(img_fn_list_fn, mode='rt', encoding='utf-8') as fin:
            img_fn_list = fin.readlines()
        img_fn_list = [ img_fn.strip() for img_fn in img_fn_list ]
        sample_list += [{'fn':img_fn, 'label_idx':label_idx, 'label_name': label_name} for img_fn in img_fn_list ]

    random.shuffle(sample_list)

    #json.dump(sample_list, fp=sys.stdout, indent=2)
    
    test_count = int(len(sample_list)/10)
    train_sample_list = sample_list[test_count:-test_count]
    test_sample_list  = sample_list[:test_count]
    valid_sample_list = sample_list[-test_count:]
    
    train_img_list, train_label_onehot_list = sample_list_to_data_set(train_sample_list,label_count)
    test_img_list,  test_label_onehot_list  = sample_list_to_data_set(test_sample_list ,label_count)
    valid_img_list, valid_label_onehot_list = sample_list_to_data_set(valid_sample_list,label_count)
    
    #print(test_img_list.shape)
    #print(test_label_onehot_list.shape)
    
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, padding='valid', activation='elu', input_shape=train_img_list.shape[1:]))
    model.add(Conv2D(filters=16, kernel_size=1, padding='valid', activation='elu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='valid', activation='elu'))
    model.add(Conv2D(filters=32, kernel_size=1, padding='valid', activation='elu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding='valid', activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=1, padding='valid', activation='elu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(2-PHI))
    model.add(Dense(64, activation='elu'))
    model.add(Dropout(2-PHI))
    model.add(Dense(label_count, activation='softmax'))
    model.summary()
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    epochs = 999
    checkpointer = ModelCheckpoint(filepath='model/classifier_state.hdf5', verbose=1, save_best_only=True)
    model.fit(train_img_list, train_label_onehot_list,
        validation_data=(valid_img_list, valid_label_onehot_list),
        epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
    