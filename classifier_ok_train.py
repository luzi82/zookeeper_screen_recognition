import os
import json
import sys
import random
import cv2
import numpy as np
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import json
import classifier_ok_model
import classifier_ok
import csv

WIDTH  = classifier_ok_model.WIDTH
HEIGHT = classifier_ok_model.HEIGHT

def sample_list_to_data_set(sample_list):
    fn_list = [ sample['fn'] for sample in sample_list ]
    img_list = load_img_list(fn_list)
    ok_list_list = []
    for sample in sample_list:
        ok_list = [ 1 if ((i>=sample['y'])and(i<sample['y']+sample['h'])) else -1 for i in range(256) ]
        ok_list = np.array(ok_list)
        ok_list = ok_list[128:]
        ok_list = np.reshape(ok_list,(HEIGHT,2))
        ok_list = np.mean(ok_list,axis=1)
        ok_list_list.append(ok_list)
    ok_list_list = np.array(ok_list_list)
    ok_list_list = np.reshape(ok_list_list,(len(sample_list), HEIGHT) )
    return img_list, ok_list_list

def load_img_list(fn_list):
    img_list = [ load_img(fn) for fn in fn_list ]
    return np.array(img_list)

def load_img(fn):
    img = classifier_ok.load_img(fn)
    img = classifier_ok_model.preprocess_img(img)
    return img

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier trainer')
    parser.add_argument('epochs', nargs='?', type=int, help="epochs count")
    parser.add_argument('--testonly', action='store_true', help="test only")
    parser.add_argument('--summaryonly', action='store_true', help="summary only")
    args = parser.parse_args()

    assert((args.epochs!=None)or(args.testonly))

    sample_csv_path = os.path.join('label','ok.txt')
    sample_list = []
    with open(sample_csv_path,'r') as fin:
        for line in csv.reader(fin):
            assert(len(line)==3)
            sample_list.append({'fn':line[0],'y':int(line[1]),'h':int(line[2])})

    random.shuffle(sample_list)

    test_count = int(len(sample_list)/10)

    train_sample_list = sample_list[test_count:-test_count]
    test_sample_list  = sample_list[:test_count]
    valid_sample_list = sample_list[-test_count:]

    model = classifier_ok_model.create_model()
    model.summary()
    
    if args.summaryonly:
        quit()
    
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
        
    if not args.testonly:
        train_img_list, train_ok_list_list = sample_list_to_data_set(train_sample_list)
        valid_img_list, valid_ok_list_list = sample_list_to_data_set(valid_sample_list)
        
        epochs = args.epochs
        checkpointer = ModelCheckpoint(filepath='model/classifier_ok.hdf5', verbose=1, save_best_only=True)
        model.fit(train_img_list, train_ok_list_list,
            validation_data=(valid_img_list, valid_ok_list_list),
            epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

    model.load_weights('model/classifier_ok.hdf5')

    test_img_list,  test_ok_list_list  = sample_list_to_data_set(test_sample_list)
    test_loss = model.test_on_batch(test_img_list,  test_ok_list_list)
    print('Test loss: %.4f' % test_loss)
