import os
import json
import sys
import random
import cv2
import numpy as np
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import json
import classifier_board_animal_model
import _util
import numpy as np
import add_board_animal as aba
import time

ICON_WIDTH  = classifier_board_animal_model.ICON_WIDTH
ICON_HEIGHT = classifier_board_animal_model.ICON_HEIGHT

def sample_list_to_data_set(v_dict_list, label_list):
    img_list = load_img_list_csv(v_dict_list)
    v_label_list = [i['label'] for i in v_dict_list]
    v_label_list = [label_list.index(i) for i in v_label_list]
    label_onehot_list = np_utils.to_categorical(v_label_list, len(label_list))
    return img_list, label_onehot_list

def load_img_list_csv(v_dict_list):
    #print(v_dict_list)
    vv_list_dict = {}
    for i in range(len(v_dict_list)):
        v_dict = v_dict_list[i]
        fn = v_dict['fn']
        if fn not in vv_list_dict:
            vv_list_dict[fn] = []
        vv_list_dict[fn].append((i,int(v_dict['pos'])))
    #print(vv_list_dict)
    ret = [None] * len(v_dict_list)
    for fn, pos_list in vv_list_dict.items():
        img_list = load_img_list_fn(fn)
        for i, pos in pos_list:
            ret[i] = img_list[pos]
    ret = np.array(ret)
    assert(ret.shape==(len(v_dict_list),ICON_HEIGHT,ICON_WIDTH,5))
    return ret

def load_img_list_fn(fn):
    img = _util.load_img(fn)
    img_list = classifier_board_animal_model.preprocess_img(img)
    #print(img_list.shape,file=sys.stderr)
    #print(len(pos_list),file=sys.stderr)
    #img_list = np.take(img_list,pos_list,axis=0)
    #print(img_list.shape,file=sys.stderr)
    return img_list

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier trainer')
    parser.add_argument('epochs', nargs='?', type=int, help="epochs count")
    parser.add_argument('--testonly', action='store_true', help="test only")
    parser.add_argument('--summaryonly', action='store_true', help="summary only")
    args = parser.parse_args()

    assert((args.epochs!=None)or(args.testonly))
    
    out_dir = os.path.join('model','board_animal')

    csv_path = os.path.join('label','board_animal.csv')
    sample_list = _util.read_csv(csv_path,aba.CSV_COL_LIST)

    label_list = [ i['label'] for i in sample_list ]
    label_list = sorted(list(set(label_list)))
    label_count = len(label_list)

    random.shuffle(sample_list)

    test_count = int(len(sample_list)/10)

    train_sample_list = sample_list[test_count:-test_count]
    test_sample_list  = sample_list[:test_count]
    valid_sample_list = sample_list[-test_count:]
    
    model = classifier_board_animal_model.create_model(label_count)
    model.summary()
    
    if args.summaryonly:
        quit()
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    weight_fn = os.path.join(out_dir,'weight.hdf5')

    if not args.testonly:
        _util.reset_dir(out_dir)
        json_path = os.path.join(out_dir,'data.json')
        with open(json_path,'w') as fout:
            json.dump({
                'label_list': label_list
            },fout,indent=2)
            fout.write('\n')

        train_img_list, train_label_onehot_list = sample_list_to_data_set(train_sample_list,label_list)
        valid_img_list, valid_label_onehot_list = sample_list_to_data_set(valid_sample_list,label_list)
        
        epochs = args.epochs
        checkpointer = ModelCheckpoint(filepath=weight_fn, verbose=1, save_best_only=True)
        model.fit(train_img_list, train_label_onehot_list,
            validation_data=(valid_img_list, valid_label_onehot_list),
            epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

    
    model.load_weights(weight_fn)

    test_img_list,  test_label_onehot_list  = sample_list_to_data_set(test_sample_list ,label_list)
    start_time = time.time()
    test_predictions = [np.argmax(model.predict(np.expand_dims(img_list, axis=0))) for img_list in test_img_list]
    end_time = time.time()
    test_accuracy = np.sum(np.array(test_predictions)==np.argmax(test_label_onehot_list, axis=1))/len(test_predictions)
    print('Test accuracy: %.4f' % test_accuracy)
    time_used = int((end_time-start_time)*1000)
    print('{}/{}s = {}/s'.format(len(test_img_list),time_used,len(test_img_list)*1000/time_used))
