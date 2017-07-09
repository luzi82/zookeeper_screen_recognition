import os
import json
import sys
import random
import cv2
import numpy as np
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import json
import classifier_state_model

WIDTH  = classifier_state_model.WIDTH
HEIGHT = classifier_state_model.HEIGHT

def sample_list_to_data_set(sample_list, label_count):
    fn_list = [ sample['fn'] for sample in sample_list ]
    img_list = load_img_list(fn_list, WIDTH, HEIGHT)
    label_idx_list = np.array([ sample['label_idx'] for sample in sample_list ])
    label_onehot_list = np_utils.to_categorical(label_idx_list, label_count)
    return img_list, label_onehot_list

def load_img_list(fn_list,width,height):
    img_list = [ load_img(fn, width, height) for fn in fn_list ]
    return np.array(img_list)

def load_img(fn, width, height):
    img = cv2.imread(fn).astype('float32')*2/255-1
    img = cv2.resize(img,dsize=(width,height),interpolation=cv2.INTER_AREA)
    return img

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier trainer')
    parser.add_argument('epochs', nargs='?', type=int, help="epochs count")
    parser.add_argument('--testonly', action='store_true', help="test only")
    args = parser.parse_args()

    assert((args.epochs!=None)or(args.testonly))

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

    model = classifier_state_model.create_model(label_count)
    model.summary()
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        
    if not args.testonly:
        j = {
            'label_name_list': label_name_list
        }
        with open(os.path.join('model','data.json'),'w') as fout:
            json.dump(j, fp=fout, indent=2)
            fout.write('\n')

        train_img_list, train_label_onehot_list = sample_list_to_data_set(train_sample_list,label_count)
        valid_img_list, valid_label_onehot_list = sample_list_to_data_set(valid_sample_list,label_count)
        
        epochs = args.epochs
        checkpointer = ModelCheckpoint(filepath='model/classifier_state.hdf5', verbose=1, save_best_only=True)
        model.fit(train_img_list, train_label_onehot_list,
            validation_data=(valid_img_list, valid_label_onehot_list),
            epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

    
    model.load_weights('model/classifier_state.hdf5')

    test_img_list,  test_label_onehot_list  = sample_list_to_data_set(test_sample_list ,label_count)
    test_predictions = [np.argmax(model.predict(np.expand_dims(img_list, axis=0))) for img_list in test_img_list]
    test_accuracy = np.sum(np.array(test_predictions)==np.argmax(test_label_onehot_list, axis=1))/len(test_predictions)
    print('Test accuracy: %.4f' % test_accuracy)
