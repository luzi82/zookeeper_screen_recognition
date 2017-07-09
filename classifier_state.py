import os
import json
import cv2
import numpy as np

MODEL_PATH = 'model'
WEIGHT_FILENAME = 'classifier_state.hdf5'
DATA_FILENAME   = 'data.json'

import classifier_state_model

_ex_ch_np_slot = [None]

def load_img(fn):
    img = cv2.imread(fn).astype('float32')*2/255-1
    h,w,_ = img.shape
    xx = np.array(list(range(w))).astype('float32')*2/(w-1)-1
    xx = np.tile(xx,h)
    xx = np.reshape(xx,(h,w,1))
    yy = np.array(list(range(h))).astype('float32')*2/(h-1)-1
    yy = np.repeat(yy,w)
    yy = np.reshape(yy,(h,w,1))
    xxyy = np.append(xx,yy,axis=2)
    img = np.append(img,xxyy,axis=2)
    return img

class StateClassifier:

    def __init__(self, model_path):
        weight_path = os.path.join(model_path, WEIGHT_FILENAME)
        data_path   = os.path.join(model_path, DATA_FILENAME)
        with open(data_path,'r') as fin:
            self.data = json.load(fin)
        self.model = classifier_state_model.create_model(len(self.data['label_name_list']))
        self.model.load_weights(weight_path)

    def get_state(self, img):
        #print(img.shape)
        img = cv2.resize(img,dsize=(classifier_state_model.WIDTH,classifier_state_model.HEIGHT),interpolation=cv2.INTER_AREA)
        p = self.model.predict(np.expand_dims(img, axis=0))
        score = np.max(p)
        label_idx = np.argmax(p)
        return self.data['label_name_list'][label_idx], score

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier')
    parser.add_argument('img_file', help="img_file")
    args = parser.parse_args()
    
    img = load_img(args.img_file)
    print(img.shape)

#    sc = StateClassifier(MODEL_PATH)
#
#    label, score = sc.get_state(img)
#    print('{} {}'.format(label, score))
