import os
import json
import cv2
import numpy as np

MODEL_PATH = 'model'
WEIGHT_PATH = os.path.join(MODEL_PATH,'classifier_state.hdf5')
DATA_PATH   = os.path.join(MODEL_PATH,'data.json')

import classifier_state_model

class StateClassifier:

    def __init__(self):
        with open(DATA_PATH,'r') as fin:
            self.data = json.load(fin)
        self.model = classifier_state_model.create_model(len(self.data['label_name_list']))

    def load_file(self):
        #self.model.load_weights(WEIGHT_PATH)
        self.model.load_weights('model/classifier_state.hdf5')

    def get_state(self, img):
        #print(img.shape)
        img = cv2.resize(img,dsize=(classifier_state_model.WIDTH,classifier_state_model.HEIGHT),interpolation=cv2.INTER_AREA)
        p = self.model.predict(np.expand_dims(img, axis=0))
        label_idx = np.argmax(p)
        return self.data['label_name_list'][label_idx]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier')
    parser.add_argument('img_file', help="img_file")
    args = parser.parse_args()
    
    img = cv2.imread(args.img_file).astype('float32')/255

    sc = StateClassifier()
    sc.load_file()

    print(sc.get_state(img))
