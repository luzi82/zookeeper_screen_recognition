import os
import json
import cv2
import numpy as np

MODEL_PATH = 'model'
WEIGHT_FILENAME = 'classifier_state.hdf5'
DATA_FILENAME   = 'data.json'

import classifier_state_model

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
    
    img = cv2.imread(args.img_file).astype('float32')*2/255-1

    sc = StateClassifier(MODEL_PATH)

    label, score = sc.get_state(img)
    print('{} {}'.format(label, score))
