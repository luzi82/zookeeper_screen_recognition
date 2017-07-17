import os
import json
import cv2
import numpy as np
from functools import lru_cache
from . import _util

MODEL_PATH = os.path.join('model','board_animal')
WEIGHT_FILENAME = 'weight.hdf5'
DATA_FILENAME   = 'data.json'

from . import classifier_board_animal_model

class BoardAnimalClassifier:

    def __init__(self, model_path):
        weight_path = os.path.join(model_path, WEIGHT_FILENAME)
        data_path   = os.path.join(model_path, DATA_FILENAME)
        with open(data_path,'r') as fin:
            self.data = json.load(fin)
        self.model = classifier_board_animal_model.create_model(len(self.data['label_list']))
        self.model.load_weights(weight_path)

    def predict(self, img):
        import time
        tt = time.time()
        img_list = classifier_board_animal_model.preprocess_img(img)
        #print(str(time.time()-tt))
        p_list_list = self.model.predict(img_list)
        #print(str(time.time()-tt))
        score_list = np.max(p_list_list,axis=1)
        #print(str(time.time()-tt))
        label_idx_list = np.argmax(p_list_list,axis=1)
        #print(str(time.time()-tt))
        r0, r1 = [self.data['label_list'][label_idx] for label_idx in label_idx_list], score_list
        #print(str(time.time()-tt))
        return r0, r1

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier')
    parser.add_argument('img_file', help="img_file")
    args = parser.parse_args()
    
    img = _util.load_img(args.img_file)

    sc = BoardAnimalClassifier(MODEL_PATH)

    label_list, score_list = sc.predict(img)
    assert(len(label_list)==len(score_list))
    for i in range(len(label_list)):
        print('{} {} {}'.format(i,label_list[i],score_list[i]))
