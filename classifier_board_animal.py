import os
import json
import cv2
import numpy as np
from functools import lru_cache
import _util

MODEL_PATH = os.path.join('model','board_animal')
WEIGHT_FILENAME = 'weight.hdf5'
DATA_FILENAME   = 'data.json'

import classifier_board_animal_model

class BoardAnimalClassifier:

    def __init__(self, model_path):
        weight_path = os.path.join(model_path, WEIGHT_FILENAME)
        data_path   = os.path.join(model_path, DATA_FILENAME)
        with open(data_path,'r') as fin:
            self.data = json.load(fin)
        self.model = classifier_board_animal_model.create_model(len(self.data['label_list']))
        self.model.load_weights(weight_path)

    def predict(self, img):
        img_list = classifier_board_animal_model.preprocess_img(img)
        p_list_list = self.model.predict(img_list)
        score_list = np.max(p_list_list,axis=1)
        label_idx_list = np.argmax(p_list_list,axis=1)
        return [self.data['label_list'][label_idx] for label_idx in label_idx_list], score_list

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
