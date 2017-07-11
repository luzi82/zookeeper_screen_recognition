import os
import json
import cv2
import numpy as np
from functools import lru_cache
import _util

MODEL_PATH = 'model'
WEIGHT_FILENAME = 'classifier_ok.hdf5'

import classifier_ok_model

WIDTH  = classifier_ok_model.WIDTH
HEIGHT = classifier_ok_model.HEIGHT

load_img = _util.load_img

preprocess_img = classifier_ok_model.preprocess_img

_SCORE_LIST = np.array([i*2+128+.5 for i in range(64)]).astype(np.float)

class OkClassifier:

    def __init__(self, model_path):
        weight_path = os.path.join(model_path, WEIGHT_FILENAME)
        self.model = classifier_ok_model.create_model()
        self.model.load_weights(weight_path)

    def get_ok(self, img):
        img = preprocess_img(img)
        p_list = self.model.predict(np.expand_dims(img, axis=0))
        score_list = p_list.reshape(HEIGHT)
        #p_list = (p_list+1)/2
        p_list = np.maximum(p_list,0)
        p_list_base = np.sum(p_list)
        if p_list_base <= 0:
            return None, score_list
        p_list = np.reshape(p_list,(1,HEIGHT))
        p_list = p_list * _SCORE_LIST
        center = np.sum(p_list) / p_list_base
        return center, score_list

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier')
    parser.add_argument('img_file', help="img_file")
    args = parser.parse_args()
    
    img = load_img(args.img_file)

    sc = OkClassifier(MODEL_PATH)

    center, score_list = sc.get_ok(img)
    print('{} {}'.format(center, list(score_list)))
