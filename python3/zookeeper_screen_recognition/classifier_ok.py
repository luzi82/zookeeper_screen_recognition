import os
import json
import cv2
import numpy as np
from functools import lru_cache
from . import _util

MODEL_PATH = os.path.join('model','ok')
WEIGHT_FILENAME = 'weight.hdf5'

from . import classifier_ok_model

SCREEN_HEIGHT = classifier_ok_model.SCREEN_HEIGHT
WIDTH  = classifier_ok_model.WIDTH
HEIGHT = classifier_ok_model.HEIGHT
CROP_Y0 = classifier_ok_model.CROP_Y0

load_img = _util.load_img

preprocess_img = classifier_ok_model.preprocess_img

_SCORE_LIST = np.array(list(range(SCREEN_HEIGHT)))
_SCORE_LIST = _SCORE_LIST[CROP_Y0:]
_SCORE_LIST = np.reshape(_SCORE_LIST,(HEIGHT,2))
_SCORE_LIST = np.mean(_SCORE_LIST,axis=1)

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
