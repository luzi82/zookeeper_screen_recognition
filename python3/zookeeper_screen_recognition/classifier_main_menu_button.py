import os
import json
import cv2
import numpy as np
from functools import lru_cache
from . import _util

MODEL_PATH = os.path.join('model','main_menu_button')
WEIGHT_FILENAME = 'weight.hdf5'
DATA_FILENAME   = 'data.json'

from . import classifier_main_menu_button_model

class MainMenuButtonClassifier:

    def __init__(self, model_path):
        weight_path = os.path.join(model_path, WEIGHT_FILENAME)
        data_path   = os.path.join(model_path, DATA_FILENAME)
        with open(data_path,'r') as fin:
            self.data = json.load(fin)
        self.model = classifier_main_menu_button_model.create_model(len(self.data['label_list']))
        self.model.load_weights(weight_path)

    def predict(self, img):
        import time
        tt = time.time()
        img = classifier_main_menu_button_model.preprocess_img(img)
        p_list = self.model.predict(np.expand_dims(img, axis=0))
        score = np.max(p_list)
        label_idx = np.argmax(p_list)
        return self.data['label_list'][label_idx], score

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier')
    parser.add_argument('img_file', help="img_file")
    args = parser.parse_args()
    
    img = _util.load_img(args.img_file)

    sc = MainMenuButtonClassifier(MODEL_PATH)

    label, score = sc.predict(img)
    print('{} {}'.format(label, score))
