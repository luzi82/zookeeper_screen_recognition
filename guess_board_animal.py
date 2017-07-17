import glob
import os
import sys
from zookeeper_screen_recognition import _util
import shutil
from zookeeper_screen_recognition import classifier_state
from zookeeper_screen_recognition import classifier_board_animal_model
from zookeeper_screen_recognition import classifier_board_animal
import add_board_animal
import cv2
import numpy as np
import json
import time

ICON_COUNT_2 = classifier_board_animal_model.ICON_COUNT_2
SIZE = 8

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state label util')
    parser.add_argument('timestamp', nargs='?', help='timestamp')
    parser.add_argument('--unknown_only', action='store_true', help="unknown_only")
    parser.add_argument('--json', action='store_true', help="json output")
    args = parser.parse_args()

    _util.reset_dir('output')
    
    clr = classifier_board_animal.BoardAnimalClassifier(classifier_board_animal.MODEL_PATH)

    fn_list = _util.readlines(os.path.join('label','state','battle.txt'))

    if args.timestamp:
        fn_list = list(filter(lambda v:args.timestamp in v,fn_list))

    if args.unknown_only:
        known_list = _util.read_csv(os.path.join('label','board_animal.csv'),add_board_animal.CSV_COL_LIST)
        known_list = ['{} {}'.format(i['fn'],i['pos']) for i in known_list]
    else:
        known_list = []

    if args.json:
        j_out = []

    fn_list_len = len(fn_list)
    for fn in fn_list:
        if len(list(filter(lambda v:fn in v,known_list))) >= ICON_COUNT_2:
            continue
        img = _util.load_img(fn)
        img_list = classifier_board_animal_model.preprocess_img(img)
        img_list = img_list[:,:,:,:3]
        img_list = ((img_list+1)*255/2).astype(np.uint8)
        ttime = time.time()
        predict_list, _ = clr.predict(img)
        ttime = time.time() - ttime
        if fn_list_len == 1:
            print('time consumed: {}ms'.format(int(ttime*1000)))
        if not args.json:
            for i in range(len(predict_list)):
                ii = '%02d'%i
                if '{} {}'.format(fn,ii) in known_list:
                    continue
                predict = predict_list[i]
                _util.makedirs(os.path.join('output',predict))
                _, fn_out = os.path.split(fn)
                fn_out = fn_out[:-4]
                fn_out = os.path.join('output',predict,'{}-{}.png'.format(fn_out,ii))
                #print(fn_out)
                cv2.imwrite(fn_out,img_list[i])
        if args.json:
            board = [[predict_list[i+j*SIZE] for j in range(SIZE)] for i in range(SIZE)]
            j_out.append({'fn':fn,'board':board})

    if args.json:
        json.dump({'guess_board_animal_list':j_out},sys.stdout,indent=2)
        sys.stdout.write('\n')
