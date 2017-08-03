import glob
import os
import sys
from . import _util
import shutil
from . import classifier_state
from . import classifier_battle_second_model
from . import classifier_battle_second
from . import add_battle_second
import cv2
import numpy as np
import json
import time

#ICON_COUNT_2 = classifier_battle_second_model.ICON_COUNT_2
#SIZE = 8

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state label util')
    parser.add_argument('timestamp', nargs='?', help='timestamp')
    parser.add_argument('--unknown_only', action='store_true', help="unknown_only")
    parser.add_argument('--disagree', action='store_true', help="disagree")
    parser.add_argument('--json', action='store_true', help="json output")
    args = parser.parse_args()
    
    assert(not(args.unknown_only and args.disagree))

    _util.reset_dir('output')
    
    clr = classifier_battle_second.BattleSecondClassifier(classifier_battle_second.MODEL_PATH)

    fn_list = _util.readlines(os.path.join('label','state','battle.txt'))

    if args.timestamp:
        fn_list = list(filter(lambda v:args.timestamp in v,fn_list))

    if args.unknown_only or args.disagree:
        known_list = _util.read_csv(os.path.join('label','battle_second.csv'),add_battle_second.COL_NAME_LIST)
        known_dict = {i['fn']: i for i in known_list}
        known_list = [i['fn'] for i in known_list]
    else:
        known_list = []
        known_dict = {}

    if args.json:
        j_out = []

    fn_list_len = len(fn_list)
    for fn in fn_list:
        if args.unknown_only and (fn in known_list):
            continue
        if args.disagree and (fn not in known_list):
            continue
        img = _util.load_img(fn)
        ttime = time.time()
        predict, _ = clr.predict(img)
        ttime = time.time() - ttime
        if fn_list_len == 1:
            print('time consumed: {}ms'.format(int(ttime*1000)))
        if not args.json:
            known_key = fn
            if (args.unknown_only) and (known_key in known_list):
                continue
            if (args.disagree) and (not(known_key in known_list)):
                continue
            if (args.disagree) and (predict == known_dict[known_key]['label']):
                continue
            _util.makedirs(os.path.join('output',predict))
            _, fn_out = os.path.split(fn)
            fn_out = fn_out[:-4]
            fn_out = os.path.join('output',predict,'{}.png'.format(fn_out))
            #print(fn_out)
            cv2.imwrite(fn_out,((img[:,:,:3]+1)*255/2).astype(np.uint8))
        if args.json:
            j_out.append({'fn':fn,'predict':predict})

    if args.json:
        json.dump({'guess_battle_second_list':j_out},sys.stdout,indent=2)
        sys.stdout.write('\n')
