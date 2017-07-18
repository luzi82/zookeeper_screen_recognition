import glob
import os
import sys
from . import _util
import shutil
from . import classifier_state

MODEL_PATH = 'model'

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state label util')
    parser.add_argument('timestamp', nargs='?', help='timestamp')
    parser.add_argument('--unknown_only', action='store_true', help="unknown_only")
    parser.add_argument('--disagree', action='store_true', help="disagree")
    args = parser.parse_args()
    
    assert(not(args.unknown_only and args.disagree))

    _util.reset_dir('output')
    
    clr = classifier_state.StateClassifier(MODEL_PATH)

    fn_list = glob.glob(os.path.join('raw_image','*','*','*.png'))

    if args.timestamp:
        fn_list = list(filter(lambda v:args.timestamp in v,fn_list))

    img_fn_filter_set = set()
    known_dict = {}
    if args.unknown_only or args.disagree:
        label_state_list = _util.get_label_state_list()
        for label_state in label_state_list:
            img_fn_list = _util.readlines(os.path.join('label','state','{}.txt'.format(label_state)))
            img_fn_filter_set = img_fn_filter_set | set(img_fn_list)
            for img_fn in img_fn_list:
                known_dict[img_fn] = label_state

    for img_fn in fn_list:
        if (args.unknown_only) and (img_fn in img_fn_filter_set):
            continue
        if (args.disagree) and (not(img_fn in img_fn_filter_set)):
            continue
        _, img_fn_t = os.path.split(img_fn)
        img = classifier_state.load_img(img_fn)
        label, _ = clr.get_state(img)
        if (args.disagree) and (label==known_dict[img_fn]):
            continue
        timestamp = str(_util.get_timestamp(int(img_fn_t[:-4])))
        out_fn_dir = os.path.join('output',timestamp,label)
        out_fn = os.path.join(out_fn_dir,img_fn_t)
        _util.makedirs(out_fn_dir)
        shutil.copyfile( img_fn, out_fn )
