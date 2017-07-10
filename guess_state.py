import glob
import os
import sys
import _util
import shutil
import classifier_state

MODEL_PATH = 'model'

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state label util')
    parser.add_argument('timestamp', nargs='?', help='timestamp')
    parser.add_argument('--unknown_only', action='store_true', help="unknown_only")
    args = parser.parse_args()

    shutil.rmtree('guess',ignore_errors=True)
    
    sc = classifier_state.StateClassifier(MODEL_PATH)

    if args.timestamp:
        timestamp_list = [args.timestamp]
    else:
        raw_image_timestamp_list = os.listdir('raw_image')
        raw_image_timestamp_list = filter(lambda v:os.path.isdir(os.path.join('raw_image',v)),raw_image_timestamp_list)
        timestamp_list = raw_image_timestamp_list

    img_fn_filter_set = set()
    if args.unknown_only:
        label_state_list = _util.get_label_state_list()
        for label_state in label_state_list:
            with open(os.path.join('label','state','{}.txt'.format(label_state))) as fin:
                img_fn_list = fin.readlines()
            img_fn_list = [ img_fn.strip() for img_fn in img_fn_list ]
            img_fn_filter_set = img_fn_filter_set | set(img_fn_list)

    for timestamp in timestamp_list:
        raw_image_ts_path = os.path.join('raw_image',timestamp)
        
        img_fn_list = glob.glob(os.path.join(raw_image_ts_path,'*','*.png'))
        for img_fn in img_fn_list:
            if img_fn in img_fn_filter_set:
                continue
            _, img_fn_t = os.path.split(img_fn)
            img = classifier_state.load_img(img_fn)
            label, _ = sc.get_state(img)
            out_fn_dir = os.path.join('guess',timestamp,label)
            out_fn = os.path.join(out_fn_dir,img_fn_t)
            _util.makedirs(out_fn_dir)
            shutil.copyfile( img_fn, out_fn )
