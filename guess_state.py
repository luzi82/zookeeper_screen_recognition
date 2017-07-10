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
    parser.add_argument('timestamp', help='timestamp')
    args = parser.parse_args()

    shutil.rmtree(os.path.join('guess',args.timestamp),ignore_errors=True)
    
    sc = classifier_state.StateClassifier(MODEL_PATH)

    raw_image_ts_path = os.path.join('raw_image',args.timestamp)
    
    img_fn_list = glob.glob(os.path.join(raw_image_ts_path,'*','*.png'))
    for img_fn in img_fn_list:
        _, img_fn_t = os.path.split(img_fn)
        img = classifier_state.load_img(img_fn)
        label, _ = sc.get_state(img)
        out_fn_dir = os.path.join('guess',args.timestamp,label)
        out_fn = os.path.join(out_fn_dir,img_fn_t)
        _util.makedirs(out_fn_dir)
        shutil.copyfile( img_fn, out_fn )
