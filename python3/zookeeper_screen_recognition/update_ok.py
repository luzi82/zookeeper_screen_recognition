import glob
import os
import sys
from . import _util
import shutil
import csv

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state label util')
    args = parser.parse_args()

    label_ok_path = os.path.join('label','ok.csv')

    #raw_image_timestamp_list = os.listdir('raw_image')
    #raw_image_timestamp_list = filter(lambda v:os.path.isdir(os.path.join('raw_image',v)),raw_image_timestamp_list)
    #raw_image_timestamp_list = [ int(i) for i in raw_image_timestamp_list ]
    #print(raw_image_timestamp_list,file=sys.stderr)
    
    #def get_timestamp(v):
    #    return max(filter(lambda i:i<v,raw_image_timestamp_list))

    ok_list = []
    
    for label_ok in os.listdir('input'):
        ok_range = label_ok.split('+')
        if len(ok_range) != 2:
            continue
        label_path = os.path.join('input',label_ok)
        path_list = []
        if not os.path.isdir(label_path):
            continue
        for image_fn in os.listdir(label_path):
            if not image_fn.endswith('.png'):
                continue
            image_idx = image_fn[:-4]
            image_idx = int(image_idx)
            image_timestamp=_util.get_timestamp(image_idx)
            image_ori_path = os.path.join('raw_image',str(image_timestamp),str(int(image_idx/100000)),'{}.png'.format(image_idx))
            if not os.path.isfile(image_ori_path):
                print('{} not found'.format(image_ori_path),file=sys.stderr)
                continue
            ok_list.append({'fn':image_ori_path, 'y':ok_range[0], 'h':ok_range[1]})

    _util.write_csv(label_ok_path,ok_list,['fn','y','h'],lambda i:i['fn'])
