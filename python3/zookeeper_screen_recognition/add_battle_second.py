import glob
import os
import sys
from . import _util

COL_NAME_LIST = ['fn','label']

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='add battle second label')
    args = parser.parse_args()

    data_fn = os.path.join('label','battle_second.csv')
    
    if os.path.isfile(data_fn):
        data_dict = _util.read_csv(data_fn,COL_NAME_LIST)
        data_dict = { i['fn']:i for i in data_dict }
    else:
        data_dict = {}

    input_dir = 'input'
    for label_name in os.listdir(input_dir):
        label_path = os.path.join(input_dir,label_name)
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
            data_dict[image_ori_path] = {'fn':image_ori_path,'label':label_name}

    _util.write_csv(data_fn,data_dict.values(),COL_NAME_LIST,lambda i:i['fn'])
