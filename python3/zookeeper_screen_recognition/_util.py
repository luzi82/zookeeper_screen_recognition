import os
import numpy as np
import cv2
from functools import lru_cache
import csv
import shutil

def get_label_state_list():
    label_state_path = os.path.join('label','state')
    label_name_list = os.listdir(label_state_path)
    label_name_list = filter(lambda v:os.path.isfile(os.path.join(label_state_path,v)),label_name_list)
    label_name_list = filter(lambda v:v.endswith('.txt'),label_name_list)
    label_name_list = [ i[:-4] for i in label_name_list]
    label_name_list = sorted(label_name_list)
    return label_name_list

def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

PHI = (1+5**0.5)/2

def load_img(fn):
    img = cv2.imread(fn).astype('float32')*2/255-1
    return img

def xy_layer(w,h):
    xx = np.array(list(range(w))).astype('float32')*2/(w-1)-1
    xx = np.tile(xx,h)
    xx = np.reshape(xx,(h,w,1))
    yy = np.array(list(range(h))).astype('float32')*2/(h-1)-1
    yy = np.repeat(yy,w)
    yy = np.reshape(yy,(h,w,1))
    xxyy = np.append(xx,yy,axis=2)
    return xxyy

def xy1_layer(w,h):
    ret = xy_layer(w,h)

    oo = np.ones(shape=(h,w,1),dtype=np.float)
    ret = np.append(ret,oo,axis=2)

    return ret

@lru_cache(maxsize=4)
def get_raw_image_timestamp_list():
    raw_image_timestamp_list = os.listdir('raw_image')
    raw_image_timestamp_list = filter(lambda v:os.path.isdir(os.path.join('raw_image',v)),raw_image_timestamp_list)
    raw_image_timestamp_list = [ int(i) for i in raw_image_timestamp_list ]
    return raw_image_timestamp_list

def get_timestamp(v):
    return max(filter(lambda i:i<v,get_raw_image_timestamp_list()))

def read_csv(fn,col_name_list):
    ret = []
    with open(fn,'r') as fin:
        for line in csv.reader(fin):
            assert(len(line)==len(col_name_list))
            ret.append({col_name_list[i]:line[i] for i in range(len(col_name_list))})
    return ret

def write_csv(fn,v_dict_list,col_name_list,sortkey_func=None):
    if sortkey_func != None:
        v_dict_dict = { sortkey_func(v_dict):v_dict for v_dict in v_dict_list }
        v_dict_dict_key_sort = sorted(v_dict_dict.keys())
        v_dict_list = [ v_dict_dict[k] for k in v_dict_dict_key_sort ]
    with open(fn,'w') as fout:
        csv_out = csv.writer(fout)
        for v_dict in v_dict_list:
            csv_out.writerow([v_dict[col_name] for col_name in col_name_list])

def reset_dir(out_dir):
    shutil.rmtree(out_dir,ignore_errors=True)
    os.makedirs(out_dir)

def readlines(fn):
    with open(fn,'rt',encoding='utf-8') as fin:
        ret = fin.readlines()
    ret = [ i.strip() for i in ret ]
    return ret
