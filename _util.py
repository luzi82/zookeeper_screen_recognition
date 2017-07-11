import os
import numpy as np
import cv2

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
