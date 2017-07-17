from zookeeper_screen_recognition import _util
from zookeeper_screen_recognition import classifier_board_animal_model
import cv2
import glob
import os
import shutil
import numpy as np

ICON_WIDTH  = classifier_board_animal_model.ICON_WIDTH
ICON_HEIGHT = classifier_board_animal_model.ICON_HEIGHT
ICON_COUNT = classifier_board_animal_model.ICON_COUNT
BOARD_WIDTH = ICON_WIDTH * ICON_COUNT
BOARD_HEIGHT = ICON_HEIGHT * ICON_COUNT
#ORI_HEIGHT = classifier_board_animal_model.ORI_HEIGHT
ORI_CROP_Y = classifier_board_animal_model.ORI_CROP_Y

SHAPE = (ICON_COUNT*ICON_COUNT,ICON_HEIGHT,ICON_WIDTH,3)

def preprocess_img(img):
#    y0 = round(img.shape[0]*ORI_CROP_Y/ORI_HEIGHT)
#    y1 = y0 + img.shape[1]
#    img = img[y0:y1,:,:]
#    assert(img.shape==(BOARD_HEIGHT,BOARD_WIDTH,3))
#    img_list = [img[i*ICON_HEIGHT:(i+1)*ICON_HEIGHT,j*ICON_WIDTH:(j+1)*ICON_WIDTH,:]for i in range(ICON_COUNT) for j in range(ICON_COUNT)]
#    img_list = np.array(img_list)
#    print(img_list.shape)
#    assert(img_list.shape==SHAPE)
#    return img_list
    ret = classifier_board_animal_model.preprocess_img(img)
    ret = ret[:,:,:,:3]
    return ret

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='extract board animal')
    parser.add_argument('img_ts', help='img_ts')
    args = parser.parse_args()
    
    fn_list = glob.glob(os.path.join('raw_image','*','*','{}.png'.format(args.img_ts)))
    assert(len(fn_list)==1)
    fn = fn_list[0]
    
    shutil.rmtree('output',ignore_errors=True)
    os.makedirs('output')

    img = _util.load_img(fn)
    img_list = preprocess_img(img)

    for i in range(len(img_list)):
        fn = '%s-%02d.png'%(args.img_ts,i)
        fn = os.path.join('output',fn)
        #print(fn)
        img = img_list[i]
        img = ((img+1)*255/2).astype(np.uint8)
        img = np.array(img,dtype=np.uint8)
        #print(img.shape)
        #print(img)
        cv2.imwrite(fn,img)
