import os
from zookeeper_screen_recognition import _util
import add_board_animal as aba
from zookeeper_screen_recognition import classifier_board_animal_train
import cv2

if __name__ == '__main__':
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description='state')
    args = parser.parse_args()

    csv_path = os.path.join('label','board_animal.csv')
    sample_list = _util.read_csv(csv_path,aba.CSV_COL_LIST)

    img_list = classifier_board_animal_train.load_img_list_csv(sample_list)
    img_list = img_list[:,:,:,:3]
    img_list = (img_list+1)*255/2

    _util.reset_dir('output')

    for i in range(len(sample_list)):
        sample = sample_list[i]
        img = img_list[i]
        fn = '{}-{}.png'.format(sample['idx'],sample['pos'])
        out_dir = os.path.join('output',sample['label'])
        _util.makedirs(out_dir)
        fn = os.path.join(out_dir,fn)
        cv2.imwrite(fn,img)
    