import glob
import os
import sys

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state label util')
    args = parser.parse_args()

    raw_image_timestamp_list = os.listdir('raw_image')
    raw_image_timestamp_list = filter(lambda v:os.path.isdir(os.path.join('raw_image',v)),raw_image_timestamp_list)
    raw_image_timestamp_list = [ int(i) for i in raw_image_timestamp_list ]
    #print(raw_image_timestamp_list,file=sys.stderr)
    
    def get_timestamp(v):
        return max(filter(lambda i:i<v,raw_image_timestamp_list))
    
    for label_name in os.listdir('input'):
        label_path = os.path.join('input',label_name)
        path_list = []
        if not os.path.isdir(label_path):
            continue
        for image_fn in os.listdir(label_path):
            if not image_fn.endswith('.png'):
                continue
            image_idx = image_fn[:-4]
            image_idx = int(image_idx)
            image_timestamp=get_timestamp(image_idx)
            image_ori_path = os.path.join('raw_image',str(image_timestamp),str(int(image_idx/100000)),'{}.png'.format(image_idx))
            if not os.path.isfile(image_ori_path):
                print('{} not found'.format(image_ori_path),file=sys.stderr)
                continue
            path_list.append(image_ori_path)
            #print('{} {} {}'.format(image_timestamp,image_idx,label_name))

        label_path = os.path.join('label','state','{}.txt'.format(label_name))

        path_list_ori = []
        if os.path.isfile(label_path):
            with open(label_path, mode='rt', encoding='utf-8') as fin:
                path_list_ori = fin.readlines()

        path_list = path_list + path_list_ori
        path_list = list(set(path_list))
        path_list = sorted(path_list)

        with open(label_path, mode='wt', encoding='utf-8') as fout:
            for path in path_list:
                fout.write('{}\n'.format(path))
