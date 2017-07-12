import glob
import os
import sys
import _util
import json

CSV_COL_LIST = ['fn','idx','pos','label']

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state label util')
    args = parser.parse_args()

    csv_path = os.path.join('label','board_animal.csv')
    if os.path.isfile(csv_path):
        entry_dict = { '{} {}'.format(i['fn'],i['pos']):i for i in _util.read_csv(csv_path,CSV_COL_LIST) }
    else:
        entry_dict = {}

#    json_path = os.path.join('label','board_animal.json')
#    if os.path.isfile(json_path):
#        with open(json_path,'r') as fin:
#            data_dict = json.load(fin)
#    else:
#        data_dict = {
#            'label_list' : []
#        }

    for label_name in os.listdir('input'):
        label_path = os.path.join('input',label_name)
        if not os.path.isdir(label_path):
            continue
        #data_dict['label_list'].append(label_name)
        for image_fn in os.listdir(label_path):
            if not image_fn.endswith('.png'):
                continue
            image_fn = image_fn[:-4]
            image_fn = image_fn.split('-')
            assert(len(image_fn)==2)
            image_idx = int(image_fn[0])
            image_pos = image_fn[1]
            image_timestamp=_util.get_timestamp(image_idx)
            image_ori_path = os.path.join('raw_image',str(image_timestamp),str(int(image_idx/100000)),'{}.png'.format(image_idx))
            if not os.path.isfile(image_ori_path):
                print('{} not found'.format(image_ori_path),file=sys.stderr)
                continue
            key = '{} {}'.format(image_ori_path,image_pos)
            entry_dict[key]={
                'fn':image_ori_path,
                'idx':image_idx,
                'pos':image_pos,
                'label':label_name
            }

    #data_dict['label_list'] = sorted(list(set(data_dict['label_list'])))

    entry_key_list = sorted(entry_dict.keys())
    entry_list = [ entry_dict[i] for i in entry_key_list ]
    _util.write_csv(csv_path,entry_list,CSV_COL_LIST)

#    with open(json_path,'w') as fout:
#        json.dump(data_dict,fout,indent=2)
#        fout.write('\n')
