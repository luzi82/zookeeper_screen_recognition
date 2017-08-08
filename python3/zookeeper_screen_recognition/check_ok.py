import os
from . import _util

if __name__ == '__main__':
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description='state')
    args = parser.parse_args()

    _util.reset_dir('output')

    sample_csv_path = os.path.join('label','ok.csv')
    sample_list = _util.read_csv(sample_csv_path,['fn','y','h'])

    for sample in sample_list:
        label = '{}+{}'.format(sample['y'],sample['h'])
        output_path = os.path.join('output',label)
        _util.makedirs(output_path)
        _, fn2 = os.path.split(sample['fn'])
        shutil.copyfile( sample['fn'], os.path.join(output_path,fn2) )
