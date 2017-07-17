import os
from . import _util

if __name__ == '__main__':
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description='state')
    parser.add_argument('state', nargs='?', help='state')
    args = parser.parse_args()

    if args.state != None:
        state_list = [args.state]
    else:
        state_list = _util.get_label_state_list()

    shutil.rmtree('output',ignore_errors=True)
    os.makedirs('output')
    for state in state_list:
        with open(os.path.join('label','state','{}.txt'.format(state)),'r') as fin:
            file_list = fin.readlines()
        file_list = [ i.strip() for i in file_list ]
        for f in file_list:
            _, t = os.path.split(f)
            _util.makedirs(os.path.join('output',state))
            shutil.copyfile( f, os.path.join('output',state,t) )
