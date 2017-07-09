import os

if __name__ == '__main__':
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description='state')
    parser.add_argument('state', help='state')
    args = parser.parse_args()

    shutil.rmtree('check',ignore_errors=True)
    os.makedirs('check')
    with open(os.path.join('label','state','{}.txt'.format(args.state)),'r') as fin:
        file_list = fin.readlines()
    file_list = [ i.strip() for i in file_list ]
    for f in file_list:
        _, t = os.path.split(f)
        shutil.copyfile( f, os.path.join('check',t) )
