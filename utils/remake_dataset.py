import argparse
import numpy as np
import os


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, type=str)
    ap.add_argument('--output', required=True, type=str)
    ap.add_argument('--dev_frac', type=float, default=0.1)
    ap.add_argument('--test_frac', type=float, default=0.1)
    return ap.parse_args()


if __name__ == "__main__":

    args = get_args()

    # Set up some boundaries
    prohibited_folders = ['tesseract', 'lists', 'scripts']
    prohibited_extensions = ['py', 'py~', 'pyc', 'txt', 'dat', 'pkl']

    # Used later to select randomly for train or
    # dev/test.
    threshold = args.dev_frac + args.test_frac
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        os.makedirs(os.path.join(args.output,'train'))
        os.makedirs(os.path.join(args.output,'dev'))
        os.makedirs(os.path.join(args.output,'test'))
        
    for folder, _, files in os.walk(args.data):
        root_folder = folder.split('/')[-1]
        os.makedirs(os.path.join(args.output,'train/' + root_folder))
        os.makedirs(os.path.join(args.output,'dev/' + root_folder))
        os.makedirs(os.path.join(args.output,'test/' + root_folder))
        
        if root_folder not in prohibited_folders:
            print('Working on {}.'.format(root_folder))


            for f in files:
                extension = f.split('.')[-1]
                if extension not in prohibited_extensions and f[0] != '.':
                    if (np.random.uniform() > threshold):
                        os.system('cp {} {}'.format(
                            os.path.join(folder, f),
                            os.path.join(args.output + '/train/' + root_folder, f)
                        ))
                    else:
                        test_threshold = args.test_frac / (args.dev_frac + args.test_frac)

                        if (np.random.uniform() > test_threshold):
                            os.system('cp {} {}'.format(
                                os.path.join(folder, f),
                                os.path.join(args.output + '/dev/' + root_folder, f)
                            ))
                        else:
                            os.system('cp {} {}'.format(
                                os.path.join(folder, f),
                                os.path.join(args.output + '/test/' + root_folder, f)
                            ))

    
