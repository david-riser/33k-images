import argparse
import numpy as np
import os
import sys

from PIL import Image

# Add project_core
PROJECT_DIR = os.path.abspath('../')
sys.path.append(PROJECT_DIR)


from project_core.duplicates import hash_search
from project_core.utils import print_directory_stats

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=str)
    return parser.parse_args()

def main(args):

    print_directory_stats(args.data)
        
    for folder, _, files in os.walk(args.data):
        print('[INFO] Processing images from {}'.format(folder))

        if len(files) > 0:
            paths = [os.path.join(folder, f) for f in files]
            matches = hash_search(paths)
            print(matches)
        
    print_directory_stats(args.data)
    
if __name__ == "__main__":
    main(get_args())
