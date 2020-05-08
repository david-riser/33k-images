import argparse
import os
import sys

# Add project_core
PROJECT_DIR = os.path.abspath('../')
sys.path.append(PROJECT_DIR)

from project_core.utils import print_directory_stats

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=str)
    return parser.parse_args()

def main(args):

    print_directory_stats(args.data)
        
    for folder, _, files in os.walk(args.data):
        if len(files) == 0 and folder != args.data:
            os.rmdir(folder)
        
    print_directory_stats(args.data)
    
if __name__ == "__main__":
    main(get_args())
