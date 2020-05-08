import argparse
import numpy as np
import os
import sys

from PIL import Image, UnidentifiedImageError

# Add project_core
PROJECT_DIR = os.path.abspath('../')
sys.path.append(PROJECT_DIR)

TEMPLATE = "DUMP/2_DUMP_462a383b9e48765d0ee96e4d5a3cfb30b30ed586fd5e011c411ee0840c88702c-0.jpg"

from project_core.utils import print_directory_stats

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=str)
    return parser.parse_args()

def main(args):

    print_directory_stats(args.data)

    try:
        template_image = np.asarray(Image.open(
            os.path.join(args.data, TEMPLATE)
        ))
    except:
        print('[FATAL] 2_delete_missing.py can only run one time.  There is no template!')
        
    for folder, _, files in os.walk(args.data):
        for f in files:
            path = os.path.join(folder, f)
            try:
                img = np.asarray(Image.open(path))

                if img.shape == template_image.shape:
                    if np.sum((img - template_image)**2) == 0:
                        os.remove(path)
                
            except Exception as e:
                print(path, e)

    print_directory_stats(args.data)
    
if __name__ == "__main__":
    main(get_args())
