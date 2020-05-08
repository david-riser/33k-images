import argparse
import numpy as np
import os
import sys

from PIL import Image

# Add project_core
PROJECT_DIR = os.path.abspath('../')
sys.path.append(PROJECT_DIR)

from project_core.utils import print_directory_stats

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=str)
    return parser.parse_args()

def are_the_same(image1, image2):
    if image1.shape != image2.shape:
        return False
    else:
        return np.sum((image2 - image1)**2) == 0

def main(args):

    print_directory_stats(args.data)
        
    for folder, _, files in os.walk(args.data):
        print('[INFO] Processing images from {}'.format(folder))

        loaded_images = {}
        removed = []
        for i1, f1 in enumerate(files):
            path1 = os.path.join(folder, f1)

            # Look to see if we can open the image
            # and if so, look through all the others
            # and delete them.  This is only looking
            # in the same folder. 
            try:
                if path1 not in loaded_images:
                    loaded_images[path1] = np.asarray(Image.open(path1))

                for f2 in files[i1:]:
                    path2 = os.path.join(folder, f2)
                    if path1 != path2 and path2 not in removed:

                        try:
                            if path2 not in loaded_images:
                                loaded_images[path2] = np.asarray(Image.open(path2))
                                
                            if are_the_same(loaded_images[path1], loaded_images[path2]):
                                print('[INFO] {} = {}!'.format(path1, path2))
                                os.remove(path2)
                                removed.append(path2)
                                
                        # Maybe the second image does not open
                        except Exception as e:
                            print(path2, e)

            # Maybe the first image does not open
            except Exception as e:
                print(path1, e)

    print_directory_stats(args.data)
    
if __name__ == "__main__":
    main(get_args())
