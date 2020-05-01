""" 

I do not want to look at 33k images.  I want to 
estimate some basic information about the data.  
I will do it stochastically by selecting random
samples and tracking the number of occurances.

When I get tired I will call it good enough!

"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from PIL import Image

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, type=str)
    ap.add_argument('--output', type=str)
    ap.add_argument('--rows', type=int, default=6)
    ap.add_argument('--cols', type=int, default=6)
    return ap.parse_args()

def get_quantities():
    quantities = []
    while True:
        user_text = input('Enter a quantity to track (press enter if done): ')
        if user_text is not '':
            quantities.append(user_text)
        else:
            break
    print('[INFO] Tracking {}'.format(quantities))
    return quantities

def get_images_list(path, image_extensions=['jpg', 'png', 'jpeg']):
    images = []
    for folder, _, files in os.walk(path):
        images.extend([os.path.join(folder,image) for image in files])
    return [image for image in images if image.split('.')[-1].lower() in image_extensions]
    
if __name__ == "__main__":

    args = get_args()
    quantities = get_quantities()

    print('[INFO] Getting images from {}'.format(args.images))
    images_list = get_images_list(args.images)
    print('[INFO] Found {} images.'.format(len(images_list)))
    
    
    
