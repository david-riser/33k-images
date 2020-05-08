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
import json

from PIL import Image

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, type=str)
    ap.add_argument('--output', required=True, type=str)
    return ap.parse_args()

def get_images_list(path, image_extensions=['jpg', 'png', 'jpeg']):
    images = []
    for folder, _, files in os.walk(path):
        images.extend([os.path.join(folder,image) for image in files])
    return [image for image in images if image.split('.')[-1].lower() in image_extensions]

def process_image(image):

    plt.close()
    plt.figure(figsize=(4,3))
    img, has_image = load_image(image)
            
    if has_image:
        plt.imshow(img)
        plt.pause(0.05)

    c = None
    while c not in [0, 1]:
        try:
            c = int(input('Enter label: '))
        except:
            c = None
            
    return c
    
def load_image(image_path):
    try:
        image = Image.open(image_path)
        return image, True
    except:
        return np.zeros((224,224,3)), False
    
if __name__ == "__main__":

    args = get_args()
    print('[INFO] Getting images from {}'.format(args.images))
    images_list = get_images_list(args.images)
    print('[INFO] Found {} images.'.format(len(images_list)))

    # Enable interactive plot mode and
    # begin collecting data from the human.
    plt.ion()

    labels = []
    for img in images_list:
        label = process_image(img)
        labels.append(label)

    pd.DataFrame({'file':images_list, 'label':labels}).to_csv(args.output, index=False)
