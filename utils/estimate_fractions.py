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
    ap.add_argument('--rows', type=int, default=6)
    ap.add_argument('--cols', type=int, default=6)
    return ap.parse_args()

def get_quantities():
    quantities = {}
    while True:
        user_text = input('Enter a quantity to track (press enter if done): ')
        if user_text is not '':
            quantities[user_text] = 0 
        else:
            break
    print('[INFO] Tracking {}'.format(quantities))
    quantities['total'] = 0
    return quantities

def get_images_list(path, image_extensions=['jpg', 'png', 'jpeg']):
    images = []
    for folder, _, files in os.walk(path):
        images.extend([os.path.join(folder,image) for image in files])
    return [image for image in images if image.split('.')[-1].lower() in image_extensions]

def save_quantities(quantities, output):
    json_output = {}
    json_output['data'] = []
    json_output['total'] = quantities['total']
    
    for key, value in quantities.items():
        if key is not "total":
            data = {
                'quantity':key,
                'occurances':value,
                'estimated_probability':float(value) / float(quantities['total'])
            }
            json_output['data'].append(data)

    with open(output, 'w') as out:
        json.dump(json_output, out, indent=4)

def process_batch(images_list, indices, quantities, rows, cols):

    images = [images_list[i] for i in indices]
    batch_size = rows*cols
    assert(len(images) == batch_size)
    
    plt.figure(figsize=(12,8))
    for row in range(rows):
        for col in range(cols):
            index = col + row * cols
            image, has_image = load_image(images[index])
            
            if has_image:
                plt.subplot(rows, cols, index + 1)
                plt.imshow(image)
                plt.pause(0.05)
                
    plt.tight_layout()

    # Ask the user
    quantities['total'] += batch_size

    for key,value in quantities.items():
        if key is not "total":
            quantities[key] += int(input('Enter the number of {} occurances:'.format(key)))

    plt.close()

def load_image(image_path):
    try:
        image = Image.open(image_path)
        return image, True
    except:
        return np.zeros((224,224,3)), False
    
if __name__ == "__main__":

    args = get_args()
    quantities = get_quantities()

    print('[INFO] Getting images from {}'.format(args.images))
    images_list = get_images_list(args.images)
    print('[INFO] Found {} images.'.format(len(images_list)))

    batch_size = args.rows * args.cols
    possible_indices = np.arange(0,len(images_list),1)

    # Enable interactive plot mode and
    # begin collecting data from the human.
    plt.ion()
    while True:
        indices = np.random.choice(possible_indices, batch_size, replace=False)

        quit_option = input('Continue (enter for yes, anything else for no)?')
        if quit_option is "":
            process_batch(images_list, indices, quantities, args.rows, args.cols)
        else:
            save_quantities(quantities, args.output)
            exit()

    
