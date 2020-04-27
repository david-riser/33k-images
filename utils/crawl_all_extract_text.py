""" 

crawl_all_extract_text.py

Pull text out of all images.

"""

import argparse
import os
import pytesseract
import pickle

from PIL import Image

if __name__ == "__main__":

    workdir = os.path.abspath('../')
    print("Getting images from {}".format(workdir))

    exts = ['db', 'txt', 'py', 'py~', 'dat', 'pkl']

    for folder, _, files in os.walk(workdir):
        full_paths = [os.path.join(folder,file) for file in files]

        data = {}
        class_name = folder.split('/')[-1]
        if not os.path.exists(class_name + '.pkl'):
            print("Crawling {} with {} images...".format(class_name, len(full_paths)))
            for image in full_paths:

                ext = image.split('.')[-1]
                if ext not in exts:
                    short_folder = "/".join(image.split('/')[-2:])
                    text = pytesseract.image_to_string(Image.open(image))
                    data[short_folder] = text

            with open('{}.pkl'.format(class_name), 'wb') as ofile:
                pickle.dump(data, ofile)

        
