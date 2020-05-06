import argparse
import os
import numpy as np
import pandas as pd

from PIL import Image, ImageChops
from tensorflow.keras.preprocessing.image import load_img


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--output', type=str, required=True)
    args = ap.parse_args()

    data = {}
    data['image'] = []
    data['shape'] = []
    data['greyscale'] = []
    for folder, b, files in os.walk(args.data):
        for image in files:
            if image.startswith('./'):
                image = image[2:]

            image_path = os.path.join(folder, image)
            ext = image_path.split('.')[-1]

            
            try:
                if image_path.count('.') < 3 and ext not in ['zip', '7z', 'py', 'py~', 'db', 'txt']:
                    numpy_image = np.asarray(load_img(image_path))
                    data['image'].append(image_path)
                    data['shape'].append(numpy_image.shape)
                    data['greyscale'].append(numpy_image.std(axis=2).mean() == 0)
                    
            except Exception as e:
                print("Exception {}".format(e))


    pd.DataFrame(data).to_csv(args.output, index=False)
