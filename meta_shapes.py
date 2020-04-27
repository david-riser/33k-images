import argparse
import os
import numpy as np
import pandas as pd

from PIL import Image

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--output', type=str, required=True)
    args = ap.parse_args()

    data = {}
    data['image'] = []
    data['shape'] = []
    for folder, b, files in os.walk(args.data):
        for image in files:
            if image.startswith('./'):
                image = image[2:]

            image_path = os.path.join(folder, image)
            ext = image_path.split('.')[-1]

            
            try:
                if image_path.count('.') < 3 and ext not in ['zip', '7z', 'py', 'py~', 'db', 'txt']:
                    numpy_image = np.asarray(Image.open(image_path))

                    if len(numpy_image.shape) == 2:
                        numpy_image = numpy_image.reshape(*numpy_image.shape, 1)

                    data['image'].append(image_path)
                    data['shape'].append(numpy_image.shape)
                    
            except Exception as e:
                print("Exception {}".format(e))


    pd.DataFrame(data).to_csv(args.output, index=False)
