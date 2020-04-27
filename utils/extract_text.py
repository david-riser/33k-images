""" 

extract_text.py

Pull text out of an image.

"""

import argparse
import os
import pytesseract

from PIL import Image

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', required=True, type=str)
    args = ap.parse_args()

    cwd = os.path.abspath('.')
    folder_path = os.path.normpath(os.path.join(cwd, args.folder))
    print("Getting images from {}".format(folder_path))

    for image in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image)
        text = pytesseract.image_to_string(Image.open(image_path))
        print(image, text)
