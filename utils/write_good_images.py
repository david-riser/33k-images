import argparse
import pandas as pd
import os

from PIL import Image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--missing', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    return parser.parse_args()

def read_missing(filename):
    with open(filename, 'r') as inputfile:
        missing = [x.strip()[2:] for x in inputfile.readlines()]
    return missing

def try_to_open(image_path):
    try:
        _ = Image.open(image_path)
        return True
    except:
        return False

if __name__ == "__main__":

    args = get_args()
    missing = read_missing(args.missing)

    prohibited_extensions = ('txt', 'py', 'db', 'dat', 'csv')
    
    labels, files = [], []
    for folder, _, image_paths in os.walk(args.data_dir):
        abs_folder = os.path.abspath(folder)
        label = abs_folder.split('/')[-1]

        for image_path in image_paths:
            extension = image_path.split('.')[-1]
            full_name = label + '/' + image_path
            if extension not in prohibited_extensions and full_name not in missing:
                if try_to_open(abs_folder + '/' + image_path):
                    labels.append(label)
                    files.append(label + '/' + image_path)
                else:
                    print(f'{image_path} does not open.')

    data = pd.DataFrame({'label':labels, 'file':files})
    data.to_csv(args.output, index=False)
    


    
