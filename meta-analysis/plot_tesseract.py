""" 

Create meta plots based on the
output of tesseract.

"""

import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

def process_file(pickle_file):
    """ Open and process one file. """
    with open(pickle_file, 'rb') as input_file:
        data = pickle.load(input_file)

        output = {}
        output['image'] = []
        output['n_words'] = []
        output['has_label'] = []
        for image, words in data.items():
            output['image'].append(image)
            output['n_words'].append(len(words))

            if 'unclassified' in words.lower():
                output['has_label'].append(True)
            elif 'classified' in words.lower():
                output['has_label'].append(True)
            else:
                output['has_label'].append(False)
                
    return pd.DataFrame(output)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    args = parser.parse_args()

    print('[INFO] Reading from {} into {}.'.format(args.input, args.output))

    data_store = []
    for pickle_file in glob.glob('{}/*.pkl'.format(args.input)):
        print('[INFO] Processing {}'.format(pickle_file))
        data_store.append(process_file(pickle_file))

    data = pd.concat(data_store) 

    plt.figure(figsize=(8,6))
    plt.hist(data['n_words'], bins=np.linspace(0,20,20), edgecolor='k')
    plt.grid(alpha=0.2)
    plt.savefig(args.output + '/hist_nwords_zoom.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,6))
    plt.hist(data['n_words'], bins=np.linspace(0,200,20), edgecolor='k')
    plt.grid(alpha=0.2)
    plt.savefig(args.output + '/hist_nwords.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,6))
    plt.hist(data['has_label'], bins=np.linspace(0,2,2), edgecolor='k')
    plt.grid(alpha=0.2)
    plt.savefig(args.output + '/hist_has_label.png', bbox_inches='tight')
    plt.close()

    
