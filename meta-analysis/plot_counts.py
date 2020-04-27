""" 

Plot classes and images per class.

"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    args = parser.parse_args()

    folders = 0
    count = 0 
    data = {}
    data['label'] = []
    data['nimages'] = []
    for folder, _, files in os.walk(args.input):
        data['label'].append(folder.split('/')[-1])
        data['nimages'].append(len(files))

    df = pd.DataFrame(data)

    plt.figure(figsize=(8,6))
    plt.hist(df['nimages'], bins=np.linspace(0,5000,40), edgecolor='k')
    plt.grid(alpha=0.2)
    plt.xlabel('Images/Class')
    plt.savefig(args.output + '/hist_nimages.png', bbox_inches='tight')
    plt.close()

    index = np.argsort(df['nimages'])[::-1]
    plt.figure(figsize=(8,24))
    plt.barh(df['label'].iloc[index][:40], df['nimages'].iloc[index][:40], edgecolor='k')
    plt.ylabel('Class')
    plt.xlabel('Number of Images')
    plt.savefig(args.output + '/barh_nimages.png', bbox_inches='tight')
    plt.close()
    
