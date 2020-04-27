""" 

Plot shapes.

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

    df = pd.read_csv(args.input)
    df['height'] = df['shape'].apply(lambda x: int(x.split(',')[0].split('(')[-1]))
    df['width'] = df['shape'].apply(lambda x: int(x.split(',')[1]))
    df['channels'] = df['shape'].apply(lambda x: int(x.split(',')[2].split(')')[0]))
    df['megapixels'] = df['height'] * df['width'] * df['channels']
    df['megapixels'] = df['megapixels'] / 1e6
    
    plt.figure(figsize=(8,6))
    plt.hist(df['megapixels'], bins=40, edgecolor='k')
    plt.grid(alpha=0.2)
    plt.xlabel('Megapixels')
    plt.savefig(args.output + '/hist_megapixels.png', bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8,6))
    plt.scatter(df['width'], df['height'], color='white', edgecolor='k')
    plt.grid(alpha=0.2)
    plt.xlabel('Width')
    plt.xlabel('Height')
    plt.savefig(args.output + '/scatter_shape.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,6))
    plt.hist(df['megapixels'], bins=40, edgecolor='k')
    plt.grid(alpha=0.2)
    plt.xlabel('Megapixels')
    plt.savefig(args.output + '/hist_megapixels.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,6))
    plt.hist(df['width'], edgecolor='k', bins=20)
    plt.grid(alpha=0.2)
    plt.ylabel('Width')
    plt.savefig(args.output + '/hist_width.png', bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8,6))
    plt.hist(df['height'], edgecolor='k', bins=20)
    plt.grid(alpha=0.2)
    plt.ylabel('height')
    plt.savefig(args.output + '/hist_height.png', bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8,6))
    plt.hist(df['channels'], edgecolor='k', bins=20)
    plt.grid(alpha=0.2)
    plt.ylabel('channels')
    plt.savefig(args.output + '/hist_channels.png', bbox_inches='tight')
    plt.close()

