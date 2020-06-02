""" 

June 2, 2020

Plot images from the dev data to sorted by 
weights. 

"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import string

from tensorflow.keras.preprocessing.image import load_img


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataframe', required=True, type=str)
    parser.add_argument('--data', default='/home/ubuntu/data/dev', type=str)
    parser.add_argument('--clusters', type=int, default=5)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--worst', action='store_true')
    parser.add_argument('--output_dir', type=str, default='/home/ubuntu')
    parser.add_argument('--weight_col', type=str, default='weight')
    return parser.parse_args()


def main(args):

    # Get a dictionary that contains the mapping of
    # folders and their associated files.
    dataset = pd.read_csv(args.dataframe)
    
    # Choose randomly from the classes we have to
    # display some images.
    targets = random.sample(
        set(dataset['label'].unique()), args.clusters
    )
    print(targets)
    
    # Iterate on classes and plot the figure
    fig, axs = plt.subplots(
        nrows=args.clusters, ncols=args.samples,
        figsize=(12, 6), dpi=100,
        sharex=True, sharey=True,
        gridspec_kw = {'wspace':0}
    )
    
    for i in range(args.clusters):

        samples = []
        target = targets[i]

        data = dataset[dataset['label'] == target]
        data.sort_values(args.weight_col, ascending=True)

        if args.worst:
            samples = data[-args.samples:]
        else:
            samples = data[:args.samples]

        for j, sample in enumerate(samples['file']):
            axs[i,j].imshow(
                load_img(os.path.join(args.data, sample), target_size=(224,224))
            )

            axs[i,j].set_xticklabels([])
            axs[i,j].set_xticks([])
            axs[i,j].set_yticklabels([])
            axs[i,j].set_yticks([])

    fig.tight_layout()
    fig.savefig(args.output_dir + '/sorted_' + str(args.worst) + "".join(random.sample(list(string.hexdigits), 8)) + '.png')        


def get_empty_image(shape=(224,224,3)):
    return np.zeros(shape)


if __name__ == "__main__":
    main(get_args())
    
