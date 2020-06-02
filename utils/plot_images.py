""" 

June 2, 2020

Plot images from the training data to illustrate 
challenges and examples.  Landscape format for 
presentation slides.

"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import string

from tensorflow.keras.preprocessing.image import load_img


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/home/ubuntu/data/train')
    parser.add_argument('--clusters', type=int, default=5)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='/home/ubuntu')
    return parser.parse_args()


def main(args):

    # Get a dictionary that contains the mapping of
    # folders and their associated files.
    dataset = build_dataset_dict(args.data)

    # Prune out classes with samples < args.samples
    prunes = []
    for key, value in dataset.items():
        if len(value) < args.samples:
            prunes.append(key)

    for prune in prunes:
        dataset.pop(prune)
    
    # Choose randomly from the classes we have to
    # display some images.
    targets = random.sample(
        dataset.keys(), args.clusters
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

        samples = random.sample(
            dataset[target], args.samples
        )

        for j, sample in enumerate(samples):
            axs[i,j].imshow(
                load_img(sample, target_size=(224,224))
            )

            axs[i,j].set_xticklabels([])
            axs[i,j].set_xticks([])
            axs[i,j].set_yticklabels([])
            axs[i,j].set_yticks([])

    fig.tight_layout()
    fig.savefig(args.output_dir + '/samples_' + "".join(random.sample(list(string.hexdigits), 8)) + '.png')

def build_dataset_dict(data_folder):
    """ Search and build a dictionary that 
    contains folder names as keys and image
    paths as values. """

    dataset = {}
    for folder, _, files in os.walk(data_folder):
        files = [os.path.join(folder, f) for f in files]
        dataset[folder] = files

    return dataset
        


def get_empty_image(shape=(224,224,3)):
    return np.zeros(shape)


if __name__ == "__main__":
    main(get_args())
    
