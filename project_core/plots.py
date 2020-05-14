import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages
from utils import load_image

def main(args):
    """

    A test function only executed
    when running this script directly to
    test the plotting feature.

    """

    print('[INFO] Starting test...')
    print(f'[INFO] Loading data from {args.data} as specified in the file {args.list}.')

    try:
        dataframe = pd.read_csv(args.dataframe)
    except:
        print(f'[FATAL] Dataframe not loaded from {args.dataframe}.')
        exit()

    images = load_images(images=dataframe[args.image_col], data=args.data)
    plot_images(
        image=images,
        labels=dataframe[args.label_col],
        preds=dataframe[args.pred_col],
        nrows=4,
        ncols=3,
        name='{}.pdf'.format(args.experiment)
    )

def plot_images(images, preds, labels, name,
                ncols=4, nrows=4):
    """

    Print a pdf file of images grouped by
    label assignments to a file, specified
    by the name variable.

    :param image_paths: A list of numpy arrays.
    :param preds: A list of integers that is a predicted
    label for each image.
    :param labels: A list of integers that assigns each
    image to a label.
    :param name: A string that specifies the name of the
    output pdf file to print these images into.
    :return: None

    """

    with PdfPages(name) as pdf:

        fig, axs = plt.subplots(
            figsize=(8, 11),
            nrows=nrows, ncols=ncols,
            share_x=True, share_y=True,
            dpi=50
        )
        fig.subplots_adjust(wspace=0, hspace=0)

        plot_order = np.argsort(preds)
        for i, index in enumerate(plot_order):
            pad = 1 + i % (ncols * nrows)
            new_page = (pad == 1)

            # It is time to print out the
            # previous page of the pdf.
            if new_page and i > 0:
                pdf.savefig(fig)
                plt.close()

            row = (pad - 1) // ncols
            col = (pad - 1) % nrows

            # Plot an image there and remove axis
            # ticks if they exist to unblock the
            # figures and make them nicely sit
            # next to each other.
            axs[row,col].imshow(images[index])
            axs[row,col].set_xticklabels([])
            axs[row,col].set_yticklabels([])

        # It is possible that the last
        # page has not been printed.
        if not new_page:
            pdf.savefig(fig)
            plt.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--label_col', default='encoded_label', type=str)
    parser.add_argument('--pred_col', default='pred', type=str)
    parser.add_argument('--image_col', default='file', type=str)
    parser.add_argument('--dataframe', required=True, type=str)
    parser.add_argument('--experiment', required=True, type=str)
    return parser.parse_args()

def load_images(images, data):
    """

    Load all of the specified images from the data folder.

    :param images: A list of strings that contains the
    image name and folder name but not the entire path.
    :param data: A base directory to load images from.
    :return: A list of numpy array images.

    """
    return [load_image(os.path.join(data,image)) \
            for image in images]

if __name__ == "__main__":
    main(get_args())
