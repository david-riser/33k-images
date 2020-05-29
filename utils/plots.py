import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), '../'))
sys.path.append(PROJECT_DIR)

from matplotlib.backends.backend_pdf import PdfPages
from project_core.utils import load_image

def main(args):
    """

    A test function only executed
    when running this script directly to
    test the plotting feature.

    """

    print(f'[INFO] Loading data from {args.data} as specified in the file {args.dataframe}.')

    try:
        dataframe = pd.read_csv(args.dataframe)
    except:
        print(f'[FATAL] Dataframe not loaded from {args.dataframe}.')
        exit()

    images = load_images(images=dataframe[args.image_col], data=args.data)
    plot_images(
        images=images,
        labels=dataframe[args.label_col],
        preds=dataframe[args.pred_col],
        nrows=args.cols,
        ncols=args.rows,
        name=args.pdf,
        dpi=args.dpi
    )

def plot_images(images, preds, labels, name,
                ncols=4, nrows=4, dpi=40):
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

    total_pages = len(images) // (ncols * nrows)

    # Setup colors for true labels
    colors = { i:random_color() for i in np.unique(labels) }
    
    with PdfPages(name) as pdf:

        fig, axs = plt.subplots(
            figsize=(8, 11),
            nrows=nrows, ncols=ncols,
            sharex=True, sharey=True,
            dpi=dpi
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

                print('[INFO] Printing page {} of {}.'.format(
                    i // (ncols * nrows), total_pages
                ))

            row = (pad - 1) // ncols
            col = (pad - 1) % ncols
    
            # Plot an image there and remove axis
            # ticks if they exist to unblock the
            # figures and make them nicely sit
            # next to each other.
            x = color_pad(images[index], colors[labels[index]])
            axs[row,col].imshow(x)
            axs[row,col].set_xticklabels([])
            axs[row,col].set_yticklabels([])
            axs[row,col].set_title(preds[index])


        # Pad out the last page with empty
        # figures so we don't have repeats.
        for i in range(len(preds), (total_pages + 1) * nrows * ncols):
            pad = 1 + i % (ncols * nrows)
            row = (pad - 1) // ncols
            col = (pad - 1) % ncols
            
            x = np.zeros(images[0].shape)
            axs[row,col].imshow(x)
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
    parser.add_argument('--pdf', required=True, type=str)
    parser.add_argument('--dpi', default=40, type=int)
    parser.add_argument('--cols', default=6, type=int)
    parser.add_argument('--rows', default=5, type=int)
    return parser.parse_args()

def load_images(images, data):
    """

    Load all of the specified images from the data folder.

    :param images: A list of strings that contains the
    image name and folder name but not the entire path.
    :param data: A base directory to load images from.
    :return: A list of numpy array images.

    """
    return [load_image(os.path.join(data,image)).reshape(224,224,3) \
            for image in images]

def color_pad(image, color=(122,122,122), pixels=16):
    """ 
    Add a pixel width colored border to the image.

    There must be a better way to write this function. 

    """

    h, w, c = image.shape
    new_image = np.zeros(
        (h + 2 * pixels, w + 2 * pixels, c),
        dtype=np.uint8)

    #for i in range(h + 2*pixels):
    #    for j in range(w + 2*pixels):
    #        if (i < pixels) or (j < pixels):
    #            new_image[i,j,:] = color
    #        elif (i > h + pixels) or (j > w + pixels):
    #            new_image[i,j,:] = color

    new_image[:,:,:] = color
    new_image[pixels : h + pixels, pixels : w + pixels] = image
                
    return new_image

def random_color():
    return np.random.randint(0, 255, 3)

if __name__ == "__main__":
    main(get_args())
