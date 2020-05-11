import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(PROJECT_DIR)

from project_core.utils import (get_image_loss, list_greyscale_images,
                                predict_images)
from sklearn.metrics import (adjusted_rand_score, confusion_matrix,
                             classification_report)
from tensorflow.keras.applications import resnet50
from tensorflow.keras.models import load_model


def get_args():

    def add_required_str_arg(parser, name):
        parser.add_argument(
            '--{}'.format(name),
            required=True,
            type=str)        

    parser = argparse.ArgumentParser()
    add_required_str_arg(parser, 'model')
    add_required_str_arg(parser, 'data')
    add_required_str_arg(parser, 'images')
    add_required_str_arg(parser, 'experiment')

    return parser.parse_args()

def plot_confusion_matrix(labels, preds, name):

    if len(labels) < 20:
        figsize = (16,12)
    else:
        figsize = (32,24)

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.savefig(name, bbox_inches='tight', dpi=100)

def plot_pr(labels, preds, name):

    cm = confusion_matrix(labels, preds)
    rows, _ = cm.shape

    classes = np.arange(rows)
    recall = []
    precision = []
    support = []
    for i in range(rows):
        true_positives = cm[i,i]
        total_true = np.sum(cm[i,:])
        total_pred = np.sum(cm[:,i])
        recall.append(float(true_positives / total_true))
        precision.append(float(true_positives / total_pred))
        support.append(total_true)
        

    plt.figure(figsize=(8,6))
    plt.scatter(support, recall, edgecolor='k',
                color='red', label='Recall')
    plt.scatter(support, precision, edgecolor='k',
                color='blue', label='Precision')
    plt.grid(alpha=0.2)
    plt.xlabel('Support')
    plt.ylabel('Value')
    plt.legend(frameon=False)
    plt.savefig(name, bbox_inches='tight', dpi=100)
    plt.close()

def plot_greyscale_loss(images, name):

    plt.figure(figsize=(8,6))
    plt.hist(
        images[images['greyscale'] == 0]['loss'],
        bins=np.linspace(0,17,60),
        edgecolor='k',
        color='red',
        alpha=0.65,
        label='Color',
        normed=True
    )
    plt.hist(
        images[images['greyscale'] == 1]['loss'],
        bins=np.linspace(0,17,60),
        edgecolor='k',
        color='blue',
        alpha=0.65,
        label='Greyscale',
        normed=True
    )
    plt.grid(alpha=0.2)
    plt.xlabel('Loss (Categorical Cross Entropy)')
    plt.legend(frameon=False)
    plt.savefig(name, bbox_inches='tight', dpi=100)
    
def main(args):
    """ 
    Load images and model from main.py.  Use those
    to make predictions for all images and then 
    evaluate those predictions using several metrics.

    Finally, save the output to the figures folder.

    """

    try:
        model = load_model(args.model)
        images = pd.read_csv(args.images)
    except Exception as e:
        print('Trouble loading something: {}'.format(e))
        exit()

    # Add the root directory to the image path
    images['path'] = images['file'].apply(lambda x: '{}/{}'.format(args.data, x))

    categorical_preds = predict_images(model, images['path'].values,
                                       target_size=(224,224),
                                       preprocess_input=resnet50.preprocess_input)
    preds = np.argmax(categorical_preds, axis=1)

    print("[INFO] Getting image loss, this can take some time... ")
    losses = get_image_loss(categorical_preds, images['encoded_label'])

    if len(losses) == len(images):
        images['loss'] = losses
    else:
        print("[FATAL] The image loss calculation missed something, exiting.")
        exit()

    # Get the list of greyscale images for later calculations.
    images['greyscale'] = list_greyscale_images(images['path'].values)

    # Calculate some metrics
    ar_score = adjusted_rand_score(images['encoded_label'].values, preds)
    print('Adjusted Rand Score: {0:6.4f}'.format(ar_score))
    print('Classification  Report: \n{}'.format(
        classification_report(images['encoded_label'].values, preds))
    )

    # And the confusion matrix
    if not os.path.exists('figures'):
        os.mkdir('figures')
        
    plot_confusion_matrix(images['encoded_label'].values, preds,
                          'figures/{}_confusion_matrix.png'.format(args.experiment))
    plot_pr(images['encoded_label'].values, preds,
            'figures/{}_pr_scatter.png'.format(args.experiment))

    plot_greyscale_loss(
        images,
        'figures/{}_greyscale_loss.png'.format(args.experiment)
    )
    
    # Add some information to validation images
    # file and save it again.  It's safe to overwrite
    # the input.
    images[['file', 'label', 'encoded_label', 'greyscale', 'loss']].to_csv(args.images, index=False)
    print('[INFO] Greyscale information {}'.format(np.unique(images['greyscale'], return_counts=True)))
    
if __name__ == '__main__':
    main(get_args())
