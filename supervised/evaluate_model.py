import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(PROJECT_DIR)

from project_core.utils import predict_images
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
    
if __name__ == '__main__':
    main(get_args())
