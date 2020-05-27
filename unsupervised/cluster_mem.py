""" 

Forked from cluster.py to cluster_mem.py 
to load dev data in memory and try other 
methods (DBSCAN).  Do I need to load the 
entire dataset into memory?  

"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import wandb

PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), '../'))
print(PROJECT_DIR)
sys.path.append(PROJECT_DIR)

from project_core.models import model_factory, LinearModel
from project_core.utils import load_image, build_files_dataframe, prune_file_list
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.metrics import (adjusted_rand_score, balanced_accuracy_score,
                             normalized_mutual_info_score)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main(args):
 
    print('[INFO] Starting clustering...')

    # Setup weights and biases
    setup_wandb(args)
    
    # Setup the pre-trained backbone for our model.  This is
    # done first to get the preprocessing function for the net.
    encoder, preprocess = model_factory(
        args.backbone, pooling=args.pooling
    )

    # Load the images into memory.  Right now
    # I am not supporting loading from disk.
    train, dev = load_dataframes(args.base_dir, args.min_samples)

    # Load the features for the dev set which we
    # are going to cluster up.
    features = np.zeros(shape=(len(dev),encoder.output.shape[1]))
    for i, imagefile in enumerate(dev['file']):
        img = load_image(
            image_path=os.path.join(args.base_dir, 'dev/' + imagefile),
            preprocess_input=preprocess
        )
        features[i,:] = encoder.predict(img)

    # Normalize the features if we are asked to do
    # that.
    if args.normalize == "standard":
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    elif args.normalize == "minmax":
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)

    # Setup DBSCAN
    dbscan = DBSCAN(
        eps=args.eps,
        min_samples=args.min_dbscan_samples,
        n_jobs=-1
    )
    
    print('[INFO] Running DBSCAN...')
    dbscan.fit(features)
    
    # Save the dataframe of validation predictions and labels
    pd.DataFrame({
        'file':dev['file'],
        'label':dev['label'],
        'pred':dbscan.labels_
    }).to_csv('dev_dbscan_ms{}_{}.csv'.format(args.min_samples, wandb.run.id),
              index=False)

    # A quick performance estimate.
    ar_score = adjusted_rand_score(dev['label'], dbscan.labels_)
    wandb.log({'ari':ar_score})
    wandb.log({'nmi':normalized_mutual_info_score(dev['label'], dbscan.labels_)})
    
    # Log the number of empty clusters
    # on the dev set.
    nunique = len(np.unique(dbscan.labels_))
    wandb.log({'dev_clusters':nunique})
    
    print('[INFO] Finished!')
    
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_dir', type=str, default='/home/ubuntu/data')
    ap.add_argument('--min_samples', type=int, default=320)
    ap.add_argument('--backbone', type=str, default='ResNet50')
    ap.add_argument('--pooling', type=str, default='avg')
    ap.add_argument('--min_dbscan_samples', type=int, default=10)
    ap.add_argument('--eps', type=float, default=0.01)
    ap.add_argument('--normalize', type=str, default=None)
    return ap.parse_args()

def load_dataframes(data_dir, min_samples):

    train = build_files_dataframe(os.path.join(data_dir, 'train'))
    train = prune_file_list(train, 'label', min_samples)

    dev = build_files_dataframe(os.path.join(data_dir, 'dev'))
    dev_cols = list(dev.columns)
    classes = np.unique(train['label'])
    dev['keep'] = dev['label'].apply(lambda x: x in classes)
    dev = dev[dev['keep'] == True]
    
    train = train.sample(frac=1).reset_index(drop=True)
    dev = dev.sample(frac=1).reset_index(drop=True)
    return train, dev


def setup_wandb(args):
    """ Setup weights and biases logging. """

    config = dict(
        architecture = ":".join([args.backbone, args.pooling]),
        min_samples = args.min_samples,
        min_dbscan_samples = args.min_dbscan_samples,
        normalize = args.normalize or 'None',
        eps = args.eps
    )

    wandb.init(
        project='33ku',
        notes='cluster_mem.py',
        tags=['baseline', 'dbscan'],
        config=config
    )

    

if __name__ == "__main__":
    main(get_args())


