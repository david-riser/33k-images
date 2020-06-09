""" 

Forked from cluster_mem.py to cluster_pca.py 
to load dev data in memory and try other 
methods (DBSCAN).  This version also applies 
PCA.

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
from project_core.metrics import hungarian_accuracy, hungarian_balanced_accuracy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
    features = np.zeros(
        shape=(len(train) + len(dev), encoder.output.shape[1])
    )
    for i, imagefile in enumerate(train['file']):
        img = load_image(
            image_path=os.path.join(args.base_dir, 'train/' + imagefile),
            preprocess_input=preprocess
        )
        features[i,:] = encoder.predict(img)

    for i, imagefile in enumerate(dev['file']):
        img = load_image(
            image_path=os.path.join(args.base_dir, 'dev/' + imagefile),
            preprocess_input=preprocess
        )
        features[i + len(train),:] = encoder.predict(img)

    # Scale before doing PCA, which
    # expects to have standardized
    # features.
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    if args.pca_components > 0:
        pca = PCA(
            n_components=args.pca_components
        )
        features = pca.fit_transform(features)
        
    # Setup clustering
    kmeans = KMeans(
        n_clusters=args.clusters        
    )
        
    print('[INFO] Running clustering...')
    kmeans.fit(features)
    
    # Save the dataframe of validation predictions and labels
    df = pd.DataFrame({
        'file':dev['file'],
        'label':dev['label'],
        'pred':kmeans.labels_[len(train):]
    })
    df = add_soft_cluster_probs(df, features[len(train):], kmeans.cluster_centers_)
    df = reindex_df_with_cluster_var(df, features[len(train):], kmeans.cluster_centers_)
    df.to_csv('dev_kmeans_pca_ms{}_{}.csv'.format(args.min_samples, wandb.run.id),
              index=False)

    # A quick performance estimate.
    ar_score = adjusted_rand_score(dev['label'], kmeans.labels_[len(train):])
    wandb.log({'ari':ar_score})
    wandb.log({'nmi':normalized_mutual_info_score(dev['label'], kmeans.labels_[len(train):])})

    wandb.log({'dev_clusters':
               len(np.unique(kmeans.labels_[len(train):]))})
    
    if args.pca_components > 0:
        wandb.log({'explained_variance':np.sum(pca.explained_variance_ratio_)})
    else:
        wandb.log({'explained_variance':0.00})

    # if the number of clusters is the same as
    # the true labels, we can do hungarian
    if args.clusters == dev['label'].nunique():
        hba = hungarian_balanced_accuracy(
            LabelEncoder().fit_transform(df['label']), df['pred']
        )
        wandb.log({'balanced_accuracy':hba})

        ha = hungarian_accuracy(
            LabelEncoder().fit_transform(df['label']), df['pred']
        )
        wandb.log({'accuracy':ha})

        
        
    print('[INFO] Finished!')
    
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_dir', type=str, default='/home/ubuntu/data')
    ap.add_argument('--min_samples', type=int, default=320)
    ap.add_argument('--backbone', type=str, default='Xception')
    ap.add_argument('--pooling', type=str, default='avg')
    ap.add_argument('--pca_components', type=int, default=64)
    ap.add_argument('--clusters', type=int, default=12)
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
        pca_components = args.pca_components,
        clusters = args.clusters
    )

    wandb.init(
        project='33ku',
        notes='cluster_pca.py',
        tags=['baseline', 'dbscan', 'pca'],
        config=config
    )

    
def reindex_df_with_cluster_var(df, features, centroids):
    """ 
    Re-index the dataframe by using the cluster variance. 
    """

    n_clusters, z_dim = centroids.shape

    cluster_var = np.zeros(n_clusters)
    for i in range(n_clusters):
        indices = np.where(df['pred'] == i)[0]
        cluster_var[i] = np.sum(
            np.var(
                features[indices, :], axis = 0
            )
        )

    re_index = np.argsort(cluster_var)
    mapping = { i:r for i,r in enumerate(re_index) }

    df['pred'] = df['pred'].apply(lambda x: mapping[x])
    return df
    

def soft_clustering_weights(data, cluster_centers):

    samples, features = data.shape
    centroids, _ = cluster_centers.shape
    weights = np.zeros(data.shape)

    for i in range(samples):
        for j in range(centroids):

            distances = np.zeros(centroids)
            for k in range(centroids):
                distances[k] = np.linalg.norm(data[i,:] - cluster_centers[k,:])

            # I eliminated the m factor and just used m = 2.
            weights[i,j] = (np.sum(distances) / distances[j])**2
            
            
    return weights
            
    
def add_soft_cluster_probs(df, features, centroids):
    """ 
    Add soft k-means cluster probs to dataframe.
    """
    weights = soft_clustering_weights(features, centroids)
    df['weight'] = np.max(weights, axis=1) / np.sum(weights, axis=1)
    return df

    
if __name__ == "__main__":
    main(get_args())


