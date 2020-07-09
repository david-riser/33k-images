""" 

dec.py

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

from project_core.metrics import hungarian_accuracy, hungarian_balanced_accuracy
from project_core.models import model_factory, LinearModel, PretrainedDeepClusteringModel
from project_core.train import train_clustering_model_generator
from project_core.utils import load_image, build_files_dataframe, prune_file_list, clustering_target_distribution
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# For autoencoder
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers  import (Input, Conv2D, Dense,
                                      Conv2DTranspose,
                                      Flatten, MaxPooling2D,
                                      BatchNormalization, Reshape,
                                      UpSampling2D, Add, Activation)
  
def normalize(x):
    return x / 255.


def main(args):
 
    print('[INFO] Starting clustering...')

    # Setup weights and biases
    setup_wandb(args)
    
    # Setup the pre-trained encoder from autoencoder.py.
    encoder = load_model(args.model)
    model = PretrainedDeepClusteringModel(backbone=encoder, n_clusters=args.n_clusters)
    optimizer = Adam(
        learning_rate=args.learning_rate,
        beta_1=args.beta1,
        beta_2=args.beta2
    )
    model.compile(optimizer=optimizer, loss='kld')
    
    # Load the images into memory.  Right now
    # I am not supporting loading from disk.
    train, dev, test = load_dataframes(args.base_dir, args.min_samples)
    
    # Use an image data generator to save memory.
    augs = dict(
        preprocessing_function=normalize,
    )
    
    gen = ImageDataGenerator(**augs)
    train_flow = gen.flow_from_dataframe(
        dataframe=train,
        directory=os.path.join(args.base_dir, 'train'),
        batch_size=args.batch_size,
        target_size=(args.pixels,args.pixels),
        shuffle=True,
        x_col='file',
        class_mode=None
    )

    # Setup a generator for dev
    dev_flow = gen.flow_from_dataframe(
        dataframe=dev,
        directory=os.path.join(args.base_dir, 'dev'),
        batch_size=args.batch_size,
        target_size=(args.pixels,args.pixels),
        shuffle=False,
        x_col='file',
        class_mode=None
    )

    test_flow = gen.flow_from_dataframe(
        dataframe=test,
        directory=os.path.join(args.base_dir, 'test'),
        batch_size=args.batch_size,
        target_size=(args.pixels,args.pixels),
        shuffle=False,
        x_col='file',
        class_mode=None
    )

    print('[INFO] Starting initialization of clusters')
    model.initialize_clusters_generator(
        train_flow,
        epochs=1,
        steps_per_epoch=int(np.ceil(len(train) / args.batch_size))
    )

    print('[INFO] Fitting autoencoder...')
    for layer in encoder.layers:
        layer.trainable = True

    # -----------------
    #    Train here
    # -----------------
    loss = np.inf
    for ite in range(int(args.total_batches)):

        batch = next(train_flow)
        while len(batch) != args.batch_size:
            batch = next(train_flow)
                
        q = model.predict(batch, verbose=0)
        p = clustering_target_distribution(q)

        sub_batches = int(np.ceil(args.batch_size / 32))
        for i in range(sub_batches):
            loss = model.train_on_batch(x=batch[i*32:(i+1)*32], y=p[i*32:(i+1)*32])
            wandb.log({'kld_loss':loss})

    
    # Fit the sucker
    batches = int(np.ceil(len(train) / args.batch_size))
    dev_batches = int(np.ceil(len(dev) / args.batch_size))
            
    # This scaler is used to normalize before
    # doing clustering.  The online run is done
    # on the training data to collect statistics.
    print('[INFO] Fitting the scaler.')
    scaler = StandardScaler()
    for batch in range(batches):
        x_batch = next(train_flow)
        scaler.partial_fit(encoder.predict(x_batch))
        
    label_encoder = LabelEncoder()
    train['encoded_label'] = label_encoder.fit_transform(train['label'])
    dev['encoded_label'] = label_encoder.transform(dev['label'])
    test['encoded_label'] = label_encoder.transform(test['label'])

    kmeans = MiniBatchKMeans(n_clusters=train['label'].nunique())
    batches = int(np.ceil(len(train) / args.batch_size))
    for i in range(batches):
        kmeans.partial_fit(encoder.predict(
            next(train_flow)))
        
    dev_clusters = []
    test_clusters = []
    batches = int(np.ceil(len(dev) / args.batch_size))
    for i in range(batches):
        dev_clusters.extend(
            kmeans.predict(encoder.predict(next(dev_flow)))
        )
        
    batches = int(np.ceil(len(test) / args.batch_size))
    for i in range(batches):
        test_clusters.extend(
            kmeans.predict(encoder.predict(next(test_flow)))
        )

    dev_clusters = np.array(dev_clusters)
    test_clusters = np.array(test_clusters)

    accuracy = hungarian_accuracy(dev['encoded_label'], dev_clusters)
    balanced_accuracy = hungarian_balanced_accuracy(dev['encoded_label'], dev_clusters)
    wandb.log(
        {"dev_accuracy":accuracy, "dev_balanced_accuracy":balanced_accuracy}
    )

    accuracy = hungarian_accuracy(test['encoded_label'], test_clusters)
    balanced_accuracy = hungarian_balanced_accuracy(test['encoded_label'], test_clusters)
    wandb.log(
        {"test_accuracy":accuracy, "test_balanced_accuracy":balanced_accuracy}
    )

    x_batch = next(dev_flow)

    encoder.save("encoder.dec.{}.hdf5".format(wandb.run.id))
    
    print('[INFO] Finished!')


    
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_dir', type=str, default='/home/ubuntu/data')
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--min_samples', type=int, default=320)
    ap.add_argument('--n_clusters', type=int, default=12)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--learning_rate', type=float, default=0.001)
    ap.add_argument('--beta1', type=float, default=0.9)
    ap.add_argument('--beta2', type=float, default=0.99)
    ap.add_argument('--total_batches', type=int, default=40)
    ap.add_argument('--pixels', type=int, default=256)
    return ap.parse_args()

def load_dataframes(data_dir, min_samples):

    train = build_files_dataframe(os.path.join(data_dir, 'train'))
    train = prune_file_list(train, 'label', min_samples)

    dev = build_files_dataframe(os.path.join(data_dir, 'dev'))
    dev_cols = list(dev.columns)
    classes = np.unique(train['label'])
    dev['keep'] = dev['label'].apply(lambda x: x in classes)
    dev = dev[dev['keep'] == True]

    test = build_files_dataframe(os.path.join(data_dir, 'test'))
    test_cols = list(test.columns)
    test['keep'] = test['label'].apply(lambda x: x in classes)
    test = test[test['keep'] == True]
    
    train = train.sample(frac=1).reset_index(drop=True)
    dev = dev.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)
    return train, dev, test


def setup_wandb(args):
    """ Setup weights and biases logging. """

    config = dict(
        batch_size = args.batch_size,
        min_samples = args.min_samples,
        learning_rate = args.learning_rate,
        beta1 = args.beta1,
        beta2 = args.beta2,
        encoder = args.model
    )

    wandb.init(
        project='33ku',
        notes='dec.py',
        tags=['dec','fine-tuning'],
        config=config
    )


if __name__ == "__main__":
    main(get_args())


