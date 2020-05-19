import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), '../'))
print(PROJECT_DIR)
sys.path.append(PROJECT_DIR)

from project_core.models import model_factory, PretrainedDeepClusteringModel
from project_core.utils import load_image, build_files_dataframe, prune_file_list
from project_core.train import train_clustering_model, KMeansImageDataGeneratorWrapper
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main(args):

    print('[INFO] Starting deep clustering...')
    print('[INFO] Loading images.')
    
    # Setup the pre-trained backbone for our model.  This is
    # done first to get the preprocessing function for the net.
    backbone, preprocess = model_factory('ResNet50', pooling='avg')
    model = PretrainedDeepClusteringModel(n_clusters=args.clusters, backbone=backbone)
    opt = Adam(0.0001)
    model.compile(optimizer=opt, loss='kld')

    # Load the images into memory.  Right now
    # I am not supporting loading from disk.
    (train_images, dev_images),  (train_labels, dev_labels), (train_files, dev_files) = load_images_and_labels(
        args.base_dir, args.min_samples, preprocess)
    train_images = np.array(train_images)
    dev_images = np.array(dev_images)

    print('[INFO] Initializing clusters.')
    print(model.clustering_layer.get_weights())
    model.initialize_clusters(train_images)

    print('[INFO] Training...')
    kld = train_clustering_model(
        model=model,
        X_train=train_images,
        max_iter=125,
        update_interval=10,
        batch_size=32,
        verbose=True
    )

    print('[INFO] Predicting...')
    pred = model.predict(dev_images)
    pred = np.argmax(pred, axis=1)
    
    print('[INFO] Saving...')
    # Save the clustering model weights.
    model.save_weights('weights_{}.hdf5'.format(args.experiment))
    
    # Save the dataframe of validation predictions and labels
    pd.DataFrame({
        'file':dev_files,
        'label':dev_labels,
        'pred':pred
    }).to_csv('validation_ms{}_{}.csv'.format(args.min_samples, args.experiment),
              index=False)

    plot_kld(kld, args.experiment)
    
    print('[INFO] Finished!')
    
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_dir', required=True, type=str)
    ap.add_argument('--experiment', required=True, type=str)
    ap.add_argument('--clusters', required=True, type=int)
    ap.add_argument('--min_samples', type=int, required=True)
    return ap.parse_args()

def load_images_and_labels(data_dir, min_samples, preprocess):

    train = build_files_dataframe(os.path.join(data_dir, 'train'))
    train = prune_file_list(train, 'label', min_samples)

    dev = build_files_dataframe(os.path.join(data_dir, 'dev'))
    dev_cols = list(dev.columns)
    classes = np.unique(train['label'])
    dev['keep'] = dev['label'].apply(lambda x: x in classes)
    dev = dev[dev['keep'] == True]
    
    train = train.sample(frac=1).reset_index(drop=True)
    dev = dev.sample(frac=1).reset_index(drop=True)

    train_images = [load_image(os.path.join(data_dir, 'train', img),
                         preprocess_input=preprocess).reshape(224, 224, 3)
              for img in train['file']]
    dev_images = [load_image(os.path.join(data_dir, 'dev', img),
                         preprocess_input=preprocess).reshape(224, 224, 3)
              for img in dev['file']]
    
    return (train_images, dev_images), (train['label'], dev['label']), (train['file'], dev['file'])

def plot_kld(kld, expname):

    if not os.path.exists('figures'):
        os.mkdir('figures')
    
    plt.figure(figsize=(8,6))
    plt.plot(np.arange(len(kld)), kld, marker='o', color='red')
    plt.xlabel('Update Step')
    plt.ylabel('KL Divergence')
    plt.grid(alpha=0.2)
    plt.savefig('figures/{}_kld.png'.format(expname), bbox_inches='tight', dpi=100)

if __name__ == "__main__":
    main(get_args())


