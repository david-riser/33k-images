import argparse
import numpy as np
import pandas as pd
import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), '../'))
print(PROJECT_DIR)
sys.path.append(PROJECT_DIR)

from project_core.models import model_factory, PretrainedDeepClusteringModel
from project_core.utils import load_image
from project_core.train import train_clustering_model
from tensorflow.keras.optimizers import Adam

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
    images, labels, files = load_images_and_labels(args.data, args.images, preprocess)
    images = np.array(images)
    split = int(args.train_frac * len(images))

    print('[INFO] Initializing clusters.')
    print(model.clustering_layer.get_weights())
    model.initialize_clusters(images[:split])

    print('[INFO] Training...')
    train_clustering_model(
        model=model,
        X_train=images[:split],
        max_iter=1000,
        update_interval=50,
        batch_size=32,
        verbose=True
    )


    print('[INFO] Predicting...')
    pred = model.predict(images[split:])

    print('[INFO] Saving...')
    # Save the clustering model weights.
    model.save('weights_{}.hdf5'.format(args.experiment))
    
    # Save the dataframe of validation predictions and labels
    pd.DataFrame({
        'file':files[split:],
        'label':labels[split:],
        'pred':preds
    }).to_csv('validation_{}.csv'.format(args.experiment),
              index=False)

    print('[INFO] Finished!')
    
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, type=str)
    ap.add_argument('--images', required=True, type=str)
    ap.add_argument('--experiment', required=True, type=str)
    ap.add_argument('--clusters', required=True, type=int)
    ap.add_argument('--train_frac', default=0.8, type=float)
    return ap.parse_args()

def load_images_and_labels(data_dir, images_list, preprocess):
    data = pd.read_csv(images_list)
    images = [load_image(os.path.join(data_dir, img),
                         preprocess_input=preprocess).reshape(224, 224, 3)
              for img in data['file']]
    
    return images, data['label'], data['file']
    
if __name__ == "__main__":
    main(get_args())


