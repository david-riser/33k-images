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
from project_core.train import train_clustering_model_generator, KMeansImageDataGeneratorWrapper
from sklearn.metrics import adjusted_rand_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main(args):
 
    print('[INFO] Starting deep clustering...')
    
    # Setup the pre-trained backbone for our model.  This is
    # done first to get the preprocessing function for the net.
    backbone, preprocess = model_factory(args.backbone, pooling=args.pooling)
    model = PretrainedDeepClusteringModel(n_clusters=args.clusters, backbone=backbone)
    opt = Adam(0.0001)
    model.compile(optimizer=opt, loss='kld')

    # Load the images into memory.  Right now
    # I am not supporting loading from disk.
    train, dev = load_dataframes(args.base_dir, args.min_samples)
    
    # Use an image data generator to save memory.
    augs = dict(preprocessing_function=preprocess)
    gen = ImageDataGenerator(**augs)
    train_flow = gen.flow_from_dataframe(
        dataframe=train,
        directory=os.path.join(args.base_dir, 'train'),
        batch_size=args.batch_size,
        target_size=(224,224),
        shuffle=True,
        x_col='file',
        class_mode=None
    )
    
    print('[INFO] Initializing clusters.')
    model.initialize_clusters_generator(train_flow, epochs=1,
                                        steps_per_epoch=int(np.ceil(len(train) / args.batch_size)))

    print('[INFO] Training...')
    kld = train_clustering_model_generator(
        model=model,
        gen=train_flow,
        max_iter=args.epochs,
        update_interval=args.update_interval,
        batch_size=args.batch_size,
        verbose=True
    )

    # Setup a generator for dev
    dev_flow = gen.flow_from_dataframe(
        dataframe=dev,
        directory=os.path.join(args.base_dir, 'dev'),
        batch_size=args.batch_size,
        target_size=(224,224),
        shuffle=False,
        x_col='file',
        class_mode=None
    )
    
    print('[INFO] Predicting...')
    pred = []
    batches = int(np.ceil(len(dev) / args.batch_size))
    for batch in range(batches):
        pred.extend(
            np.argmax(
                model.predict(next(dev_flow)),
                axis = 1)
        )
    
        
    print('[INFO] Saving...')
    # Save the clustering model weights.
    model.save_weights('weights_{}.hdf5'.format(args.experiment))
    
    # Save the dataframe of validation predictions and labels
    pd.DataFrame({
        'file':dev['file'],
        'label':dev['label'],
        'pred':pred
    }).to_csv('validation_ms{}_{}.csv'.format(args.min_samples, args.experiment),
              index=False)

    # A quick performance estimate.
    print("[INFO] Adjusted Rand Index {0:6.4f}".format(
        adjusted_rand_score(dev['label'], pred)
    ))
    
    plot_kld(kld, args.experiment)
    
    print('[INFO] Finished!')
    
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_dir', required=True, type=str)
    ap.add_argument('--experiment', required=True, type=str)
    ap.add_argument('--clusters', required=True, type=int)
    ap.add_argument('--min_samples', type=int, required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--backbone', type=str, default='ResNet50')
    ap.add_argument('--pooling', type=str, default='avg')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--update_interval', type=int, default=10)
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


