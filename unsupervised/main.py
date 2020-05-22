import argparse
import numpy as np
import os
import sys
import pandas as pd
import tqdm
import pickle

from functools import partial
from keras.preprocessing import image
from multiprocessing import Pool
from sklearn.cluster import KMeans
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Setup this project
PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(PROJECT_DIR)
from project_core.models import model_factory
from project_core.utils import (create_directory, prune_file_list,
                                build_files_dataframe)
from project_core.train import KMeansImageDataGeneratorWrapper


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_dir', required=True, type=str)
    ap.add_argument('--backbone', required=True, type=str)
    ap.add_argument('--pooling', required=True, type=str)
    ap.add_argument('--output_dir', required=True, type=str)
    ap.add_argument('--min_samples', required=True, type=int)
    ap.add_argument('--cores', required=True, type=int)
    ap.add_argument('--batch_size', default=32, type=int)
    ap.add_argument('--save_features', action='store_true')
    return ap.parse_args()

def load_image(image_path, target_size=(224,224)):
    x = image.load_img(image_path, target_size=target_size)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def predict_image(model, image_path, preprocess_input):
    x = load_image(image_path)
    return model.predict(x)

def predict_images(model, image_paths, preprocess_input, cores, image_loader):
    pool = Pool(cores)
    x = np.array(pool.map(image_loader, image_paths))
    pool.close()
    pool.join()    
    x = x.reshape(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
    return model.predict(x)

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

if __name__ == "__main__":

    args = get_args()

    # Load images and remove the classes with
    # too few examples.
    train, dev = load_dataframes(args.base_dir, args.min_samples)
    n_classes = train['label'].nunique()
    print("We have {} classes.".format(n_classes))
    
    # Setup output
    create_directory(args.output_dir, recursive=True)

    # Build the model and import the correct pre-processing
    # function.  Each model uses a different function.
    # Maybe they're the same under the hood because
    # they are all trained with imagenet (something to look
    # into).
    model, preprocess_input = model_factory(args.backbone, args.pooling)
    n_features = model.output.shape[1]
    features = np.zeros((len(train), n_features))

    if args.backbone == "NASNet":
        target_size = (331,331)
    else:
        target_size = (224,224)

    # Use an image data generator to save memory.
    augs = dict(preprocessing_function=preprocess_input)
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

    kmeans = KMeansImageDataGeneratorWrapper(
        keras_model=model, n_clusters=n_classes
    )
    kmeans.fit_generator(
        generator=train_flow,
        epochs=4,
        steps_per_epoch=100
    )

    # Output for dev set predictions
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
    batches = int(np.ceil(len(dev) / args.batch_size))
    print("[INFO] Using {} batches for dev set.".format(batches))    
    pred = kmeans.predict(dev_flow, steps=batches)
        
    pd.DataFrame(
        {'label':dev['label'], 'cluster':pred, 'file':dev['file']}
    ).to_csv(args.output_dir + '/{}_{}_clusters.csv'.format(
        args.backbone, args.pooling), index=False)

    #with open(args.output_dir + '/{}_{}_centroids.pkl'.format(args.backbone, args.pooling), 'wb') as out:
    #    pickle.dump({'centroids':kmeans.cluster_centers_, 'labels':kmeans.labels_, 'inertia':kmeans.inertia_}, out)

    #if args.save_features:
    #    with open(args.output_dir + '/{}_{}_features.pkl'.format(args.backbone, args.pooling), 'wb') as out:
    #        pickle.dump({'features':features}, out)
