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

if __name__ == "__main__":

    args = get_args()

    # Load images and remove the classes with
    # too few examples.
    train = build_files_dataframe(os.path.join(args.base_dir, 'train'))
    print(train.head())
    train = prune_file_list(data=train, label_col='label',
                            min_samples=args.min_samples)
    
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

    image_loader = partial(load_image, target_size=target_size)

    batch_size = 1024
    batches = int(np.ceil(len(train) / batch_size))
    for i in tqdm.tqdm(range(batches)):
        test_images = [os.path.join(args.base_dir, 'train') + '/' + img
                       for img in train['file'].values[i*batch_size : (i+1)*batch_size]]
        features[i*batch_size : (i+1)*batch_size,:] = predict_images(
            model, test_images, preprocess_input, args.cores, image_loader
        )

    # Cluster the results in feature space.  This takes quite a lot
    # of memory (depending on the number of input images).
    kmeans = KMeans(n_clusters=n_classes)
    clusters = kmeans.fit_predict(features)
    pd.DataFrame(
        {'label':train['label'], 'cluster':clusters, 'file':train['file']}
    ).to_csv(args.output_dir + '/{}_{}_clusters.csv'.format(
        args.backbone, args.pooling), index=False)

    with open(args.output_dir + '/{}_{}_centroids.pkl'.format(args.backbone, args.pooling), 'wb') as out:
        pickle.dump({'centroids':kmeans.cluster_centers_, 'labels':kmeans.labels_, 'inertia':kmeans.inertia_}, out)

    if args.save_features:
        with open(args.output_dir + '/{}_{}_features.pkl'.format(args.backbone, args.pooling), 'wb') as out:
            pickle.dump({'features':features}, out)