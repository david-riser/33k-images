import argparse
import numpy as np
import os
import pandas as pd
import tqdm
import pickle

from keras.applications import resnet50, inception_v3, nasnet, xception
from keras.preprocessing import image
from multiprocessing import Pool
from sklearn.cluster import KMeans

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, type=str)
    ap.add_argument('--images', required=True, type=str)
    ap.add_argument('--backbone', required=True, type=str)
    ap.add_argument('--pooling', required=True, type=str)
    ap.add_argument('--output_dir', required=True, type=str)
    ap.add_argument('--cores', required=True, type=int)
    return ap.parse_args()

def model_factory(model_name, pooling=None):

    if model_name == "test":
        print('Instantiating a test model.')
        exit()
    elif model_name == 'ResNet50':
        print('Instantiating a ResNet50')
        model = resnet50.ResNet50(weights='imagenet', include_top=False, pooling=pooling)
        preprocess_input = resnet50.preprocess_input
        return model, preprocess_input
    elif model_name == 'InceptionV3':
        print('Instantiating a InceptionV3')
        model = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling=pooling)
        preprocess_input = inception_v3.preprocess_input
        return model, preprocess_input
    elif model_name == 'NASNet':
        print('Instantiating a NASNet')
        model = nasnet.NASNetLarge(weights='imagenet', include_top=False, pooling=pooling)
        preprocess_input = nasnet.preprocess_input
        return model, preprocess_input
    elif model_name == 'Xception':
        print('Instantiating a Xception')
        model = xception.Xception(weights='imagenet', include_top=False, pooling=pooling)
        preprocess_input = nasnet.preprocess_input
        return model, preprocess_input
    else:
        print('Unknown model {}.  Exiting!'.format(model_name))

def load_image(image_path, target_size=(224,224)):
    x = image.load_img(image_path, target_size=target_size)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def predict_image(model, image_path, preprocess_input):
    x = load_image(image_path)
    return model.predict(x)

def predict_images(model, image_paths, preprocess_input, cores=1, target_size=(224,224)):
    pool = Pool(cores)
    x = np.array(pool.map(lambda f: load_image(f,target_size), image_paths))
    pool.close()
    pool.join()    
    x = x.reshape(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
    return model.predict(x)

if __name__ == "__main__":

    args = get_args()
    image_df = pd.read_csv(args.images)
    n_classes = image_df['label'].nunique()
    print("We have {} classes.".format(n_classes))

    # Setup output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    model, preprocess_input = model_factory(args.backbone, args.pooling)
    n_features = model.output.shape[1].value
    features = np.zeros((len(image_df), n_features))
    if args.backbone == "NASNet":
        target_size = (331,331)
    else:
        target_size = (224,224)
    
    #for i in tqdm.tqdm(range(len(image_df))):

    batch_size = 1024
    batches = int(np.ceil(len(image_df) / batch_size))
    for i in tqdm.tqdm(range(1)):
        test_images = [args.data_dir + '/' + img for img in image_df['file'].values[i*batch_size : (i+1)*batch_size]]
        features[i*batch_size : (i+1)*batch_size,:] = predict_images(model, test_images, preprocess_input, args.cores, target_size)
    
    kmeans = KMeans(n_clusters=n_classes)
    clusters = kmeans.fit_predict(features)
    pd.DataFrame(
        {'label':image_df['label'], 'cluster':clusters}
    ).to_csv(args.output_dir + '/{}_{}_clusters.csv'.format(
        args.backbone, args.pooling), index=False)

    with open(args.output_dir + '/{}_{}_centroids.pkl'.format(args.backbone, args.pooling), 'wb') as out:
        pickle.dump({'centroids':kmeans.cluster_centers_, 'labels':kmeans.labels_, 'inertia':kmeans.inertia_}, out)
