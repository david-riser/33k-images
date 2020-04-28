import argparse
import numpy as np
import os
import pandas as pd
import tqdm

from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image

from sklearn.cluster import KMeans

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, type=str)
    ap.add_argument('--images', required=True, type=str)
    return ap.parse_args()

def predict_image(model, image_path):
    x = image.load_img(image_path, target_size=(224,224))
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)

if __name__ == "__main__":

    args = get_args()
    image_df = pd.read_csv(args.images)
    n_classes = image_df['label'].nunique()
    print("We have {} classes.".format(n_classes))
    
    n_features = 2048
    features = np.zeros((len(image_df), n_features))
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    for i in tqdm.tqdm(range(len(image_df))):
        test_image = args.data_dir + '/' + image_df['file'].values[i]
        features[i,:] = predict_image(model, test_image)

    
    kmeans = KMeans(n_clusters=n_classes)
    clusters = kmeans.fit_predict(features)
    pd.DataFrame({'label':image_df['label'], 'cluster':clusters}).to_csv('clusters.csv', index=False)
    
