import argparse
import numpy as np
import pandas as pd
import os

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

from network import build_model
from train import train


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--batches_per_epoch', type=int, default=50)
    ap.add_argument('--max_epochs', type=int, default=5)
    ap.add_argument('--images', type=str, required=True)
    ap.add_argument('--base_dir', type=str, required=True)
    return ap.parse_args()

def preprocess(x):
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

if __name__ == "__main__":

    args = get_args()

    params = {
        'batch_size':args.batch_size,
        'batches_per_epoch':args.batches_per_epoch,
        'max_epochs':args.max_epochs,
    }

    # Load and shuffle image path
    images = pd.read_csv(args.images)
    images = images.sample(frac=1).reset_index(drop=True)

    # Calculate shapes for training
    input_shape = (224,224,3)
    output_shape = images['label'].nunique()

    # Build generators
    split = int(0.8 * len(images))
    train_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=None
    )
    valid_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=None
    )
    train_flow = train_gen.flow_from_dataframe(
        dataframe=images[:split],
        directory=args.base_dir,
        batch_size=params['batch_size'],
        target_size=input_shape,
        class_mode='categorical',
        shuffle=True,
        x_col='file',
        y_col='label'
    )
    valid_flow = valid_gen.flow_from_dataframe(
        dataframe=images[split:],
        directory=args.base_dir,
        batch_size=params['batch_size'],
        target_size=input_shape,
        class_mode='categorical',
        shuffle=True,
        x_col='file',
        y_col='label'
    )
    
    # Build model and freeze most of it.
    model = build_model(input_shape, output_shape)
    for layer in model.layers[:-2]:
        layer.trainable = False

    for layer in model.layers[-2:]:
        layer.trainable = True

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit_generator(train_flow, steps_per_epoch=50, epochs=args.max_epochs,
                        validation_data=valid_flow)
