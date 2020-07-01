""" 

Forked from cluster.py to autoencoder.py.

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
from project_core.models import model_factory, LinearModel
from project_core.utils import load_image, build_files_dataframe, prune_file_list
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# For autoencoder
from tensorflow.keras import Model
from tensorflow.keras.layers  import (Input, Conv2D, Dense,
                                      Conv2DTranspose,
                                      Flatten, MaxPooling2D,
                                      BatchNormalization, Reshape,
                                      UpSampling2D)
from wandb.keras import WandbCallback

def dconv(inputs, filters, kernel_size,
          strides, padding, activation,
          use_batchnorm, use_pool):
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation
    )(inputs)
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation
    )(x)

    if use_pool:
        x = MaxPooling2D()(x)

    if use_batchnorm:
        x = BatchNormalization()(x)
        
    return x

def uconv(inputs, filters, kernel_size,
          activation, strides, padding,
          use_batchnorm):

    x = UpSampling2D()(inputs)
    x = Conv2D(filters=filters, kernel_size=kernel_size,
               strides=strides, activation=activation,
               padding=padding)(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size,
               strides=strides, activation=activation,
               padding=padding)(x)

    if use_batchnorm:
        x = BatchNormalization()(x)

    return x
    
def build_model(input_shape):

    inputs = Input(input_shape)

    # 224 ---> 112
    x = dconv(inputs, filters=32, kernel_size=(3,3),
              strides=1, padding='same', activation='relu',
              use_pool=True, use_batchnorm=False)
    # 112 ---> 56
    x = dconv(x, filters=64, kernel_size=(3,3),
              strides=1, padding='same', activation='relu',
              use_pool=True, use_batchnorm=False)
    # 56 ---> 28
    x = dconv(x, filters=128, kernel_size=(3,3),
              strides=1, padding='same', activation='relu',
              use_pool=True, use_batchnorm=False)
    # 28 ---> 14
    x = dconv(x, filters=128, kernel_size=(3,3),
              strides=1, padding='same', activation='relu',
              use_pool=True, use_batchnorm=False)

    # Filter reduction
    x = Conv2D(filters=16, kernel_size=(3,3),
               strides=1, padding='same', activation='relu')(x)
    
    # Flatten the space
    latent_space = Flatten()(x)

    # 14 ---> 28
    x = uconv(x, filters=128, kernel_size=(3,3),
              strides=1, padding='same', activation='relu',
              use_batchnorm=False)

    # 28 ---> 56
    x = uconv(x, filters=128, kernel_size=(3,3),
              strides=1, padding='same', activation='relu',
              use_batchnorm=False)

    # 56 ---> 112
    x = uconv(x, filters=64, kernel_size=(3,3),
              strides=1, padding='same', activation='relu',
              use_batchnorm=False)

    # 224 ---> 224
    x = uconv(x, filters=32, kernel_size=(3,3),
              strides=1, padding='same', activation='relu',
              use_batchnorm=False)

    x = Conv2D(filters=3, kernel_size=(3,3), strides=1,
               padding='same', activation='tanh')(x)
    
    model = Model(inputs, x)
    encoder = Model(inputs, latent_space)
    return model, encoder
    
    
def main(args):
 
    print('[INFO] Starting clustering...')

    # Setup weights and biases
    setup_wandb(args)
    
    # Setup the pre-trained backbone for our model.  This is
    # done first to get the preprocessing function for the net.
    #encoder, preprocess = model_factory(args.backbone, pooling=args.pooling)
    model, encoder = build_model((224,224,3))
    print(model.summary())
    print(encoder.summary())

    optimizer = Adam(
        learning_rate=args.learning_rate,
        beta_1=args.beta1,
        beta_2=args.beta2
    )
    model.compile(optimizer=optimizer, loss='mse')
    
    # Load the images into memory.  Right now
    # I am not supporting loading from disk.
    train, dev, test = load_dataframes(args.base_dir, args.min_samples)
    
    # Use an image data generator to save memory.
    augs = dict(
        horizontal_flip=True,
        zoom_range=args.zoom,
        width_shift_range=args.width_shift,
        height_shift_range=args.height_shift,
        preprocessing_function=preprocess_input,
        rotation_range=args.rotation,        
    )
    
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

    # Setup a generator for dev
    no_augs_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    dev_flow = no_augs_gen.flow_from_dataframe(
        dataframe=dev,
        directory=os.path.join(args.base_dir, 'dev'),
        batch_size=args.batch_size,
        target_size=(224,224),
        shuffle=False,
        x_col='file',
        class_mode=None
    )

    test_flow = no_augs_gen.flow_from_dataframe(
        dataframe=test,
        directory=os.path.join(args.base_dir, 'test'),
        batch_size=args.batch_size,
        target_size=(224,224),
        shuffle=False,
        x_col='file',
        class_mode=None
    )
    
    print('[INFO] Fitting autoencoder...')
    for layer in model.layers:
        layer.trainable = True

    # Fit the sucker
    batches = int(np.ceil(len(train) / args.batch_size))
    dev_batches = int(np.ceil(len(dev) / args.batch_size))
    for epoch in range(args.epochs):

        # Train
        for batch in range(batches):
            x_batch = next(train_flow)
            loss = model.train_on_batch(x_batch, x_batch)
            wandb.log({'loss':loss})

        for batch in range(dev_batches):
            x_batch = next(dev_flow)
            dev_loss = model.evaluate(x_batch, x_batch)
            wandb.log({'dev_loss':dev_loss})
            
        
    print('[INFO] Running linear evaluation...')
    for layer in encoder.layers:
        encoder.trainable = False
        
    label_encoder = LabelEncoder()
    train['encoded_label'] = label_encoder.fit_transform(train['label'])
    dev['encoded_label'] = label_encoder.transform(dev['label'])
    test['encoded_label'] = label_encoder.transform(test['label'])

    train_flow = gen.flow_from_dataframe(
        dataframe=train,
        directory=os.path.join(args.base_dir, 'train'),
        batch_size=args.batch_size,
        target_size=(224,224),
        shuffle=True,
        x_col='file',
        y_col='label',
        class_mode='categorical'
    )


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
    
    #linear_eval(
    #    encoder=encoder,
    #    train_gen=train_flow,
    #    dev_gen=dev_flow,
    #    train_labels=train['encoded_label'],
    #    dev_labels=dev['encoded_label'],
    #    metric=balanced_accuracy_score,
    #    log_training=True
    #)
        
    print('[INFO] Finished!')


    
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_dir', type=str, default='/home/ubuntu/data')
    ap.add_argument('--min_samples', type=int, default=320)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--learning_rate', type=float, default=0.001)
    ap.add_argument('--beta1', type=float, default=0.9)
    ap.add_argument('--beta2', type=float, default=0.99)
    ap.add_argument('--height_shift', type=float, default=0.1)
    ap.add_argument('--width_shift', type=float, default=0.1)
    ap.add_argument('--rotation', type=int, default=0)
    ap.add_argument('--zoom', type=float, default=0.0)
    ap.add_argument('--epochs', type=int, default=5)
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
        zoom = args.zoom,
        rotation = args.rotation,
        height_shift = args.height_shift,
        width_shift = args.width_shift
    )

    wandb.init(
        project='33ku',
        notes='autoencoder.py',
        tags=['autoencoder',],
        config=config
    )


def linear_eval(encoder, train_gen, dev_gen,
                train_labels, dev_labels,
                metric=None, log_training=False,
                epochs=40, steps_per_epoch=32):
    """ 

    Create a linear model and evaluate the representation created by the
    encoder using the metric provided.

    """

    # Get the number of classes in our dataset.
    n_classes = len(np.unique(train_labels))

    # Setup a linear model and train it using the
    # training generator.
    model = LinearModel(encoder=encoder, n_classes=n_classes)
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy')
    history = model.fit_generator(train_gen, epochs=epochs, steps_per_epoch=steps_per_epoch)

    # I don't know maybe there is a nice way to infer
    # this from the gen but this is my method.  I use
    # train instead of dev because I don't want to peel
    # off the first batch.
    x_fake, y_fake = next(train_gen)
    batch_size = len(y_fake)
    batches = int(np.ceil(len(dev_labels) / batch_size))
 
    # Get predictions from our linear model.
    preds = []
    for _ in range(batches):
        batch = next(dev_gen)
        preds.extend(
            np.argmax(model.predict(batch), axis=1)
        )

    if metric:
        val = metric(dev_labels, preds)
    else:
        val = np.inf

    if log_training:
        wandb.log({
            'linear_metric':val,
            'linear_n_classes':n_classes,
            'linear_epochs':epochs,
            'linear_steps_per_epoch':steps_per_epoch,
            'linear_batch_size':batch_size
        })

        for v in history.history['loss']:
            wandb.log({'linear_loss':v})


def standardize_over_batches(model, flow, batches):
    """ Accumulate over batches and return for normalization. """

    dim = model.output.shape[1]
    mu, sigma = np.zeros(shape=(dim,)), np.zeros(shape=(dim,))
    
    for batch in range(batches):
        x_batch = next(flow)
        preds = model.predict(x_batch)
        mu += np.sum(preds, axis=0)
        sigma += np.sum(preds**2, axis=0)
        
    mu /= batches
    sigma = (sigma - mu**2) / (batches - 1)
    return mu, sigma


if __name__ == "__main__":
    main(get_args())


