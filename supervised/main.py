import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import (ModelCheckpoint,
                                        EarlyStopping,
                                        ReduceLROnPlateau)
from tensorflow.keras.metrics import TopKCategoricalAccuracy, Precision, Recall
from tensorflow.keras.models import load_model

from network import build_model

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--batches_per_epoch', type=int, default=50)
    ap.add_argument('--max_epochs', type=int, default=5)
    ap.add_argument('--images', type=str, required=True)
    ap.add_argument('--base_dir', type=str, required=True)
    return ap.parse_args()

def preprocess(x):
    print('[DEBUG] preprocess start {}'.format(x.shape))
    x = np.expand_dims(x, axis=0)
    print('[DEBUG] preprocess start (expanded) {}'.format(x.shape))
    return preprocess_input(x)

def plot_loss(history, name):
    """ Plot training and validation loss. """
    if not os.path.exists('figures'):
        os.mkdir('figures')
        
    plt.figure(figsize=(8,6))
    plt.plot(history.history['loss'], marker='o', label='Train', color='blue')
    plt.plot(history.history['val_loss'], marker='o', label='Valid', color='red')
    plt.grid(alpha=0.2)
    plt.legend(frameon=False, loc='upper left')
    plt.savefig('figures/{}_loss.png'.format(name), bbox_inches='tight')

def plot_metrics(history, name):
    """ Plot other metrics. """
    if not os.path.exists('figures'):
        os.mkdir('figures')

    plt.figure(figsize=(8,6))

    for metric in history.history.keys():
        if 'loss' not in metric:
            plt.plot(history.history[metric], marker='o', label=metric)

    plt.grid(alpha=0.2)
    plt.legend(frameon=False, loc='upper left')
    plt.savefig('figures/{}_metrics.png'.format(name), bbox_inches='tight')

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
    print('[DEBUG] Input shape: {}, Output shape: {}'.format(
        input_shape, output_shape))
    
    # Build generators
    split = int(0.9 * len(images))

    # Save validation folds
    images.to_csv('validation_images.csv', index=False)

    augmentations = dict(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocess_input
    )
    train_gen = ImageDataGenerator(**augmentations)
    valid_gen = ImageDataGenerator(**augmentations)

    train_flow = train_gen.flow_from_dataframe(
        dataframe=images[:split],
        directory=args.base_dir,
        batch_size=params['batch_size'],
        target_size=input_shape[:2],
        class_mode='categorical',
        shuffle=True,
        x_col='file',
        y_col='label'
    )
    valid_flow = valid_gen.flow_from_dataframe(
        dataframe=images[split:],
        directory=args.base_dir,
        batch_size=params['batch_size'],
        target_size=input_shape[:2],
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

    model_checkpoint = ModelCheckpoint(filepath='weights.hdf5',
                                       monitor='val_loss')
    early_stopping = EarlyStopping(monitor='val_loss', patience=12)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6)
    callbacks = [model_checkpoint, early_stopping, reduce_lr]

    top3_metric = TopKCategoricalAccuracy(k=3, name='top_3_cat_acc')
    precision_metric = Precision()
    recall_metric = Recall()
    metrics = ['accuracy', top3_metric, precision_metric, recall_metric]

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=metrics)
    history = model.fit_generator(train_flow, steps_per_epoch=50, epochs=args.max_epochs,
                                  validation_data=valid_flow, workers=4, callbacks=callbacks)


    # Train the entire network
    model = load_model('weights.hdf5')

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=metrics
    )

    model_checkpoint = ModelCheckpoint(filepath='weights_stage2.hdf5', monitor='val_loss')
    callbacks = [model_checkpoint, early_stopping, reduce_lr]
    history_stage2 = model.fit_generator(train_flow, steps_per_epoch=50, epochs=args.max_epochs,
                                         validation_data=valid_flow, workers=4, callbacks=callbacks)
    
    # Plot metrics
    plot_loss(history=history, name='stage1')
    plot_loss(history=history, name='stage2')
    plot_metrics(history=history, name='stage1')
    plot_metrics(history=history, name='stage2')
