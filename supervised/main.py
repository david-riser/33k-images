import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
import wandb


PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), '../'))
sys.path.append(PROJECT_DIR)

from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import (ModelCheckpoint,
                                        EarlyStopping,
                                        ReduceLROnPlateau)
from tensorflow.keras.metrics import TopKCategoricalAccuracy, Precision, Recall
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.models import load_model

# This project
from network import build_model
from project_core.utils import build_files_dataframe, prune_file_list

    
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_dir', type=str, default="/home/ubuntu/data")
    ap.add_argument('--experiment', type=str, default="none")
    ap.add_argument('--min_samples', type=int, default=320)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--batches_per_epoch', type=int, default=50)
    ap.add_argument('--max_epochs', type=int, default=100)
    ap.add_argument('--backbone', type=str, default='ResNet50')
    ap.add_argument('--pooling', type=str, default='avg')
    return ap.parse_args()


def setup_wandb(args):
    config = dict(
        architecture = ":".join([args.backbone, args.pooling]),
        min_samples = args.min_samples,
        batch_size = args.batch_size
    )
    wandb.init(
        project='33ks',
        notes='Supervised',
        tags=['Supervised'],
        config=config
    )

def plot_loss(history, name):
    """ Plot training and validation loss. """
    if not os.path.exists('figures'):
        os.mkdir('figures')
        
    plt.figure(figsize=(8,6))
    plt.plot(history.history['loss'], marker='o', label='Train', color='blue')
    plt.plot(history.history['val_loss'], marker='o', label='Valid', color='red')
    plt.grid(alpha=0.2)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(frameon=False, loc='upper left')
    plt.savefig('figures/{}_loss.png'.format(name), bbox_inches='tight')

def plot_metrics(history, name):
    """ Plot other metrics. """
    if not os.path.exists('figures'):
        os.mkdir('figures')

    for metric in list_metrics(history):
        plt.figure(figsize=(8,6))
        plt.plot(history.history[metric], marker='o', label=metric, color='blue')
        plt.plot(history.history['val_' + metric], marker='o',
                 label='Val. {}'.format(metric), color='red')
        plt.grid(alpha=0.2)
        plt.xlabel('Epoch')
        plt.legend(frameon=False, loc='upper left')
        plt.savefig('figures/{}_{}.png'.format(name, metric), bbox_inches='tight')
        plt.close()

def list_metrics(history):
    metrics = []

    prohibited = ['loss', 'lr', 'val']
    for key in history.history.keys():
        keep = not any([p in key for p in prohibited])
        if keep:
            metrics.append(key)

    return metrics

def load_data(base_dir, min_samples):
    
    train = build_files_dataframe(os.path.join(args.base_dir, 'train'))
    print(train.head())
    train = prune_file_list(train, label_col='label',
                            min_samples=args.min_samples)
    train = train.sample(frac=1).reset_index(drop=True)

    # Filter out the classes that we do not wait
    # in our dev set.
    classes = np.unique(train['label'])

    # Load dev set and return it
    dev = build_files_dataframe(os.path.join(args.base_dir, 'dev'))
    print(dev.head())
    dev = dev.sample(frac=1).reset_index(drop=True)
    return_cols = list(dev.columns)
    dev['keep'] = dev['label'].apply(lambda x: x in classes)
    dev = dev[dev['keep'] == True]

    # Load test set and return it
    test = build_files_dataframe(os.path.join(args.base_dir, 'test'))
    print(test.head())
    test = test.sample(frac=1).reset_index(drop=True)
    return_cols = list(test.columns)
    test['keep'] = test['label'].apply(lambda x: x in classes)
    test = test[test['keep'] == True]

    return train, dev[return_cols], test[return_cols]

if __name__ == "__main__":

    args = get_args()
    setup_wandb(args)
    
    params = {
        'batch_size':args.batch_size,
        'batches_per_epoch':args.batches_per_epoch,
        'max_epochs':args.max_epochs,
    }

    # Load and shuffle image path.  Also
    # encode the labels as integers for
    # ease of metric calculation later.
    train, dev, test = load_data(args.base_dir, args.min_samples)
    
    # Calculate shapes for training
    input_shape = (224,224,3)
    output_shape = train['label'].nunique()
    print('[DEBUG] Input shape: {}, Output shape: {}'.format(
        input_shape, output_shape))

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
        dataframe=train,
        directory=os.path.join(args.base_dir, 'train'),
        batch_size=params['batch_size'],
        target_size=input_shape[:2],
        shuffle=True,
        x_col='file',
        y_col='label',
        class_mode='categorical'
    )
    valid_flow = valid_gen.flow_from_dataframe(
        dataframe=dev,
        directory=os.path.join(args.base_dir, 'dev'),
        batch_size=params['batch_size'],
        target_size=input_shape[:2],
        shuffle=True,
        x_col='file',
        y_col='label',
        class_mode='categorical'
    )
    test_flow = valid_gen.flow_from_dataframe(
        dataframe=test,
        directory=os.path.join(args.base_dir, 'test'),
        batch_size=params['batch_size'],
        target_size=input_shape[:2],
        shuffle=True,
        x_col='file',
        y_col='label',
        class_mode='categorical'
    )

    # Build model and freeze most of it.
    model = build_model(input_shape, output_shape)
    for layer in model.layers[:-2]:
        layer.trainable = False

    for layer in model.layers[-2:]:
        layer.trainable = True

    model_checkpoint = ModelCheckpoint(filepath='weights_{}.hdf5'.format(args.experiment),
                                       monitor='val_loss')
    early_stopping = EarlyStopping(monitor='val_loss', patience=6)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
    callbacks = [model_checkpoint, early_stopping, reduce_lr]

    top3_metric = TopKCategoricalAccuracy(k=3, name='top_3_cat_acc')
    precision_metric = Precision()
    recall_metric = Recall()
    metrics = ['accuracy', top3_metric, precision_metric, recall_metric]

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=metrics)
    history = model.fit_generator(train_flow, steps_per_epoch=params['batches_per_epoch'], epochs=args.max_epochs,
                                  validation_data=valid_flow, workers=4, callbacks=callbacks)
 
    for epoch in range(len(history.history['loss'])):
        wandb.log(
            {
                'epoch':epoch,
                'loss':history.history['loss'][epoch],
                'val_loss':history.history['val_loss'][epoch],
                'accuracy':history.history['accuracy'][epoch],
                'val_accuracy':history.history['val_accuracy'][epoch]
            }
        )
    
    # Train the entire network
    model = load_model('weights_{}.hdf5'.format(args.experiment))

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=metrics
    )

    model_checkpoint = ModelCheckpoint(filepath='weights_stage2_{}.hdf5'.format(args.experiment),
                                       monitor='val_loss')
    callbacks = [model_checkpoint, early_stopping, reduce_lr]
    history_stage2 = model.fit_generator(train_flow, steps_per_epoch=params['batches_per_epoch'],
                                         epochs=args.max_epochs, validation_data=valid_flow,
                                         workers=4, callbacks=callbacks)
    start = len(history.history['loss'])
    for epoch in range(len(history_stage2.history['loss'])):
        wandb.log(
            {
                'epoch':epoch + start,
                'loss':history_stage2.history['loss'][epoch],
                'val_loss':history_stage2.history['val_loss'][epoch],
                'accuracy':history_stage2.history['accuracy'][epoch],
                'val_accuracy':history_stage2.history['val_accuracy'][epoch]
            }
        )

    # Combine history
    for key in history.history.keys():
        if key in history_stage2.history.keys():
            history.history[key].extend(
                history_stage2.history[key]
            )
    
    # Plot metrics
    plot_loss(history=history, name=args.experiment)
    plot_metrics(history=history, name=args.experiment)

    # Save validation folds with class encoded value
    encoding = train_flow.class_indices
    dev['encoded_label'] = dev['label'].apply(lambda x: encoding[x])

    model = load_model('weights_{}.hdf5'.format(args.experiment))

    dev_batches = int(np.ceil(len(dev) // args.batch_size))
    preds, trues = [], []
    for i in range(dev_batches):
        x_batch, y_batch = next(valid_flow)
        preds.extend(np.argmax(model.predict(x_batch), axis=1))
        trues.extend(np.argmax(y_batch, axis=1))

    wandb.log({'dev_balanced_accuracy':balanced_accuracy_score(trues, preds)})
    dev.to_csv('dev_{}.csv'.format(wandb.run.id), index=False)

    test_batches = int(np.ceil(len(test) // args.batch_size))
    preds, trues = [], []
    for i in range(test_batches):
        x_batch, y_batch = next(test_flow)
        preds.extend(np.argmax(model.predict(x_batch), axis=1))
        trues.extend(np.argmax(y_batch, axis=1))

    wandb.log({'test_balanced_accuracy':balanced_accuracy_score(trues, preds)})
        
