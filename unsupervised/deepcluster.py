import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import wandb

PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), '../'))
print(PROJECT_DIR)
sys.path.append(PROJECT_DIR)

from project_core.models import model_factory, PretrainedDeepClusteringModel, LinearModel
from project_core.utils import load_image, build_files_dataframe, prune_file_list
from project_core.train import train_clustering_model_generator, KMeansImageDataGeneratorWrapper
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main(args):
 
    print('[INFO] Starting deep clustering...')

    # Setup weights and biases
    setup_wandb(args)
    
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
    model.initialize_clusters_generator(train_flow, epochs=args.kmeans_epochs,
                                        steps_per_epoch=int(np.ceil(len(train) / args.batch_size)))
    wandb.log({'inertia':model.inertia})
    
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

    print('[INFO] Running linear evaluation...')
    encoder = Model(model.backbone.inputs, model.backbone.outputs)

    for layer in encoder.layers:
        encoder.trainable = False

    label_encoder = LabelEncoder()
    train['encoded_label'] = label_encoder.fit_transform(train['label'])
    dev['encoded_label'] = label_encoder.transform(dev['label'])

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

    linear_eval(
        encoder=encoder,
        train_gen=train_flow,
        dev_gen=dev_flow,
        train_labels=train['encoded_label'],
        dev_labels=dev['encoded_label'],
        metric=balanced_accuracy_score,
        log_training=True
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
    ar_score = adjusted_rand_score(dev['label'], pred)
    print("[INFO] Adjusted Rand Index {0:6.4f}".format(
        ar_score
    ))

    # Log the adjusted rand score.
    wandb.log({'ari':ar_score})

    for epoch, loss in enumerate(kld):
        wandb.log({'epoch':epoch, 'kl_divergence':loss})
    
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
    ap.add_argument('--kmeans_epochs', type=int, default=1)
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


def setup_wandb(args):
    """ Setup weights and biases logging. """

    config = dict(
        batch_size = args.batch_size,
        epochs = args.epochs,
        update_interval = args.update_interval,
        architecture = ":".join([args.backbone, args.pooling]),
        kmeans_epochs = args.kmeans_epochs
    )

    wandb.init(
        project='33ku',
        notes='deepcluster.py fine tuning',
        tags=['fine-tuning', 'deepcluster'],
        config=config
    )


def linear_eval(encoder, train_gen, dev_gen,
                train_labels, dev_labels,
                metric=None, log_training=False,
                epochs=20, steps_per_epoch=32):
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
        
if __name__ == "__main__":
    main(get_args())


