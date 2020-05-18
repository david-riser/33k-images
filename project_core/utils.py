import numpy as np
import os
import pandas as pd

import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical


def load_image(image_path, target_size=(224,224), preprocess_input=None):
    """ Load and preprocess an image."""

    x = image.load_img(image_path, target_size=target_size)
    x = np.expand_dims(x, axis=0)

    if preprocess_input:
        x = preprocess_input(x)

    return x

def predict_images(model, image_paths, target_size=(224,224),
                   preprocess_input=None):
    """ 
    Load images specified by their paths and predict
    whatever the model predicts.  Return that.

    """

    images = np.zeros((len(image_paths), target_size[0], target_size[1], 3))
    for i, image_path in enumerate(image_paths):
        try:
            images[i,:,:,:] = load_image(image_path, target_size, preprocess_input)
        except Exception as e:
            print('Trouble loading: {}'.format(image_path))

    return model.predict(images)

def get_image_loss(preds, labels, target_size=(224,224),
                   preprocess_input=None):
    """ 
    Load images specified by their paths and predict
    whatever the model predicts.  Return that.

    """

    assert(len(preds) == len(labels))
    clabels = to_categorical(labels)
    cost = CategoricalCrossentropy(reduction='none')
    return cost(preds, clabels).numpy()

def infer_model_shapes(model):
    """
    Infer the model input and output shape for a Keras model.
    :param model: An instance of tensorflow.keras.models.Model
    :return: input_shape, output_shape (2 tuples)
    """
    input_shape = model.input.shape.value
    output_shape = model.output.shape[1].value
    return input_shape, output_shape

def create_directory(path, recursive=False):
    """
    If the output directory does not exist
    create it.

    :param path: A string that contains the
    directory path to be created.
    :param recursive: If True, all directories are created.
    :return: True/False boolean value that indicates existence
    of or creation of target directory.
    """
    make_dir = os.makedirs if recursive else os.mkdir
    if not os.path.exists(path):
        try:
            make_dir(path)
            return True
        except Exception as e:
            print('Error in the creation of directory: {}'.format(path))
            return False

    return True

def clustering_target_distribution(q):
    """
    This is an empirical function used to define good
    cluster centers in the deep clustering scheme.

    :param q: Cluster assignment probabilities.
    :return: The tensor which defines the new probabilities.
    """
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def list_greyscale_images(image_paths, target_size=(224,224)):
    """ 
    Load images specified by their paths and see if they
    are greyscale.

    The channel axis is axis because we have a four 
    dimensional matrix at this point. 
    (batch_size, height, width, channels)
    """

    output = []
    for i, image_path in enumerate(image_paths):
        try:
            img = load_image(image_path, target_size)
            if img.std(axis=3).mean() == 0:
                output.append(1)
            else:
                output.append(0)
            
        except Exception as e:
            print('Trouble loading: {}'.format(image_path))
            output.append(2)
            
    return output

def print_directory_stats(project_dir):
    """ 
    Print a summary of the number of files and 
    folders currently in the project. 

    :param project_dir: A string that specifies the
    path to the data folders.

    :returns: None
    """

    counter = {}
    for folder, _, files in os.walk(project_dir):
        if folder != project_dir:
            counter[folder] = len(files)

    print('[INFO] There are {} images in {} folders.'.format(
        sum(list(counter.values())), len(list(counter.keys()))
    ))

    
def prune_file_list(data, label_col, min_samples):
    """ 
    
    Given a list of files, prune the list to contain 
    only the entries for which the label specified by
    label_col contains at least min_samples.
    
    :param data: Dataframe containing at least one column 
    called label_col
    :param label_col: The column used to specify class
    :param min_samples: The minimum number of samples a
    class needs to have to be kept.

    """

    if label_col not in list(data.columns):
        raise ValueError("[ERROR] {} not in dataframe columns!".format(
            label_col
        ))
    return_cols = list(data.columns)
    
    keep = data.groupby(label_col).transform(
        lambda x: len(x) > min_samples
    ).values
    return data.iloc[keep][return_cols]
    
if __name__ == "__main__":

    test_data = pd.DataFrame({
        'file':np.arange(10),
        'label':[0, 0, 0, 1, 1, 2, 2, 2, 2, 3]
    })

    filtered = prune_file_list(
        data=test_data, label_col='label',
        min_samples=2
    )

    print(test_data)
    print(filtered)
