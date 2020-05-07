import numpy as np
import os

from tensorflow.keras.preprocessing import image

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
    make_dir = os.path.makedirs if recursive else os.path.mkdir
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

