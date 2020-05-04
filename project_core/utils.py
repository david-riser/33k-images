import numpy as np

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

