# Standard library imports
import os

# Third party imports
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.applications import (inception_v3,
                                           nasnet, resnet50,
                                           xception)
from tensorflow.keras.layers import InputSpec, Layer, Dense

# This project
from . import train


MODELS = ['InceptionV3', 'NASNet',
            'ResNet50', 'Xception']


def model_factory(model_name, pooling=None):

    if model_name not in MODELS:
        print(("[FATAL] Model name {} not found in models."
               "  Please choose from the following list.  {}").format(
               model_name, MODELS))
        exit()

    elif model_name == 'InceptionV3':
        print('Instantiating a InceptionV3')
        model = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling=pooling)
        preprocess_input = inception_v3.preprocess_input

    elif model_name == 'NASNet':
        print('Instantiating a NASNet')
        model = nasnet.NASNetLarge(weights='imagenet', include_top=False, pooling=pooling)
        preprocess_input = nasnet.preprocess_input

    elif model_name == 'ResNet50':
        print('Instantiating a ResNet50')
        model = resnet50.ResNet50(weights='imagenet', include_top=False, pooling=pooling)
        preprocess_input = resnet50.preprocess_input

    elif model_name == 'Xception':
        print('Instantiating a Xception')
        model = xception.Xception(weights='imagenet', include_top=False, pooling=pooling)
        preprocess_input = nasnet.preprocess_input

    return model, preprocess_input

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        # self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        # self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(
            name='clusters',
            shape=(self.n_clusters, input_dim),
            initializer='glorot_uniform',
            trainable=True)
        super(ClusteringLayer, self).build(input_shape)
        
    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class PretrainedDeepClusteringModel(Model):

    def __init__(self, n_clusters, backbone):
        super(PretrainedDeepClusteringModel, self).__init__()
        self.backbone = backbone
        self.n_clusters = n_clusters
        self.is_initialized_ = False

        # Create the clustering layer and build the model as
        # the pretrained network plus the clustering layer.
        self.clustering_layer = ClusteringLayer(
            n_clusters=n_clusters, name='clustering',
            input_shape=self.backbone.output.shape)
        self.clustering_layer.trainable = True
        self.clustering_layer.build(self.backbone.output.shape)
        
    def call(self, inputs):
        x = self.backbone(inputs)
        output = self.clustering_layer(x)
        return output

    @property
    def is_initialized(self):
        return self.is_initialized_

    def initialize_clusters(self, x):
        """
        Run KMeans on the input data and setup
        a good starting point for clusters.  First,
        the data is transformed into the latent space
        by calling the backbone.
        """
        z = self.backbone.predict(x)
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(z)
        self.clustering_layer.set_weights(
            [kmeans.cluster_centers_]
        )
        self.is_initialized_ = True

    def initialize_clusters_generator(self, gen, epochs, steps_per_epoch):
        kmeans = train.KMeansImageDataGeneratorWrapper(
            keras_model=self.backbone, n_clusters=self.n_clusters
        )
        kmeans.fit_generator(gen, epochs=epochs, steps_per_epoch=steps_per_epoch)
        self.clustering_layer.set_weights(
            [kmeans.centroids]
        )
        self.inertia_ = kmeans.inertia
        self.is_initialized_ = True

    @property
    def inertia(self):
        if self.is_initialized_:
            return self.inertia_
        else:
            return np.inf


class LinearModel(Model):

    def __init__(self, encoder, n_classes):
        super(LinearModel, self).__init__()
        self.encoder = encoder
        self.linear = Dense(
            units=n_classes,
            activation='softmax'
        )
        self.linear.build(self.encoder.output.shape)
        self.build(self.encoder.input.shape)
        self.linear.trainable = True
        
        for layer in self.encoder.layers:
            layer.trainable = False

    def call(self, x):
        output = self.linear(self.encoder(x))
        return output
