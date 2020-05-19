# Standard library imports 
import os

# Third-party imports
import numpy as np
import tqdm

from sklearn.cluster import MiniBatchKMeans

# This project
from . import utils
from . import models

def train_clustering_model(model, X_train, max_iter,
                           update_interval, batch_size,
                           verbose=True):
    """
    Training the deep clustering model involves
    alternating steps between soft assignment of labels,
    calculating an ideal target distribution, and running
    a series of supervised learning steps to minimize the
    KL Divergence between the soft assignments and the
    target distribution.

    The total number of examples seen by the network
    before updating the target distribution is:
    examples = batch_size * update_interval

    :param model: PretrainedDeepClusteringModel to be trained
    :param X_train: An input tensor to train on.
    :param max_iter: The maximum number of supervised learning steps
    in total.  This is the number of batches that will be processed.
    :param update_interval: How often the target distribution is updated.
    :param batch_size: The number of instances in each training batch.

    :returns kld_loss: A list of loss values throughout the training.
    """
    
    coverage_fraction = float((batch_size * max_iter) / X_train.shape[0])
    if coverage_fraction < 1.0:
        print(("[WARNING] project_core.train.train_clustering_model has detected"
               "a coverage fraction of {0:4.2f}.  This means that your model is"
               "not going to see all of your data.  Consider increasing the value"
               "of max_iter or the batch size to allow your model to see more data.".format(
            coverage_fraction
        )))

    kld_loss = []
    loss = np.inf
    for ite in range(int(max_iter)):

        if ite % update_interval == 0:
            

            q = model.predict(X_train, verbose=0)
            p = utils.clustering_target_distribution(q)


        # Sample a batch of size batch_size from the training examples
        idx = np.random.choice(X_train.shape[0], batch_size, replace=False)
        loss = model.train_on_batch(x=X_train[idx], y=p[idx])
        kld_loss.append(loss)

        if verbose:
            print("[INFO] Epoch: {0}, Loss: {1:8.4f}".format(ite, loss))
            
    return kld_loss


def train_clustering_model_generator(model, gen, max_iter,
                                     update_interval, batch_size,
                                     verbose=True):
    """

    :param model: PretrainedDeepClusteringModel to be trained
    :param gen: An input ImageDataGenerator.
    :param max_iter: The maximum number of supervised learning steps
    in total.  This is the number of batches that will be processed.
    :param update_interval: How often the target distribution is updated.
    :param batch_size: The number of instances in each training batch.

    :returns kld_loss: A list of loss values throughout the training.
    """
    

    kld_loss = []
    loss = np.inf
    for ite in range(int(max_iter)):

        if ite % update_interval == 0:
            
            batch = next(gen)
            while len(batch) != batch_size:
                batch = next(gen)
                
            q = model.predict(batch, verbose=0)
            p = utils.clustering_target_distribution(q)


        batch = next(gen)
        if len(batch) == batch_size:
            loss = model.train_on_batch(x=batch, y=p)
            kld_loss.append(loss)

        if verbose:
            print("[INFO] Epoch: {0}, Loss: {1:8.4f}".format(ite, loss))
            
    return kld_loss


class KMeansImageDataGeneratorWrapper:

    def __init__(self, keras_model, **kwargs):
        self.model = keras_model
        self.kmeans = MiniBatchKMeans(**kwargs)
        
    def fit_generator(self, generator, epochs, steps_per_epoch):
        
        for epoch in range(epochs):
            print(f"[INFO] Processing epoch {epoch} of {epochs}.")
            for step in tqdm.tqdm(range(steps_per_epoch)):
                self.kmeans.partial_fit(self.model.predict(
                    next(generator)
                ))
                
                
    def predict(self, generator, steps):
        clusters = []
        for step in range(steps):
            clusters.extend(self.kmeans.predict(
                self.model.predict(
                    next(generator)
            )))

        return clusters
    
    @property
    def centroids(self):
        return self.kmeans.cluster_centers_
