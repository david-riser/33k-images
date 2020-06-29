import logging
import numpy as np
import tensorflow as tf

from base.base_model import BaseModel
from utils.factory import create
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA


class MockModel:

    def __init__(self, encoder, pca, kmeans):
        self.encoder = encoder
        self.kmeans = kmeans
        self.pca = pca
        
    def predict(self, x):
        if self.pca:
            return self.kmeans.predict(
                self.pca.transform(
                    self.encoder.predict(x)
                ))
        else:
            return self.kmeans.predict(
                self.encoder.predict(x))

        

class PretrainedKMeansModel(BaseModel):
    def __init__(self, config):
        super(PretrainedKMeansModel, self).__init__(config)

        self.logger = logging.getLogger('train')
        self.logger.info("Done configuring PretrainedKMeansModel")

        self.model_builder = create("tensorflow.keras.applications.{}".format(
            self.config.model.backbone
        ))

        # Set the number of clusters, but with the
        # condition that if the number is "auto", we set
        # the number to the number of classes.
        if self.config.model.n_clusters == "auto":
            self.n_clusters = self.config.n_classes
        else:
            self.n_clusters = self.config.model.n_clusters

        self.logger.info("PretrainedKMeansModel setup using {} clusters.".format(
            self.n_clusters
        ))
        self.build_model()

        
    def build_model(self):
        """ 

        Build a prediction pipeline that 
        consists of a pretrained network encoder, 
        a dimensionality reduction, and a clustering. 

        """
        
        self.encoder = self.model_builder(
            weights='imagenet',
            pooling=self.config.model.pooling,
            include_top=False
        )
                
        
        self.use_pca = 'pca_components' in self.config.model.toDict()
        if self.use_pca:
            self.logger.debug('Setting up PretrainedKMeansModel with PCA')
            self.pca = IncrementalPCA(n_components=self.config.model.pca_components)

        self.kmeans = MiniBatchKMeans(self.n_clusters)

        # This little MockModel class is used to
        # ensure that the evaluators have a standard
        # prediction interface by calling model.model.predict(x).
        if self.use_pca:
            self.model = MockModel(self.encoder, self.pca, self.kmeans)
        else:
            self.model = MockModel(self.encoder, None, self.kmeans)
