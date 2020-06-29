import logging
import os
from base.base_trainer import BaseTrain
from tensorflow.keras.callbacks import ModelCheckpoint


class PretrainedKMeansTrainer(BaseTrain):
    """ This class just pushes some batches through the 
    pca and kmeans portions of the model so that the weights
    are reasonable before making predictions. """
    def __init__(self, model, data, config):
        super(PretrainedKMeansTrainer, self).__init__(model, data, config)

        
    def train(self):
        logger = logging.getLogger('train')
        logger.debug('Start of call to model.fit')

        batch_size = 32
        if 'batch_size' in self.config.data_loader.toDict():
            batch_size = self.config.data_loader.batch_size
        
        # Push batches through iterative PCA first if the
        # model uses it.
        flow = self.data.get_train_flow()

        if self.model.use_pca:
            epochs = self.data.n_training_samples // batch_size
            for epoch in range(epochs):
                logger.debug('Calling IterativePCA on epoch {} of {}.'.format(
                    epoch, epochs))
                self.model.pca.partial_fit(self.model.encoder.predict(
                    next(flow)
                ))

        # Push batches through mini-batch k-means
        for epoch in range(epochs):
            logger.debug('Calling MBKMeans on epoch {} of {}.'.format(
                epoch, epochs))
            self.model.kmeans.partial_fit(self.model.pca.transform(self.model.encoder.predict(
                next(flow)
            )))
        
                
        logger.debug('End of call to model.fit')
        logger.debug('End of call to PretrainedKMeansTrainer.train')
