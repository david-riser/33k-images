import logging
import numpy as np
import tensorflow as tf

from base.base_model import BaseModel
from utils.factory import create
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam



class PretrainedModel(BaseModel):
    def __init__(self, config):
        super(PretrainedModel, self).__init__(config)

        self.logger = logging.getLogger('train')
        self.logger.info("Done configuring PretrainedModel")

        self.model_builder = create("tensorflow.keras.applications.{}".format(
            self.config.model.backbone
        ))
        self.build_model()

        
    def build_model(self):
        """ The model is built with most of the layers frozen. """
        
        self.backbone = self.model_builder(
            weights='imagenet',
            pooling=self.config.model.pooling,
            include_top=False
        )

        use_dropout = 'dropout' in self.config.model.toDict()
        use_dense = 'dense_neurons' in self.config.model.toDict()

        if use_dense:
            # Get dense activation type
            if 'dense_activation' in self.config.model.toDict():
                activation = self.config.model.dense_activation
            else:
                activation = 'relu'
                
            x = Dense(self.config.model.dense_neurons, activation=activation)(self.backbone.output)
        else:
            x = self.backbone.output
            
        if use_dropout:
            x = Dropout(self.config.model.dropout)(x)
            outputs = Dense(self.config.n_classes, activation='softmax')(x)
        else:
            outputs = Dense(self.config.n_classes, activation='softmax')(x)

                
        self.model = Model(inputs=self.backbone.input, outputs=outputs)
        
        if use_dense:
            last_frozen_layer = -2
        else:
            last_frozen_layer = -1
            
        for layer in self.model.layers[:last_frozen_layer]:
            layer.trainable = False

        for layer in self.model.layers[last_frozen_layer:]:
            layer.trainable = True

        self.model.compile(
              loss='categorical_crossentropy',
              optimizer=self._build_optimizer(),
              metrics=['accuracy']
        )
        

    def _build_optimizer(self):

        if self.config.model.optimizer.name == 'adam':
            self.logger.info("Setting up Adam")
            return Adam(**self.config.model.optimizer.params.toDict())
        else:
            return Adam(0.001, 0.9, 0.99)
