import logging
import numpy as np
import tensorflow as tf

from base.base_model import BaseModel
from utils.factory import create
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten

class PretrainedLinearModel(BaseModel):
    def __init__(self, config):
        super(PretrainedLinearModel, self).__init__(config)
        self.model_builder = create("tensorflow.keras.applications.{}".format(
            self.config.model.backbone
        ))
        self.build_model()

        logger = logging.getLogger('train')
        logger.info("Done configuring PretrainedLinearModel")
        
    def build_model(self):
        """ The model is built with most of the layers frozen. """
        
        self.backbone = self.model_builder(
            weights='imagenet',
            pooling=self.config.model.pooling,
            include_top=False
        )

        #x = Dense(2048, activation='relu')(
        #    self.backbone.output
        #)
        outputs = Dense(self.config.n_classes, activation='softmax')(self.backbone.output)
        self.model = Model(inputs=self.backbone.input, outputs=outputs)
        self.model.compile(
              loss='sparse_categorical_crossentropy',
              optimizer=self.config.model.optimizer,
              metrics=['accuracy']
        )
        
        for layer in self.model.layers[:-1]:
            layer.trainable = False

        for layer in self.model.layers[-1:]:
            layer.trainable = True
