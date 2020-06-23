import logging
import os
from base.base_trainer import BaseTrain
from tensorflow.keras.callbacks import ModelCheckpoint


class SupervisedTrainer(BaseTrain):
    """ A standard supervised training class that 
    uses cross-entropy loss. """
    def __init__(self, model, data, config):
        super(SupervisedTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

    def train(self):
        history = self.model.model.fit(
            x=self.data.get_train_flow(),
            validation_data=self.data.get_dev_flow(),
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            callbacks=self.callbacks,
            steps_per_epoch=self.config.trainer.steps_per_epoch
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['accuracy'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_accuracy'])
