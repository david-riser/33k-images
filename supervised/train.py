import argparse
import numpy as np
import os
import pandas as pd

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train(model, params, train_flow, valid_flow,
          patience=10, savename='model.h5'):
    """ 
    Train the model according to the parameters given, this
    function is meant to work with sherpa. 

    params (dict):
    -------
    This must contain the following variables.
    - max_epochs
    - batches_per_epoch
    - batch_size

    """
    
    # Patience and Model Checkpointing
    down_rounds = 0
    best_loss = np.inf
    
    t_loss, v_loss = [], []
    for epoch in range(params['max_epochs']):

        batch_loss = 0.
        for batch in range(params['batches_per_epoch']):
            X_batch, Y_batch = next(train_flow)
            batch_loss += model.train_on_batch(X_batch, Y_batch)

        # Training loss from above
        t_loss.append(batch_loss / params['batches_per_epoch'])

        # Evaluate validation set
        X_valid, Y_valid = next(valid_flow)
        v_loss.append(model.evaluate(X_valid, Y_valid))

        # Model checkpoint and early stopping.
        if v_loss[epoch] < best_loss:
            best_loss = v_loss[epoch]
            down_rounds = 0
            model.save(savename)
            print("[New Best] Epoch {0}, Training Loss: {1:8.4f}, Testing Loss: {2:8.4f}".format(
                epoch, t_loss[epoch], v_loss[epoch]))
        else:
            down_rounds += 1

        if down_rounds >= patience:
            print("Earlying stopping at epoch {}.".format(epoch))
            break

        
    return t_loss, v_loss
