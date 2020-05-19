import numpy as np
import os
import sys

from tensorflow.keras import Model
from tensorflow.keras.applications import resnet50

PROJ_DIR = os.path.abspath(os.path.join(os.getcwd(), '../'))
sys.path.append(PROJ_DIR)

from project_core.models import LinearModel


if __name__ == "__main__":
    

    encoder = resnet50.ResNet50(
        include_top=False,
        pooling='max'
    )

    encoder.compile(optimizer='adam', loss='categorical_crossentropy')

    linear = LinearModel(encoder=encoder, n_classes=10)
    linear.compile(optimizer='adam', loss='categorical_crossentropy')
    print(linear.summary())
