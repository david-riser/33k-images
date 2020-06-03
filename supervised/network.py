from tensorflow.keras import Model
from tensorflow.keras.applications import resnet50
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout

def build_model(input_shape, output_shape):
    pretrained_model = resnet50.ResNet50(
        weights='imagenet',
        include_top=False
    )
    x = pretrained_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(output_shape, activation='softmax')(x)
    model = Model(inputs=pretrained_model.input, outputs=outputs)
    return model
