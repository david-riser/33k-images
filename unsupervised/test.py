from tensorflow.keras.models import load_model
import numpy as np

def compare_layers(layer1, layer2):
    print(layer1 == layer2)

def compare_weights(weights1, weights2):
    w1, b1 = weights1
    w2, b2 = weights2
    print((b1 == b2).all())
    print(np.sum(w1-w2))
    
model_file1 = 'encoder.smerxkme.hdf5'
model_file2 = 'encoder.w28vxn73.hdf5'
model1 = load_model(model_file1)
model2 = load_model(model_file2)
print(model1.input.shape)
print(model2.input.shape)
print(type(model1.layers[1]))
print(model1.layers[1])


compare_layers(model1.layers[1], model2.layers[1])
compare_layers(model1.layers[1], model1.layers[1])


print(model1.layers[1].get_weights())
print(model2.layers[1].get_weights())



compare_weights(model1.layers[1].get_weights(), model2.layers[1].get_weights())
compare_weights(model1.layers[1].get_weights(), model1.layers[1].get_weights())
