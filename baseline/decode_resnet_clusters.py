import pickle

from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Model
from tensorflow.keras.layers import Input


with open('artifacts/ResNet50_avg_centroids.pkl', 'rb') as f:
    centroids = pickle.load(f)

clusters, features = centroids['centroids'].shape
resnet = ResNet50(weights='imagenet', include_top=True)
#model = Model(inputs=Input((features,)), outputs=resnet.get_layer('probs').output)
model = Model(inputs=resnet.get_layer('probs').input, outputs=resnet.get_layer('probs').output)
#pred = model.predict(centroids['centroids'])
#print(decode_predictions(pred, 5))
