import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

data = pd.read_csv('clusters.csv')
encoder = LabelEncoder()
data['label_code'] = encoder.fit_transform(data['label'].values)
conf_mat = confusion_matrix(data['label_code'].values, data['cluster'].values)

plt.figure(figsize=(8,6))
plt.imshow(conf_mat, cmap=plt.cm.Reds)
plt.savefig('figures/confusion.png', bbox_inches='tight')
plt.close()
