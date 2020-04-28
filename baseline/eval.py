import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

data = pd.read_csv('clusters.csv')
encoder = LabelEncoder()
data['label_code'] = encoder.fit_transform(data['label'].values)
conf_mat = confusion_matrix(data['label_code'].values, data['cluster'].values)
assignments = np.argmax(conf_mat, axis=0)
print(assignments)


plt.figure(figsize=(12,12))
sns.heatmap(conf_mat, annot=True)
plt.savefig('figures/confusion.png', bbox_inches='tight', dpi=200)
plt.close()

plt.figure(figsize=(12,12))
sns.heatmap(np.divide(conf_mat, conf_mat.sum(axis=0)), annot=True)
plt.savefig('figures/conflusion_normed_by_truth.png', bbox_inches='tight', dpi=200)
plt.close()

plt.figure(figsize=(12,12))
sns.heatmap(np.divide(conf_mat, conf_mat.sum(axis=1)), annot=True)
plt.savefig('figures/conflusion_normed_by_cluster_size.png', bbox_inches='tight', dpi=200)
plt.close()
