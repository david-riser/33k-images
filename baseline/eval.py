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

agg = data.groupby(['label_code']).agg({'cluster':lambda x: np.unique(x, return_counts=True)}).reset_index()
print(agg)

plt.figure(figsize=(32,32))
sns.heatmap(conf_mat, annot=True)
plt.savefig('figures/confusion.png', bbox_inches='tight')
plt.close()
