import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Some clustering utilities 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, confusion_matrix

# Compute the Adjusted Rand Score
data = pd.read_csv('clusters.csv')
encoder = LabelEncoder()
data['label_code'] = encoder.fit_transform(data['label'].values)
ar_score = adjusted_rand_score(data['label_code'].values, data['cluster'].values)
print("Adjusted Rand Score: {0:8.6f}".format(ar_score))

# Print it a few times for random trials
for i in range(12):
    ar_score = adjusted_rand_score(data['label_code'].sample(frac=1).values,
                                   data['cluster'].values)
    print("Adjusted Rand Score (randomized): {0:8.6f}".format(ar_score))

ar_score = adjusted_rand_score(data['cluster'].values,
                               data['cluster'].values)
print("Adjusted Rand Score (exact labels): {0:8.6f}".format(ar_score))

# Number of classes
n_classes = data['label_code'].nunique()
side_len = int(np.ceil(np.sqrt(len(data))))
print(f"Creating {side_len} side length figure for {n_classes} classes.")

colors = cm.rainbow(0, 1, n_classes)
print(colors)

# For figure
true_label_sorted_data = data.sort_values('label_code')
pred_label_sorted_data = data.sort_values('cluster')

ideal_figure = np.zeros((side_len, side_len))
not_ideal_figure = np.zeros((side_len, side_len))
for row in range(side_len):
    for col in range(side_len):
        index = col + row * side_len
        if index < len(data):
            ideal_figure[row, col] = true_label_sorted_data['label_code'].values[index]
            not_ideal_figure[row, col] = pred_label_sorted_data['label_code'].values[index]

# Create the map from ideal points
for cmap in ['rainbow', 'plasma', 'viridis', 'Reds', 'Blues']:
    plt.figure(figsize=(16,6), dpi=100)
    plt.subplot(1, 2, 1)
    plt.imshow(ideal_figure, cmap=plt.cm.get_cmap(cmap, n_classes))
    plt.title('True Label Assignment')
    plt.subplot(1, 2, 2)
    plt.imshow(not_ideal_figure, cmap=plt.cm.get_cmap(cmap, n_classes))
    plt.title('Cluster Assignment')
    plt.savefig('figures/cluster_colormap_{}.png'.format(cmap.lower()),
                bbox_inches='tight')
    plt.close()

