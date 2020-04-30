import glob
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

# Some clustering utilities 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from metrics import hungarian_method

def plot_cluster_viz(true_label_sorted_data, pred_label_sorted_data,
                     backbone, pooling):
    """ 
    Create a figure for the visualization of clustering results. 
    """

    side_len = int(np.ceil(np.sqrt(len(true_label_sorted_data))))
    n_classes = true_label_sorted_data['label_code'].nunique()
    print('Creating visual for {}:{} size: {}x{} with {} classes'.format(
        backbone, pooling, side_len, side_len, n_classes
    ))

    # Create images
    ideal_figure = np.zeros((side_len, side_len))
    not_ideal_figure = np.zeros((side_len, side_len))
    for row in range(side_len):
        for col in range(side_len):
            index = col + row * side_len
            if index < len(true_label_sorted_data):
                ideal_figure[row, col] = true_label_sorted_data['label_code'].values[index]
                not_ideal_figure[row, col] = pred_label_sorted_data['label_code'].values[index]

    # Create the map from ideal points
    for cmap in ['Spectral', 'jet']:
        plt.figure(figsize=(16,6), dpi=100)
        plt.subplot(1, 2, 1)
        plt.imshow(ideal_figure, cmap=plt.cm.get_cmap(cmap, n_classes))
        plt.title('True Label Assignment')
        plt.subplot(1, 2, 2)
        plt.imshow(not_ideal_figure, cmap=plt.cm.get_cmap(cmap, n_classes))
        plt.title('Cluster Assignment')
        plt.savefig('figures/cluster_{}_{}_colormap_{}.png'.format(backbone, pooling, cmap.lower()),
                    bbox_inches='tight')
        plt.close()

def plot_confusion(true, pred, backbone, pooling, keyword):
    confmat = confusion_matrix(true, pred)
    plt.figure(figsize=(8,6), dpi=200)
    sns.heatmap(confmat, cmap="plasma", annot=True, fmt='d')
    plt.xlabel('Cluster Label')
    plt.ylabel('Truth Label')
    plt.savefig('figures/confusion_{}_{}_{}.png'.format(keyword, backbone, pooling),
                bbox_inches='tight')
    plt.close()

    
if __name__ == "__main__":

    input_folder = './artifacts'
    data = {}
    rand_index = {}
    name_lut = {}
    for csvfile in glob.glob(input_folder + '/*'):

        backbone = csvfile.split('/')[-1].split('_')[0]
        pooling = csvfile.split('/')[-1].split('_')[1]
        name_lut[csvfile] = ':'.join([backbone, pooling])
        
        data[csvfile] = pd.read_csv(csvfile)
        encoder = LabelEncoder()
        data[csvfile]['label_code'] = encoder.fit_transform(data[csvfile]['label'].values)
        assignment = hungarian_method(data[csvfile]['label_code'], data[csvfile]['cluster'])

        plot_confusion(data[csvfile]['label_code'], data[csvfile]['cluster'],
                       backbone, pooling, 'before_mapping')
        
        for pair in assignment:
            data[csvfile]['cluster'].replace(pair[0], pair[1], inplace=True)

        plot_confusion(data[csvfile]['label_code'], data[csvfile]['cluster'],
                       backbone, pooling, 'after_mapping')

        weight_acc = balanced_accuracy_score(data[csvfile]['label_code'], data[csvfile]['cluster'])
        print("Balanced Accuracy Score: {0:8.6f}".format(weight_acc))
        rand_index[csvfile] = adjusted_rand_score(
            data[csvfile]['label_code'].values, data[csvfile]['cluster'].values)
        print("Adjusted Rand Score ({0}): {1:8.6f}".format(csvfile, rand_index[csvfile]))
        
        true_label_sorted_data = data[csvfile].sort_values('label_code')
        pred_label_sorted_data = data[csvfile].sort_values('cluster')
        plot_cluster_viz(true_label_sorted_data, pred_label_sorted_data, backbone, pooling)

    # Create a bar chart with ARS
    plt.figure(figsize=(12,12))
    plt.barh(
        list(name_lut.values()),
        list(rand_index.values()),
        edgecolor='k'
    )
    plt.grid(alpha=0.2)
    plt.title('Image Clusters w/ Transfer Learning (untuned)')
    plt.xlabel('Adjusted Rand Index')
    plt.xlim([0.0, 0.4])
    plt.savefig('figures/rand_barh.png', bbox_inches='tight', dpi=50)
    plt.close()
