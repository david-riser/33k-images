import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def get_args():

    def add_required_str_arg(parser, name):
        parser.add_argument(
            '--{}'.format(name),
            required=True,
            type=str)        

    parser = argparse.ArgumentParser()
    add_required_str_arg(parser, 'metadata')
    add_required_str_arg(parser, 'labels')
    add_required_str_arg(parser, 'experiment')
    
    return parser.parse_args()

def plot_loss(data, name):
    
    plt.figure(figsize=(8,6))

    plt.hist(
        data[data['label_y'] == 0]['loss'],
        bins=np.linspace(0,18,20),
        edgecolor='k',
        color='red',
        label='Images',
        normed=True,
        alpha=0.65
    )

    plt.hist(
        data[data['label_y'] == 1]['loss'],
        bins=np.linspace(0,18,20),
        edgecolor='k',
        color='blue',
        label='Diagrams',
        normed=True,
        alpha=0.65
    )

    plt.xlabel('Loss (Categorical Cross Entropy)')
    plt.grid(alpha=0.2)
    plt.legend(frameon=False)
    plt.savefig(name, bbox_inches='tight', dpi=100)
    
def main(args):
    """ 
    
    Load the meta-data that was produced by evaluate_model.py, which
    includes the loss values for all validation images.  Then load the 
    list of labels from ../lists/ and calculate the performance.
  

    """

    try:
        metadata = pd.read_csv(args.metadata)
        labels = pd.read_csv(args.labels)
    except Exception as e:
        print('Trouble loading something: {}'.format(e))
        exit()

    merged = pd.merge(metadata, labels, how='inner', on='file')
    print(merged)
    print(
        merged.groupby('label_y').agg(
            {'loss':np.mean}
        ).reset_index()
    )    

    # Create the figures directory and
    # plot the loss for both classes. 
    if not os.path.exists('figures/'):
        os.mkdir('figures')
    
    plot_loss(merged, 'figures/{}_diagram_loss.png'.format(
        args.experiment
    ))
        
if __name__ == '__main__':
    main(get_args())
