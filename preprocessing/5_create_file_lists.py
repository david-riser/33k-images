import argparse
import numpy as np
import os
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    return parser.parse_args()

def main(args):

    labels, images = [], []
    for folder, _, files in os.walk(args.data):
        if folder != args.data:
            for f in files:
                label = folder.split('/')[-1]
                image = os.path.join(label, f)

                labels.append(label)
                images.append(image)

    data = pd.DataFrame(
        {'label':labels, 'file':images}
    )
    data.to_csv(os.path.join(args.output_dir, 'good_images.csv'), index=False)

    # Now write some files for threshold values
    # of minimum class size.
    support = data.groupby('label').aggregate(
        {'file':len}
    ).reset_index()

    thresholds = [100, 200, 300, 400, 500]
    for t in thresholds:
        large_classes = support['label'][np.where(support['file'] > t)[0]].values

        # Save those good ones
        data['save'] = data['label'].apply(lambda x: x in large_classes)
        data_subset = data[data['save'] == True]
        data_subset.to_csv(os.path.join(
            args.output_dir, 'good_images_{}.csv'.format(t)), index=False)
    
if __name__ == "__main__":
    main(get_args())
