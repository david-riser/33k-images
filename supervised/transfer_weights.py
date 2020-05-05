import argparse
import glob
import os


if __name__ == "__main__":

    # Collect the bucket name from the command line.
    ap = argparse.ArgumentParser()
    ap.add_argument('--bucket', type=str, required=True)
    args = ap.parse_args()

    # Find all models.
    models = glob.glob('*.hdf5')

    # Move them
    for model in models:
        os.system(f'aws s3 cp {model} s3://{args.bucket}')
