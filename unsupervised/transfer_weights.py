import argparse
import boto3
import glob
import os


if __name__ == "__main__":

    # Collect the bucket name from the command line.
    ap = argparse.ArgumentParser()
    ap.add_argument('--bucket', type=str, required=True)
    args = ap.parse_args()

    # Find all models.
    models = glob.glob('*.hdf5')

    conn = boto3.client('s3')

    for model in models:
        conn.upload_file(model, args.bucket, model)
        
    
