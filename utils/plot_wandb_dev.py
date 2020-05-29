import argparse
import os


if __name__ == "__main__":

    # Collect the W and B code
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--code', type=str, required=True)
    args = ap.parse_args()

    # Setup the workspace
    data = "/home/ubuntu/data/dev"
    dataframe = f"/home/ubuntu/33k-images/unsupervised/dev_kmeans_pca_ms320_{args.code}.csv"
    pdf = f"/home/ubuntu/{args.code}.pdf"

    # Execute command.
    os.system(
        ("python plots.py --dpi=100 --rows=6 --cols=5"
         " --data={} --dataframe={} --pdf={}"
         " --label_col=label --pred_col=pred").format(data, dataframe, pdf)
    )
    
