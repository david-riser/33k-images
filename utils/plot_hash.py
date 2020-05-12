import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_study(data, xname, yname, name):
    """ Retrieve data from the json file
    for xname, yname and create a plot. """

    x = [entry[xname] for entry in data]
    y = [entry[yname] for entry in data]

    plt.figure(figsize=(8,6))
    plt.plot(x, y, marker='o', color='red')
    plt.grid(alpha=0.2)
    plt.savefig(name, bbox_inches='tight', dpi=100)
    
if __name__ == "__main__":

    print("[INFO] Starting plotting.")

    # Load the json file data
    input_file = "hashes_study.json"
    with open(input_file, "r") as inp:
        data = json.load(inp)

    print("[INFO] Loaded data: ", data)

    # Create output directory
    if not os.path.exists("figures"):
        os.makedirs("figures")
        print("[INFO] Created directory for figures.")


    # Sample size figures.
    plot_study(
        data=data['sample_size'],
        xname='sample_size', yname='hash_search_time',
        name='figures/hst_sample_size.png'
    )
    plot_study(
        data=data['sample_size'],
        xname='sample_size', yname='diff_search_time',
        name='figures/dst_sample_size.png'
    )

    # Overlap size figures.
    plot_study(
        data=data['overlap_size'],
        xname='overlap', yname='hash_search_time',
        name='figures/hst_overlap.png',
    )
    plot_study(
        data=data['overlap_size'],
        xname='overlap', yname='diff_search_time',
        name='figures/dst_overlap.png',
    )


