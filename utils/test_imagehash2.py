import numpy as np
import imagehash
import json
import time
import os
import sys

PROJECT_DIR = os.path.join(os.getcwd(), '..')
sys.path.append(PROJECT_DIR)

from project_core.duplicates import hash_search

def create_image_dataset(imagesize, samples, overlaps):
    """ Create a testing set. """

    images = []
    matches = []
    
    if overlaps >= samples:
        print('[FATAL] The number of overlaps has to be smaller than the number of samples.')
        exit()
        
    for _ in range(samples - overlaps):
        images.append(np.random.randint(
            0, 255, size=imagesize, dtype=np.uint8
        ))

    duplicates = np.random.choice(np.arange(0, len(images)-1),
                                 overlaps, replace=False)
    for dupe in duplicates:
        images.append(np.copy(images[dupe]))
        matches.append((dupe,len(images)-1))
        
    return images, matches

def validate(y_true, y_pred):
    total = len(y_true)
    matches = 0

    for yp in y_pred:
        if yp in y_true:
            matches += 1

    return float(matches / total)

if __name__ == "__main__":

    image_shape = (36, 36, 3)

    # Study the performance w.r.t. sample size
    sample_sizes = range(1000, 30000, 5000)
    print(sample_sizes)
    for sample_size in sample_sizes:
        overlaps = int(sample_size // 10)
        print('[DEBUG] Sample size {}, Overlap {}'.format(
            sample_size, overlaps
        ))

        data = {}
        data['sample_size'] = sample_size
        
        images, matches = create_image_dataset(
            imagesize=image_shape,
            samples=sample_size,
            overlaps=overlaps
        )
        
        # Test difference method.
        start_time = time.time()
        dmatches = [tuple(e) for e in hash_search(images)]
        data['search_time'] = time.time() - start_time
        data['search_validity'] = validate(matches, dmatches)

        print(data)
