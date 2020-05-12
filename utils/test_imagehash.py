import numpy as np
import imagehash
import json
import time

from PIL import Image

def hash_duplicate_search(images):
    """ 
    
    Search one set of images for duplicates.  The
    method used here is hashing the image and 
    then mapping it into the hashmap.  This finds 
    exact matches but also reports collisions in the
    hashtable.  Therefore the duplicate matching 
    algorithm must be used on the output of this.

    """

    hashtable = {}
    matches = []
    for i, image in enumerate(images):
        hashed = imagehash.dhash(Image.fromarray(image))
        if hashed not in hashtable:
            hashtable[hashed] = i
        else:
            matches.append((hashtable[hashed],i))
        
    return matches

def diff_duplicate_search(images):
    """ 

    Search one set of images for duplicates using 
    the different method. 
    
    """
    matches = []
    def check(img1, img2):
        arr1, arr2 = np.asarray(img1), np.asarray(img2)
        return np.sum((arr2 - arr1)**2) == 0

    for i in range(len(images)):
        for j in range(i+1,len(images)):
            if check(images[i], images[j]):
                matches.append((i,j))
    

    return matches


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

    # Place to keep our work!
    studies = {}
    studies['sample_size'] = []
    studies['overlap_size'] = []
    studies['image_size'] = []

    # Study the performance w.r.t. sample size
    # sample_sizes = [10, 50, 100, 200, 400, 600, 800, 1000]
    sample_sizes = range(10, 1000, 20)
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
        dmatches = diff_duplicate_search(images)
        data['diff_search_time'] = time.time() - start_time
        data['diff_search_validity'] = validate(matches, dmatches)

        # Test hash method. .
        start_time = time.time()
        dmatches = hash_duplicate_search(images)
        data['hash_search_time'] = time.time() - start_time
        data['hash_search_validity'] = validate(matches, dmatches)

        print(data)
        studies['sample_size'].append(data)
        
    # overlap_sizes = [0.01, 0.05, 0.1, 0.2, 0.4]
    overlap_sizes = np.linspace(0.01, 0.4, 20)
    for overlap in overlap_sizes:
        data = {}
        data['overlap'] = overlap
        
        images, matches = create_image_dataset(
            imagesize=image_shape,
            samples=1000,
            overlaps=int(overlap * 1000)
        )
        
        # Test difference method.
        start_time = time.time()
        dmatches = diff_duplicate_search(images)
        data['diff_search_time'] = time.time() - start_time
        data['diff_search_validity'] = validate(matches, dmatches)

        # Test hash method. .
        start_time = time.time()
        dmatches = hash_duplicate_search(images)
        data['hash_search_time'] = time.time() - start_time
        data['hash_search_validity'] = validate(matches, dmatches)

        print(data)
        studies['overlap_size'].append(data)

    image_sizes = [(8,8,3), (16,16,3), (32,32,3), (64,64,3), (128,128,3)]
    for image_size in image_sizes:
        data = {}
        data['image_size'] = image_size
        
        images, matches = create_image_dataset(
            imagesize=image_size,
            samples=1000,
            overlaps=100
        )
        
        # Test difference method.
        start_time = time.time()
        dmatches = diff_duplicate_search(images)
        data['diff_search_time'] = time.time() - start_time
        data['diff_search_validity'] = validate(matches, dmatches)

        # Test hash method. .
        start_time = time.time()
        dmatches = hash_duplicate_search(images)
        data['hash_search_time'] = time.time() - start_time
        data['hash_search_validity'] = validate(matches, dmatches)

        print(data)
        studies['image_size'].append(data)

    
    with open('hashes_study.json', 'w') as out:
        json.dump(studies, out, indent=4)
