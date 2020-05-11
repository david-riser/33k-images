import numpy as np
import imagehash
import time

from PIL import Image

def hash_duplicate_search(images1, images2):
    """ 

    Search the two image sets for common images.  This is 
    done by hashing the two images and using a sorted hashtable.

    """
    hashtable = {}
    matches = []
    for i, image in enumerate(images2):
        hashtable[imagehash.dhash(Image.fromarray(image))] = i

    for i, image in enumerate(images1):
        image_hash = imagehash.dhash(Image.fromarray(image))
        if image_hash in hashtable:
            matches.append((i, hashtable[image_hash]))

    return matches

def diff_duplicate_search(images1, images2):
    """ 
    
    Take a difference and see if two images are the same.

    """
    matches = []
    def check(img1, img2):
        arr1, arr2 = np.asarray(img1), np.asarray(img2)
        return np.sum((arr2 - arr1)**2) == 0

    for i, image1 in enumerate(images1):
        for j, image2 in enumerate(images2):
            if check(image1, image2):
                matches.append((i,j))

    return matches


def create_image_dataset(imagesize, samples1, samples2, overlaps):
    """ Create a testing set. """

    images1 = []
    images2 = []
    matches = []
    
    assert(overlaps < samples1 and overlaps < samples2)
    for _ in range(samples1 - overlaps):
        images1.append(np.random.randint(
            0, 255, size=imagesize, dtype=np.uint8
        ))
    for _ in range(samples2 - overlaps):
        images2.append(np.random.randint(
            0, 255, size=imagesize, dtype=np.uint8
        ))
    for _ in range(overlaps):
        image = np.random.randint(
            0, 255, size=imagesize, dtype=np.uint8
        )
        images1.append(image)
        images2.append(image)
        matches.append((len(images1) - 1,len(images2) - 1))

    return images1, images2, matches
        
if __name__ == "__main__":

    image_shape = (36, 36, 3)

    # Generate a testing set.
    images1, images2, matches = create_image_dataset(
        imagesize=image_shape,
        samples1=2000,
        samples2=4000,
        overlaps=12
    )
    print("True Matches: ", matches)

    timer_start = time.time()
    print("Hash Matches: ", hash_duplicate_search(images1, images2))
    print("Elapsed Time: {}".format(time.time() - timer_start))

    timer_start = time.time()
    print("Diff Matches: ", diff_duplicate_search(images1, images2))
    print("Elapsed Time: {}".format(time.time() - timer_start))
