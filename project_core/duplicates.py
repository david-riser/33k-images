import numpy as np
import PIL

from imagehash import dhash
from PIL import Image


def get_image_loader(image):
    """ """

    if isinstance(image, np.ndarray):
        def loader(img):
            return Image.fromarray(img)
    elif isinstance(image, str):
        def loader(img):
            return Image.open(img)
    elif isinstance(image, PIL.Image.Image):
        def loader(img):
            return img
    else:
        raise ValueError('List must be PIL.Image, numpy.ndarray, or str (path) got {}.'.format(
            type(image)
        ))
        
    return loader

def get_numpy_loader(image):
    """ """

    if isinstance(image, np.ndarray):
        def loader(img):
            return img
    elif isinstance(image, str):
        def loader(img):
            return np.asarray(Image.open(img))
    elif isinstance(image, PIL.Image.Image):
        def loader(img):
            return np.asarray(img)
    else:
        raise ValueError('List must be PIL.Image, numpy.ndarray, or str (path) got {}.'.format(
            type(image)
        ))
        
    return loader

def hash_search(images):
    """ 

    A method to search a list of images for duplicates.

    The method compares images indirectly in hash form.  
    There can be collisions in the hash space, which means
    that the results can mis-identify matches.  This 
    algorithm is fast O(n).

    :param images: A list of strings that specify the path
    to an image.  This can also be a list of PIL images or
    numpy ararys.

    :returns matches: A list of lists that groups matching
    images together.

    """
 
    hashmap = {}
    matches = {}

    # Depending on the type of input, get a function
    # that loads a PIL image.
    loader = get_image_loader(images[0])
    
    # Search the space linearly, finding all
    # matches and collisions.
    for i, image in enumerate(images):

        try:
            hashed = dhash(loader(image))

            # This image is new, we have not
            # seen it before.
            if hashed not in hashmap:
                hashmap[hashed] = i

            # This image is already in our hashmap.
            # It is a duplicate or a collision.
            else:
                if hashed in matches:
                    matches[hashed].append(i)
                else:
                    matches[hashed] = [hashmap[hashed], i]

        except Exception as e:
            print('[INFO] Trouble loading {}. {}'.format(
                image, e
            ))

        
    # Usually the above is fine, but there is the chance
    # of a hash collision matching two images that are
    # not the same.
    #
    # Now we have to directly compare images to ensure
    # that they really do match.
    collisions = 0 
    true_matches = []
    for hash_, matches in matches.items():
        diff_search_matches = diff_search(
            [images[m] for m in matches]
        )

        returned_images = 0
        for match in diff_search_matches:
            true_matches.append([matches[m] for m in match])
            returned_images += len(match)
            
        collisions += len(matches) - returned_images

    print('[INFO] {} collisions resolved.'.format(collisions))
    
    return true_matches


def diff_search(images):
    """ 

    A simple but slow O(n**2) way to compare images directly
    and look for duplicates.

    """

    loader = get_numpy_loader(images[0])

    cache = {}
    matches = {}

    for i in range(len(images)):

        if i not in cache:
            cache[i] = loader(images[i])
            
        for j in range(i + 1, len(images)):
            if j not in cache:
                cache[j] = loader(images[j])
                
                if np.sum((cache[i] - cache[j])**2) == 0:
                    if i in matches:
                        matches[i].append(j)
                    else:
                        matches[i] = [i,j]

    return list(matches.values())
