import argparse
import os
import numpy as np

from PIL import Image

if __name__ == "__main__":
    
    example_image_path = "DUMP/2_DUMP_462a383b9e48765d0ee96e4d5a3cfb30b30ed586fd5e011c411ee0840c88702c-0.jpg"
    img1 = np.asarray(Image.open(example_image_path))
    
    for folder, b, files in os.walk('.'):
        for image in files:
            image_path = os.path.join(folder, image)
            ext = image_path.split('.')[-1]

            if image_path.count('.') < 3 and ext not in ['zip', '7z', 'py', 'py~', 'db', 'txt']:
                img2 = np.asarray(Image.open(image_path))
                
                if img1.shape != img2.shape:
                    continue
        
                if np.sum((img1 - img2)**2) < 1e-6:
                    print(image_path)

        
