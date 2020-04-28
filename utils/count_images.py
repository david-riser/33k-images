import numpy as np
import os


data = {}
folders = 0
count = 0 
for folder, _, files in os.walk('../../work_images'):
    folders += 1
    count += len(files)
    data[folder] = len(files)

print(folders, count)

folders = np.array(list(data.keys()))
images = np.array(list(data.values()))
indices = np.argsort(images)[::-1]
running = 0

for i in indices[:80]:
    running += images[i]
    print("{0} ({1}): {2}/{3} {4:6.4f}%".format(folders[i], images[i],
                                      running, count, 100 * float(running/count)))
