import matplotlib.pyplot as plt
import numpy as np
import os


data = {}
folders = 0
count = 0 
for folder, _, files in os.walk('../../work_images'):
    if 'tesseract' not in folder and 'lists' not in folder and 'scripts' not in folder:
        folders += 1
        count += len(files)
        data[folder] = len(files)

print(folders, count)

folders = np.array(list(data.keys()))
images = np.array(list(data.values()))
indices = np.argsort(images)[::-1]
running = 0

total = []
for i in indices:
    running += images[i]
    print("{0} ({1}): {2}/{3} {4:6.4f}%".format(folders[i], images[i],
                                                running, count, 100 * float(running/count)))
    total.append(100 * float(running / count))


plt.figure(figsize=(12,5))
plt.plot(np.arange(1,len(total)+1), total)
plt.grid(alpha=0.2)
plt.xlabel('Classes Included (Largest to Smallest)')
plt.ylabel('Percent of Dataset')
plt.savefig('percent_of_dataset.png', bbox_inches='tight')
