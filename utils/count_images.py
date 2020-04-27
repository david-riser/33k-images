import os

folders = 0
count = 0 
for folder, _, files in os.walk('../'):
    folders += 1
    count += len(files)


print(folders, count)
