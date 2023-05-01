#in a folder named data/imagenet-r/ there are 200 folders, each with some image files
#this script counts the number of files with a .jpg extension in total and prints the result

import os
import sys

def count_files(path):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg"):
                count += 1
    print(count)

if __name__ == '__main__':
    path = sys.argv[1]
    count_files(path)