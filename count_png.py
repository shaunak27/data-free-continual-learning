# domainnet dataset is present at data/domainnet and is divided into 6 domains with folder structure as follows:
# data/domainnet/clipart/category_name/*.jpg or *.png
#count the number of png files in each category_name folder and store the count in a csv file
import os
import csv
from tqdm import tqdm

def count_png_files(source, destination):
    print('Counting png files')
    with open(destination, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['category_name', 'count'])
        for category_name in tqdm(os.listdir(source), desc='Counting png files', total=len(os.listdir(source))):
            count = 0
            if not os.path.isdir(os.path.join(source, category_name)):
                continue
            for file_name in os.listdir(os.path.join(source, category_name)):
                if file_name.endswith('.png'):
                    count += 1
            writer.writerow([category_name, count])

def main():
    for domain_name in os.listdir('data/domainnet'):
        if not os.path.isdir(os.path.join('data/domainnet', domain_name)):
            continue
        source = 'data/domainnet/' + domain_name
        destination = f'data/domainnet/{domain_name}/count_png.csv'
        count_png_files(source, destination)

if __name__ == '__main__':
    main()