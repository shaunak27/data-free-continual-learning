import os
import os.path
import hashlib
import errno
import torch
from torchvision import transforms
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from torchvision import transforms as T
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

dataset_stats = {
    'CIFAR10' : {'mean': (0.49139967861519607, 0.48215840839460783, 0.44653091444546567),
                 'std' : (0.2470322324632819, 0.24348512800005573, 0.26158784172796434),
                 'size' : 32},
    'ImageNet32': {'mean': (0.4037, 0.3823, 0.3432),
                 'std' : (0.2417, 0.2340, 0.2235),
                 'size' : 32},
    'ImageNet84': {'mean': (0.4399, 0.4184, 0.3772),
                 'std' : (0.2250, 0.2199, 0.2139),
                 'size' : 84},
    'ImageNet': {'mean': (0.485, 0.456, 0.406),
                 'std' : (0.229, 0.224, 0.225),
                 'size' : 224},   
    'TinyImageNet': {'mean': (0.4389, 0.4114, 0.3682),
                 'std' : (0.2402, 0.2350, 0.2268),
                 'size' : 64},   
    'ImageNet_R': {
                 'size' : 224}, 
    'DomainNet': {
                 'size' : 224},  
                }

# transformations
def get_transform(dataset='cifar100', phase='test', aug=True, resize_imnet=False):
    transform_list = []
    # get out size
    crop_size = dataset_stats[dataset]['size']

    # get mean and std
    dset_mean = (0.48145466, 0.4578275, 0.40821073)#(0.0,0.0,0.0) # dataset_stats[dataset]['mean']
    dset_std = (0.26862954, 0.26130258, 0.27577711) #(1.0,1.0,1.0) # dataset_stats[dataset]['std']
    if dataset == 'ImageNet32' or dataset == 'ImageNet84':
        transform_list.extend([
            transforms.Resize((crop_size,crop_size))
        ])

    if phase == 'train':
        transform_list.extend([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dset_mean, dset_std),
                            ])
    else:
        if dataset.startswith('ImageNet') or dataset == 'DomainNet':
            transform_list.extend([
                transforms.Resize((224,224),interpolation=Image.BICUBIC),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(dset_mean, dset_std),
                                ])
            print('applying updated transformations')
        else:
            transform_list.extend([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(dset_mean, dset_std),
                                ])


    return transforms.Compose(transform_list)

def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)