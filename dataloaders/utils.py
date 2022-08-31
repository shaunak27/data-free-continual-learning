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
    'MNIST'   : {'mean': (0.13066051707548254,),
                 'std' : (0.30810780244715075,),
                 'size' : 28},
    'FMNIST'  : {'mean': (0.28604063146254594,),
                 'std' : (0.35302426207299326,),
                 'size' : 28},
    'CIFAR10' : {'mean': (0.49139967861519607, 0.48215840839460783, 0.44653091444546567),
                 'std' : (0.2470322324632819, 0.24348512800005573, 0.26158784172796434),
                 'size' : 32},
    'CIFAR100': {'mean': (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                 'std' : (0.2673342858792409, 0.25643846291708816, 0.2761504713256834),
                 'size' : 32},
    'ImageNet32': {'mean': (0.4037, 0.3823, 0.3432),
                 'std' : (0.2417, 0.2340, 0.2235),
                 'size' : 32},
    'ImageNet84': {'mean': (0.4399, 0.4184, 0.3772),
                 'std' : (0.2250, 0.2199, 0.2139),
                 'size' : 84},
    'KMNIST'   : {'mean': (0.1917621473589439,),
                 'std' : (0.34834283034636876,),
                 'size' : 28},
    'SVHN'   : {'mean': (0.4376821,  0.4437697,  0.47280442),
                 'std' : (0.19803012, 0.20101562 ,0.19703614),
                 'size' : 32},
    'SVHNI'   : {'mean': (1.0-0.4376821,  1.0-0.4437697,  1.0-0.47280442),
                 'std' : (0.19803012, 0.20101562 ,0.19703614),
                 'size' : 32},             
    'FakeData'   : {'mean': (0.498, 0.498, 0.498),
                 'std' : (0.289, 0.289, 0.289),
                 'size' : 28},
    'FashionMNIST'   : {'mean': (0.2860405969887955,),
                 'std' : (0.35302424451492237,),
                 'size' : 28},  
    'ImageNet': {'mean': (0.485, 0.456, 0.406),
                 'std' : (0.229, 0.224, 0.225),
                 'size' : 224},   
    'TinyImageNet': {'mean': (0.4389, 0.4114, 0.3682),
                 'std' : (0.2402, 0.2350, 0.2268),
                 'size' : 64},     
                }


# k transormations 
class TransformK:
    def __init__(self, transform, transform_hard, k):
        self.transform = transform
        self.transform_hard = transform_hard
        self.k = k

    def __call__(self, inp):
        x = [self.transform(inp)]
        for _ in range(self.k-1): x.append(self.transform_hard(inp))
        return x

# transformations
def get_transform(dataset='cifar100', phase='test', aug=True, hard_aug=False, primary_dset=None, dgr=False, swap=False):
    transform_list = []

    # if externel...
    if dataset == 'SAME': dataset = primary_dset

    # get out size
    if primary_dset is not None:
        crop_size = dataset_stats[primary_dset]['size']
    else:
        crop_size = dataset_stats[dataset]['size']

    # get mean and std
    dset_mean = dataset_stats[dataset]['mean']
    dset_std = dataset_stats[dataset]['std']
    if dgr:
        if len(dset_mean) == 1:
            dset_mean = (0.0,)
            dset_std = (1.0,)
        else:
            dset_mean = (0.0,0.0,0.0)
            dset_std = (1.0,1.0,1.0)
    elif swap:
        if len(dset_mean) == 1:
            dset_mean = (0.5,)
            dset_std = (0.5,)
        else:
            dset_mean = (0.5,0.5,0.5)
            dset_std = (0.5,0.5,0.5)

    # if needing to emulate size of primary dataset
    if primary_dset is not None:
        size = (dataset_stats[primary_dset]['size'])
        transform_list.extend([
            transforms.Resize((size,size))
        ])

    if dataset == 'ImageNet32' or dataset == 'ImageNet84':
        transform_list.extend([
            transforms.Resize((crop_size,crop_size))
        ])

    # if needing to emulate chanels of primary dataset
    if primary_dset is not None and len(dset_mean) > 1 and len(dataset_stats[primary_dset]['mean']) == 1:
        transform_list.extend([
            transforms.Grayscale()
        ])
        dset_mean = ((dset_mean[0] + dset_mean[1] + dset_mean[2]) / 3.0,)
        dset_std = ((dset_std[0] + dset_std[1] + dset_std[2]) / 3.0,)

    simple_aug = ('MNIST' in dataset) or ('KMNIST' in dataset) or ('FashionMMNIST' in dataset) or ('FakeData' in dataset)
    if phase == 'train' and not simple_aug and aug:
        if hard_aug:
            transform_list.extend([
                transforms.ColorJitter(brightness=63/255, contrast=0.8),
                RandomAugment(),
                transforms.ToTensor(), \
                transforms.Normalize(dset_mean, dset_std),
                Cutout()
                                ])
        elif dataset == 'ImageNet':
            transform_list.extend([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(dset_mean, dset_std),
                                ])
        else:
            transform_list.extend([
                transforms.ColorJitter(brightness=63/255, contrast=0.8),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(crop_size, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(dset_mean, dset_std),
                                ])
    else:
        if dataset == 'ImageNet':
            transform_list.extend([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(dset_mean, dset_std),
                                ])
        else:
            transform_list.extend([
                    transforms.ToTensor(),
                    transforms.Normalize(dset_mean, dset_std),
                                    ])


    return transforms.Compose(transform_list)

class RandomAugment:
    """
    Random aggressive data augmentation transformer.
    """
    def __init__(self, N=2, M=9):
        """
        :param N: int, [1, #ops]. max number of operations
        :param M: int, [0, 9]. max magnitude of operations
        """
        self.operations = {
            'Identity': lambda img, magnitude: self.identity(img, magnitude),

            'ShearX': lambda img, magnitude: self.shear_x(img, magnitude),
            'ShearY': lambda img, magnitude: self.shear_y(img, magnitude),
            'TranslateX': lambda img, magnitude: self.translate_x(img, magnitude),
            'TranslateY': lambda img, magnitude: self.translate_y(img, magnitude),
            'Rotate': lambda img, magnitude: self.rotate(img, magnitude),
            'Mirror': lambda img, magnitude: self.mirror(img, magnitude),

            'AutoContrast': lambda img, magnitude: self.auto_contrast(img, magnitude),
            'Equalize': lambda img, magnitude: self.equalize(img, magnitude),
            'Solarize': lambda img, magnitude: self.solarize(img, magnitude),
            'Posterize': lambda img, magnitude: self.posterize(img, magnitude),
            'Invert': lambda img, magnitude: self.invert(img, magnitude),
            'Contrast': lambda img, magnitude: self.contrast(img, magnitude),
            'Color': lambda img, magnitude: self.color(img, magnitude),
            'Brightness': lambda img, magnitude: self.brightness(img, magnitude),
            'Sharpness': lambda img, magnitude: self.sharpness(img, magnitude)
        }

        self.N = np.clip(N, a_min=1, a_max=len(self.operations))
        self.M = np.clip(M, a_min=0, a_max=9)

    def identity(self, img, magnitude):
        return img

    def transform_matrix_offset_center(self, matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = offset_matrix @ matrix @ reset_matrix
        return transform_matrix

    def shear_x(self, img, magnitude):
        img = img.transpose(Image.TRANSPOSE)
        magnitudes = np.random.choice([-1.0, 1.0]) * np.linspace(0, 0.3, 11)
        transform_matrix = np.array([[1, random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]), 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])
        transform_matrix = self.transform_matrix_offset_center(transform_matrix, img.size[0], img.size[1])
        img = img.transform(img.size, Image.AFFINE, transform_matrix.flatten()[:6], Image.BICUBIC)
        img = img.transpose(Image.TRANSPOSE)
        return img

    def shear_y(self, img, magnitude):
        img = img.transpose(Image.TRANSPOSE)
        magnitudes = np.random.choice([-1.0, 1.0]) * np.linspace(0, 0.3, 11)
        transform_matrix = np.array([[1, 0, 0],
                                     [random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]), 1, 0],
                                     [0, 0, 1]])
        transform_matrix = self.transform_matrix_offset_center(transform_matrix, img.size[0], img.size[1])
        img = img.transform(img.size, Image.AFFINE, transform_matrix.flatten()[:6], Image.BICUBIC)
        img = img.transpose(Image.TRANSPOSE)
        return img

    def translate_x(self, img, magnitude):
        img = img.transpose(Image.TRANSPOSE)
        magnitudes = np.random.choice([-1.0, 1.0]) * np.linspace(0, 0.3, 11)
        transform_matrix = np.array([[1, 0, 0],
                                     [0, 1, img.size[1]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])],
                                     [0, 0, 1]])
        transform_matrix = self.transform_matrix_offset_center(transform_matrix, img.size[0], img.size[1])
        img = img.transform(img.size, Image.AFFINE, transform_matrix.flatten()[:6], Image.BICUBIC)
        img = img.transpose(Image.TRANSPOSE)
        return img

    def translate_y(self, img, magnitude):
        img = img.transpose(Image.TRANSPOSE)
        magnitudes = np.random.choice([-1.0, 1.0]) * np.linspace(0, 0.3, 11)
        transform_matrix = np.array([[1, 0, img.size[0]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])],
                                     [0, 1, 0],
                                     [0, 0, 1]])
        transform_matrix = self.transform_matrix_offset_center(transform_matrix, img.size[0], img.size[1])
        img = img.transform(img.size, Image.AFFINE, transform_matrix.flatten()[:6], Image.BICUBIC)
        img = img.transpose(Image.TRANSPOSE)
        return img

    def rotate(self, img, magnitude):
        img = img.transpose(Image.TRANSPOSE)
        magnitudes = np.random.choice([-1.0, 1.0]) * np.linspace(0, 30, 11)
        theta = np.deg2rad(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
        transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                     [np.sin(theta), np.cos(theta), 0],
                                     [0, 0, 1]])
        transform_matrix = self.transform_matrix_offset_center(transform_matrix, img.size[0], img.size[1])
        img = img.transform(img.size, Image.AFFINE, transform_matrix.flatten()[:6], Image.BICUBIC)
        img = img.transpose(Image.TRANSPOSE)
        return img

    def mirror(self, img, magnitude):
        img = ImageOps.mirror(img)
        return img

    def auto_contrast(self, img, magnitude):
        img = ImageOps.autocontrast(img)
        return img

    def equalize(self, img, magnitude):
        img = ImageOps.equalize(img)
        return img

    def solarize(self, img, magnitude):
        magnitudes = np.linspace(0, 256, 11)
        img = ImageOps.solarize(img, random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
        return img

    def posterize(self, img, magnitude):
        magnitudes = np.linspace(4, 8, 11)
        img = ImageOps.posterize(img, int(round(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))))
        return img

    def invert(self, img, magnitude):
        img = ImageOps.invert(img)
        return img

    def contrast(self, img, magnitude):
        magnitudes = 1.0 + np.random.choice([-1.0, 1.0])*np.linspace(0.1, 0.9, 11)
        img = ImageEnhance.Contrast(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
        return img

    def color(self, img, magnitude):
        magnitudes = 1.0 + np.random.choice([-1.0, 1.0])*np.linspace(0.1, 0.9, 11)
        img = ImageEnhance.Color(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
        return img

    def brightness(self, img, magnitude):
        magnitudes = 1.0 + np.random.choice([-1.0, 1.0])*np.linspace(0.1, 0.9, 11)
        img = ImageEnhance.Brightness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
        return img

    def sharpness(self, img, magnitude):
        magnitudes = 1.0 + np.random.choice([-1.0, 1.0])*np.linspace(0.1, 0.9, 11)
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
        return img

    def __call__(self, img):
        ops = np.random.choice(list(self.operations.keys()), self.N)
        for op in ops:
            mag = random.randint(0, self.M)
            img = self.operations[op](img, mag)

        return img


class Cutout:
    def __init__(self, M=0.5, fill=0.0):
        self.M = np.clip(M, a_min=0.0, a_max=1.0)
        self.fill = fill

    def __call__(self, x):
        """
        Ref https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        """
        _, h, w = x.shape
        lh, lw = int(round(self.M * h)), int(round(self.M * w))

        cx, cy = np.random.randint(0, h), np.random.randint(0, w)
        x1 = np.clip(cx - lh // 2, 0, h)
        x2 = np.clip(cx + lh // 2, 0, h)
        y1 = np.clip(cy - lw // 2, 0, w)
        y2 = np.clip(cy + lw // 2, 0, w)
        x[:, x1: x2, y1: y2] = self.fill

        return x


class SimCLRAugment:
    def __call__(self, img):
        w, h = img.size

        # Random crop and resize
        transform = T.RandomResizedCrop((h, w), interpolation=Image.LANCZOS)
        img = transform(img)

        # Random flip
        transform = T.RandomHorizontalFlip(p=0.5)
        img = transform(img)

        # Color distortion
        rand_color_jitter = T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8)
        rand_gray = T.RandomGrayscale(p=0.2)
        transform = T.Compose([rand_color_jitter, rand_gray])
        img = transform(img)

        # Gaussian blur
        s = random.uniform(0.1, 2.0) * max(w, h) / 224
        img = img.filter(ImageFilter.GaussianBlur(s))

        return img

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