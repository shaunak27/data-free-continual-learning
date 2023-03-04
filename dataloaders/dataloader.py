from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import pickle
import torch
import torch.utils.data as data
from .utils import download_url, check_integrity
import random
import torchvision.datasets as datasets
import json 
import time
#from models.zeroshot import imr_classnames
VAL_HOLD = 0.1
class iDataset(data.Dataset):
    
    def __init__(self, root,
                train=True, transform=None,
                download_flag=False, lab=True, swap_dset = None, 
                tasks=None, seed=-1, rand_split=False, validation=False, kfolds=5):

        # process rest of args
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.validation = validation
        self.seed = seed
        self.t = -1
        self.tasks = tasks
        self.download_flag = download_flag

        # load dataset
        self.load()
        self.num_classes = len(np.unique(self.targets))

        # remap labels to match task order
        c = 0
        self.class_mapping = {}
        self.class_mapping[-1] = -1
        for task in self.tasks:
            for k in task:
                self.class_mapping[k] = c
                c += 1
        # targets as numpy.array
        self.data = np.asarray(self.data)
        self.targets = np.asarray(self.targets)

        # if validation
        if self.validation:
            
            # shuffle
            state = np.random.get_state()
            np.random.seed(self.seed)
            randomize = np.random.permutation(len(self.targets))
            self.data = self.data[randomize]
            self.targets = self.targets[randomize]
            np.random.set_state(state)

            # sample
            num_data_per_fold = int(len(self.targets) / kfolds)
            start = 0
            stop = num_data_per_fold
            locs_train = []
            locs_val = []
            for f in range(kfolds):
                if self.seed == f:
                    locs_val.extend(np.arange(start,stop))
                else:
                    locs_train.extend(np.arange(start,stop))
                start += num_data_per_fold
                stop += num_data_per_fold

            # train set
            if self.train:
                self.archive = []
                for task in self.tasks:
                    locs = np.isin(self.targets[locs_train], task).nonzero()[0]
                    self.archive.append((self.data[locs_train][locs].copy(), self.targets[locs_train][locs].copy()))

            # val set
            else:
                self.archive = []
                for task in self.tasks:
                    locs = np.isin(self.targets[locs_val], task).nonzero()[0]
                    self.archive.append((self.data[locs_val][locs].copy(), self.targets[locs_val][locs].copy()))

        # else
        else:
            self.archive = []
            for task in self.tasks:
                locs = np.isin(self.targets, task).nonzero()[0]
                self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))

        if self.train:
            self.coreset = (np.zeros(0, dtype=self.data.dtype), np.zeros(0, dtype=self.targets.dtype))

    def __getitem__(self, index, simple = False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.class_mapping[target], self.t


    def load_dataset(self, t, train=True):
        
        if train:
            self.data, self.targets = self.archive[t] 
        else:
            self.data    = np.concatenate([self.archive[s][0] for s in range(t+1)], axis=0)
            self.targets = np.concatenate([self.archive[s][1] for s in range(t+1)], axis=0)
        self.t = t

        print(np.unique(self.targets))

    def append_coreset(self, only=False, interp=False):
        len_core = len(self.coreset[0])
        if self.train and (len_core > 0):
            if only:
                self.data, self.targets = self.coreset
            else:
                len_data = len(self.data)
                sample_ind = np.random.choice(len_core, len_data)
                self.data = np.concatenate([self.data, self.coreset[0][sample_ind]], axis=0)
                self.targets = np.concatenate([self.targets, self.coreset[1][sample_ind]], axis=0)

    def update_coreset(self, coreset_size, seen):
        num_data_per = coreset_size // len(seen)
        remainder = coreset_size % len(seen)
        data = []
        targets = []
        
        # random coreset management; latest classes take memory remainder
        # coreset selection without affecting RNG state
        state = np.random.get_state()
        np.random.seed(self.seed*10000+self.t)
        for k in reversed(seen):
            mapped_targets = [self.class_mapping[self.targets[i]] for i in range(len(self.targets))]
            locs = (mapped_targets == k).nonzero()[0]
            if (remainder > 0) and (len(locs) > num_data_per):
                num_data_k = num_data_per + 1
                remainder -= 1
            else:
                num_data_k = min(len(locs), num_data_per)
            locs_chosen = locs[np.random.choice(len(locs), num_data_k, replace=False)]
            data.append([self.data[loc] for loc in locs_chosen])
            targets.append([self.targets[loc] for loc in locs_chosen])
        self.coreset = (np.concatenate(list(reversed(data)), axis=0), np.concatenate(list(reversed(targets)), axis=0))
        np.random.set_state(state)

    def load(self):
        pass


    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class iIMAGENET_R(iDataset):
    
    base_folder = 'imagenet-r'
    im_size=224
    nch=3
    def load(self):
        self.data, self.targets = [], []
        images_path = os.path.join(self.root, self.base_folder)
        data_dict = get_data(images_path)
        y = 0
        for key in data_dict.keys():
            num_y = len(data_dict[key])
            self.data.extend([data_dict[key][i] for i in np.arange(0,num_y)])
            self.targets.extend([y for i in np.arange(0,num_y)])
            y += 1

        n_data = len(self.targets)
        index_sample = [i for i in range(n_data)]
        import random
        random.seed(0)
        random.shuffle(index_sample)
        if self.train or self.validation:
            index_sample = index_sample[:int(0.8*n_data)]
        else:
            index_sample = index_sample[int(0.8*n_data):]

        self.data = [self.data[i] for i in index_sample]
        self.targets = [self.targets[i] for i in index_sample]

    def __getitem__(self, index, simple = False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img_path, target = self.data[index], self.targets[index]
        img = jpg_image_to_array(img_path)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.class_mapping[target], self.t

    
    def parse_archives(self) -> None:
        if not check_integrity(os.path.join(self.root, META_FILE)):
            parse_devkit_archive(self.root)

        if not os.path.isdir(self.split_folder):
            if self.split == 'train':
                parse_train_archive(self.root)
            elif self.split == 'val':
                parse_val_archive(self.root)

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


class iDOMAIN_NET(iIMAGENET_R):
    base_folder = 'domainnet'
    im_size=224
    nch=3
    def load(self):
        self.data, self.targets = [], []
        images_path = os.path.join(self.root, self.base_folder)
        data_dict = get_data_deep(images_path)
        y = 0
        cwd = os.getcwd()
        path = os.path.join(cwd,'data/domainnet/anns/clipart_train.txt')
        f  = open(path)
        lines = f.readlines()
        seq = [i.split()[0].split('/')[1] for i in lines]  
        seen = set()
        seen_add = seen.add
        class_list = [x for x in seq if not (x in seen or seen_add(x))]
        for key in class_list:
            num_y = len(data_dict[key])
            self.data.extend([data_dict[key][i] for i in np.arange(0,num_y)])
            self.targets.extend([y for i in np.arange(0,num_y)])
            y += 1
        n_data = len(self.targets)
        index_sample = [i for i in range(n_data)]
        import random
        random.seed(0)
        random.shuffle(index_sample)
        if self.train or self.validation:
            index_sample = index_sample[:int(0.08*n_data)]
        else:
            index_sample = index_sample[int(0.08*n_data):int(0.1*n_data)]

        self.data = [self.data[i] for i in index_sample]
        self.targets = [self.targets[i] for i in index_sample]

class KD_Dataset(data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        query = torch.load(f"/home/shaunak/fed_prompt/data-free-continual-learning/data/kd_data/queries_3/z_{idx}.pt")
        prompt = torch.load(f"/home/shaunak/fed_prompt/data-free-continual-learning/data/kd_data/prompts_3/p_{idx}.pt")
        return query,prompt


def get_data(root_images):

    import glob
    files = glob.glob(root_images+'/*/*.*')
    data = {}
    for path in files:
        y = os.path.basename(os.path.dirname(path))
        if y in data:
            data[y].append(path)
        else:
            data[y] = [path]
    return data

def get_data_deep(root_images):

    import glob
    files = glob.glob(root_images+'/*/*/*.*')
    data = {}
    for path in files:
        y = os.path.basename(os.path.dirname(path))
        if y in data:
            data[y].append(path)
        else:
            data[y] = [path]
    return data

# def get_data_deep(root_images):

#     import glob
#     files = glob.glob(root_images+'/*/*/*.jpg')
#     data = {}
#     for path in files:
#         y = os.path.basename(os.path.dirname(os.path.dirname(path)))
#         if y in data:
#             data[y].append(path)
#         else:
#             data[y] = [path]
#     return data

def jpg_image_to_array(image_path):
    """
    Loads JPEG image into 3D Numpy array of shape 
    (width, height, channels)
    """
    with Image.open(image_path) as image:      
        image = image.convert('RGB')
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))                                   
    return im_arr