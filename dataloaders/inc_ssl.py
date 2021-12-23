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

VAL_HOLD = 0.1
class DoubleDataLoader(object):
    def __init__(self, dset_a, dset_b):
        self.dset_a = dset_a
        self.dset_b = dset_b

        self.iter_a = iter(self.dset_a)
        self.iter_b = iter(self.dset_b)

    def __iter__(self):
        self.iter_a = iter(self.dset_a)
        return self

    def __len__(self):
        return len(self.dset_a)

    def __next__(self):
        
        # A
        x_a, y_a, task = next(self.iter_a)
        shuffle_idx = torch.randperm(len(y_a), device=y_a.device)
        x_a, y_a, task = [x_a[k][shuffle_idx] for k in range(len(x_a))], y_a[shuffle_idx], task[shuffle_idx]

        # unlabeled
        try:
            x_b, y_b, _ = next(self.iter_b)
        except:
            self.iter_b = iter(self.dset_b)
            x_b, y_b, _ = next(self.iter_b)
        shuffle_idx = torch.randperm(len(y_b), device=y_b.device)
        x_b, y_b = [x_b[k][shuffle_idx] for k in range(len(x_b))], y_b[shuffle_idx]

        return x_a, y_a, x_b, y_b, task

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
        self.lab = lab
        self.ic_dict = {}
        self.ic = False
        self.dw = True

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

        # # get dataset statistics
        # print(self.data.shape)
        # if len(self.data.shape) > 3:
        #     self.dset_mean, self.dset_std = np.mean(self.data/255.0, axis=(0,1,2)), np.std(self.data/255.0, axis=(0,1,2))
        # else:
        #     self.dset_mean, self.dset_std = np.mean(self.data/255.0), np.std(self.data/255.0)
        # print(self.dset_mean)
        # print(self.dset_std)
        # print(len(np.unique(self.targets)))
        # print(apple)

        # # get dataset statistics
        # data_loaded = []
        # for img_path in self.data[:100]:
        #     img = jpg_image_to_array(img_path)
        #     data_loaded.append(img)
        # data_loaded = np.asarray(data_loaded)
        # print(data_loaded.shape)
        # if len(data_loaded.shape) > 3:
        #     self.dset_mean, self.dset_std = np.mean(data_loaded/255.0, axis=(0,1,2)), np.std(data_loaded/255.0, axis=(0,1,2))
        # else:
        #     self.dset_mean, self.dset_std = np.mean(data_loaded/255.0), np.std(data_loaded/255.0)
        # print(self.dset_mean)
        # print(self.dset_std)
        # print(len(np.unique(self.targets)))
        # print(apple)

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

        # dataset swapping!
        if swap_dset is not None:
            swap_ind = int(len(self.archive)/2.0)

            # convert image to grayscale and resize if needed!
            for i in range(swap_ind, len(self.archive)):
                new_data, new_target = swap_dset.archive[i]

                # # grayscale
                # if len(new_data.shape) > len(self.data.shape):
                #     new_data = np.dot(new_data,[0.299, 0.587, 0.114])
                
                # # resize
                # if not (new_data.shape[1] == self.data.shape[1]):
                #     resize_data = []
                #     for old_data in new_data:
                #         resize_data.append(cv2.resize(old_data, dsize=(self.data.shape[1], self.data.shape[2]),interpolation=cv2.INTER_AREA))
                #     new_data = np.asarray(resize_data)


                # shift distribution
                for c in range(self.nch):
                    target_mean = np.mean(self.data[:,:,:,c])
                    target_variance = np.std(self.data[:,:,:,c])

                    new_data[:,:,:,c] = new_data[:,:,:,c] - np.mean(new_data[:,:,:,c])
                    new_data[:,:,:,c] = new_data[:,:,:,c] / np.std(new_data[:,:,:,c])
                    new_data[:,:,:,c] = new_data[:,:,:,c] * target_variance
                    new_data[:,:,:,c] = new_data[:,:,:,c] + target_mean

                self.archive[i] = (new_data.copy(), new_target.copy())


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
            if simple:
                img = self.simple_transform(img)
            else:
                img = self.transform(img)

        if self.lab:
            return img, self.class_mapping[target], self.t
        else:
            return img, -1, self.t

    def load_bic_dataset(self, post = False):

        if post:
            self.data, self.targets = self.data_a, self.targets_a   

        else:

            # get number of holdout
            len_coreset = len(self.coreset[0])
            self.coreset_idx_change = int(VAL_HOLD * len_coreset)

            # get number of holdout for training data (balanced)
            num_class_past = 0
            for i_ in range(self.t):
                num_class_past += len(self.tasks[i_])
            k_per_class = int(self.coreset_idx_change / num_class_past)
            
            num_k_hold = [0 for i_ in range(1000)]
            idx_a = []
            idx_b = []
            for i_ in range(len(self.data)):

                k = self.targets[i_]
                if num_k_hold[k] < k_per_class:
                    idx_a.append(i_)
                else:
                    idx_b.append(i_)
                num_k_hold[k] += 1
            
            self.data_a, self.targets_a = self.data[idx_a], self.targets[idx_a]
            self.data, self.targets = self.data[idx_b], self.targets[idx_b]

    def append_coreset_ic(self, post = False):
        if post:
            self.data = np.concatenate([self.data, self.coreset[0][self.coreset_sample_a_idx]], axis=0)
            self.targets = np.concatenate([self.targets, self.coreset[1][self.coreset_sample_a_idx]], axis=0)

        else:

            # get number of holdout for training data (balanced)
            num_class_past = 0
            for i_ in range(self.t):
                num_class_past += len(self.tasks[i_])
            k_per_class = int(self.coreset_idx_change / num_class_past)

            num_k_hold = [0 for i_ in range(1000)]
            idx_a = []
            idx_b = []
            for i_ in range(len(self.coreset[0])):

                k = self.coreset[1][i_]
                if num_k_hold[k] < k_per_class:
                    idx_a.append(i_)
                else:
                    idx_b.append(i_)
                num_k_hold[k] += 1

            self.coreset_sample_a_idx = idx_a
            # len_data = len(self.data)
            # sample_ind = np.random.choice(len(idx_b), len_data)
            # idx_b = idx_b[sample_ind]
            self.data = np.concatenate([self.data, self.coreset[0][idx_b]], axis=0)
            self.targets = np.concatenate([self.targets, self.coreset[1][idx_b]], axis=0)

    def update_coreset_ic(self, coreset_size, seen, teacher):
        self.ic = True
        num_data_per = coreset_size // len(seen)
        remainder = coreset_size % len(seen)
        data = []
        targets = []
        for k in reversed(seen):
            mapped_targets = [self.class_mapping[self.targets[i]] for i in range(len(self.targets))]
            locs = (mapped_targets == k).nonzero()[0]
            if (remainder > 0) and (len(locs) > num_data_per):
                num_data_k = num_data_per + 1
                remainder -= 1
            else:
                num_data_k = min(len(locs), num_data_per)

            if not (k in self.ic_dict):

                # get numpy array of all feature embeddings
                feat_emb = []
                for loc in locs:

                    # get data to gpu
                    x, y, t = self.__getitem__(loc, simple=True)
                    x = x.cuda()
                    x = x[None,:,:,:]

                    # get feat embedding
                    z = teacher.generate_scores_pen(x)
                    feat_emb.append(z.detach().cpu().tolist())

                feat_emb = np.asarray(feat_emb)

                # calculate mean
                k_mean = np.mean(feat_emb, axis = 0)
                k_dist = feat_emb - k_mean[:]
                k_dist = np.squeeze(k_dist)
                k_dist = np.linalg.norm(k_dist, axis = 1)

                locs_chosen = []
                locs_k_array = np.arange(len(feat_emb))
                feat_emb_cp = np.copy(feat_emb)
                for k_ in range(num_data_k):

                    if len(locs_k_array) == 0:
                        pass
                    elif len(locs_k_array) == 1:
                        # append to save array
                        p_idx = 0
                        locs_chosen.append(locs_k_array[p_idx])

                        # remove from calculate array
                        locs_k_array = np.delete(locs_k_array, p_idx, axis = 0)
                        feat_emb_cp = np.delete(feat_emb_cp, p_idx, axis = 0)
                    else:

                        # get idx of closest to mean
                        chosen_feat = feat_emb[locs_chosen]
                        new_sum = np.sum(chosen_feat, axis = 0)
                        term_b = (feat_emb_cp + new_sum) / (len(locs_chosen) + 1)
                        term_b = np.squeeze(term_b)
                        k_dist_loop = k_mean - term_b
                        k_dist_loop = np.squeeze(k_dist_loop)
                        k_dist_loop = np.linalg.norm(k_dist_loop, axis = 1)
                        p_idx = np.argmin(k_dist_loop)
                        
                        # append to save array
                        locs_chosen.append(locs_k_array[p_idx])

                        # remove from calculate array
                        locs_k_array = np.delete(locs_k_array, p_idx, axis = 0)
                        feat_emb_cp = np.delete(feat_emb_cp, p_idx, axis = 0)

                # partition data
                k_sorted = locs_chosen
                locs_chosen = locs[locs_chosen]
                self.ic_dict[k] = [[self.data[loc] for loc in locs_chosen], [self.targets[loc] for loc in locs_chosen]]

            data.append(self.ic_dict[k][0][:num_data_k])
            targets.append(self.ic_dict[k][1][:num_data_k])
            
        self.coreset = (np.concatenate(list(reversed(data)), axis=0), np.concatenate(list(reversed(targets)), axis=0))

    def update_coreset_ete(self, coreset_size, seen, teacher):
        self.ic = True
        num_data_per = coreset_size // len(seen)
        remainder = coreset_size % len(seen)
        data = []
        targets = []
        for k in reversed(seen):
            mapped_targets = [self.class_mapping[self.targets[i]] for i in range(len(self.targets))]
            locs = (mapped_targets == k).nonzero()[0]
            if (remainder > 0) and (len(locs) > num_data_per):
                num_data_k = num_data_per + 1
                remainder -= 1
            else:
                num_data_k = min(len(locs), num_data_per)

            if not (k in self.ic_dict):

                # get numpy array of all feature embeddings
                feat_emb = []
                for loc in locs:

                    # get data to gpu
                    x, y, t = self.__getitem__(loc, simple=True)
                    x = x.cuda()
                    x = x[None,:,:,:]

                    # get feat embedding
                    z = teacher.generate_scores_pen(x)
                    feat_emb.append(z.detach().cpu().tolist())

                feat_emb = np.asarray(feat_emb)

                # calculate mean
                k_mean = np.mean(feat_emb, axis = 0)
                k_dist = feat_emb - k_mean[:]
                k_dist = np.squeeze(k_dist)
                k_dist = np.linalg.norm(k_dist, axis = 1)

                locs_chosen = []
                locs_k_array = np.arange(len(feat_emb))
                feat_emb_cp = np.copy(feat_emb)
                for k_ in range(num_data_k):

                    if len(locs_k_array) == 0:
                        pass
                    elif len(locs_k_array) == 1:
                        # append to save array
                        p_idx = 0
                        locs_chosen.append(locs_k_array[p_idx])

                        # remove from calculate array
                        locs_k_array = np.delete(locs_k_array, p_idx, axis = 0)
                        feat_emb_cp = np.delete(feat_emb_cp, p_idx, axis = 0)
                    else:

                        # get idx of closest to mean
                        chosen_feat = feat_emb[locs_chosen]
                        k_dist_loop = k_mean - feat_emb_cp
                        k_dist_loop = np.squeeze(k_dist_loop)
                        k_dist_loop = np.linalg.norm(k_dist_loop, axis = 1)
                        p_idx = np.argmin(k_dist_loop)
                        
                        # append to save array
                        locs_chosen.append(locs_k_array[p_idx])

                        # remove from calculate array
                        locs_k_array = np.delete(locs_k_array, p_idx, axis = 0)
                        feat_emb_cp = np.delete(feat_emb_cp, p_idx, axis = 0)

                # partition data
                k_sorted = locs_chosen
                locs_chosen = locs[locs_chosen]
                self.ic_dict[k] = [[self.data[loc] for loc in locs_chosen], [self.targets[loc] for loc in locs_chosen]]

            data.append(self.ic_dict[k][0][:num_data_k])
            targets.append(self.ic_dict[k][1][:num_data_k])
            
        self.coreset = (np.concatenate(list(reversed(data)), axis=0), np.concatenate(list(reversed(targets)), axis=0))

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
                if self.ic:
                    self.data = np.concatenate([self.data, self.coreset[0]], axis=0)
                    self.targets = np.concatenate([self.targets, self.coreset[1]], axis=0)
                else:
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

class iMNIST(iDataset):
    """MNIST Dataset.
    This is a subclass of the iDataset Dataset.
    """
    im_size=28
    nch=1
    def load(self):

        if self.train or self.validation:
            mnist = datasets.MNIST(root=self.root, train=True, download=self.download_flag, transform=None)
        else:
            mnist = datasets.MNIST(root=self.root, train=False, download=self.download_flag, transform=None)

        self.data = mnist.data
        self.targets = mnist.targets      
        self.data = self.data.reshape(-1, 28, 28)

class iKMNIST(iDataset):
    """KMNIST Dataset.
    This is a subclass of the iDataset Dataset.
    """
    im_size=28
    nch=1
    def load(self):

        if self.train or self.validation:
            dset = datasets.KMNIST(root=self.root, train=True, download=self.download_flag, transform=None)
        else:
            dset = datasets.KMNIST(root=self.root, train=False, download=self.download_flag, transform=None)

        self.data = dset.data
        self.targets = dset.targets      
        self.data = self.data.reshape(-1, 28, 28)

class iSVHN(iDataset):
    """SVHN Dataset.
    This is a subclass of the iDataset Dataset.
    """
    im_size=32
    nch=3
    def load(self):

        if self.train or self.validation:
            dset = datasets.SVHN(root=self.root, split='train', download=self.download_flag, transform=None)
        else:
            dset = datasets.SVHN(root=self.root, split='test', download=self.download_flag, transform=None)

        self.data = dset.data
        self.targets = dset.labels  
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

class iSVHNI(iDataset):
    """SVHN Dataset.
    This is a subclass of the iDataset Dataset.
    """
    im_size=32
    nch=3
    def load(self):

        if self.train or self.validation:
            dset = datasets.SVHN(root=self.root, split='train', download=self.download_flag, transform=None)
        else:
            dset = datasets.SVHN(root=self.root, split='test', download=self.download_flag, transform=None)

        self.data = 1.0 - dset.data
        self.targets = dset.labels  
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

class iFakeData(iDataset):
    """FakeData Dataset.
    This is a subclass of the iDataset Dataset.
    """
    im_size=32
    nch=3
    def load(self):

        if self.train or self.validation:
            ndata=10000
        else:
            ndata=5000

        self.data = torch.Tensor(np.random.randint(255, size=(ndata,32,32,3))).type(torch.uint8)
        self.targets = torch.Tensor(np.random.randint(10, size=(ndata)))

class iFashionMNIST(iDataset):
    """FashionMNIST Dataset.
    This is a subclass of the iDataset Dataset.
    """
    im_size=28
    nch=1
    def load(self):

        if self.train or self.validation:
            dset = datasets.FashionMNIST(root=self.root, train=True, download=self.download_flag, transform=None)
        else:
            dset = datasets.FashionMNIST(root=self.root, train=False, download=self.download_flag, transform=None)

        self.data = dset.data
        self.targets = dset.targets      
        self.data = self.data.reshape(-1, 28, 28)

class iCIFAR10(iDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the iDataset Dataset.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    im_size=32
    nch=3

    def load(self):

        # download dataset
        if self.download_flag:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train or self.validation:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.course_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
                if 'coarse_labels' in entry:
                    self.course_targets.extend(entry['coarse_labels'])
                
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    

class iCIFAR100(iCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the iCIFAR10 Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    im_size=32
    nch=3




class iIMAGENET(iDataset):
    
    base_folder = 'ilsvrc'
    im_size=224
    nch=3
    def load(self):
        self.dw = False
        self.data, self.targets = [], []
        images_path = os.path.join(self.root, self.base_folder)
        if self.train or self.validation:
            images_path = os.path.join(images_path, 'train')
            data_dict = get_data(images_path)
        else:
            images_path = os.path.join(images_path, 'val')
            data_dict = get_data(images_path)
        y = 0
        for key in data_dict.keys():
            num_y = len(data_dict[key])
            self.data.extend([data_dict[key][i] for i in np.arange(0,num_y)])
            self.targets.extend([y for i in np.arange(0,num_y)])
            y += 1


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
            if simple:
                img = self.simple_transform(img)
            else:
                img = self.transform(img)

        if self.lab:
            return img, self.class_mapping[target], self.t
        else:
            return img, -1, self.t

    
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


# #wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1FyaXjtCPg1_33i30--oORtspzFwSAa30' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1FyaXjtCPg1_33i30--oORtspzFwSAa30" -O imagenet_train_500.h5 && rm -rf /tmp/cookies.txt
# #       
# #wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=18vYTxXpVB0lMrMitw3fVm2abGhWW37sN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18vYTxXpVB0lMrMitw3fVm2abGhWW37sN" -O imagenet_test_100.h5 && rm -rf /tmp/cookies.txt
# #
# # move to data/imagenet
# #
# # wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet 
# # --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1FyaXjtCPg1_33i30--oORtspzFwSAa30' 
# # -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1FyaXjtCPg1_33i30--oORtspzFwSAa30" -O imagenet_train_500.h5 && rm -rf /tmp/cookies.txt
# #
# # wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet 
# # --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=18vYTxXpVB0lMrMitw3fVm2abGhWW37sN' 
# # -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18vYTxXpVB0lMrMitw3fVm2abGhWW37sN" -O imagenet_test_100.h5 && rm -rf /tmp/cookies.txt
# #
class iIMAGENETs(iDataset):
    
    base_folder = 'images'
    url = "https://www.dropbox.com/s/ed1s1dgei9kxd2p/mini-imagenet.zip?dl=0"
    filename = "mini-imagenet-python.zip"

    im_size=32
    nch=3
    
    def load(self):
        self.data, self.targets = [], []
        images_path = os.path.join(self.root, self.base_folder)
        data_dict = get_data(images_path)
        y = 0
        for key in data_dict.keys():
            num_y = len(data_dict[key])
            pivot = int(num_y * (5/6))
            if self.train or self.validation:
                self.data.extend([data_dict[key][i] for i in np.arange(0,pivot)])
                self.targets.extend([y for i in np.arange(0,pivot)])
            else:
                self.data.extend([data_dict[key][i] for i in np.arange(pivot, num_y)])
                self.targets.extend([y for i in np.arange(pivot, num_y)])
            y += 1


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
            if simple:
                img = self.simple_transform(img)
            else:
                img = self.transform(img)

        if self.lab:
            return img, self.class_mapping[target], self.t
        else:
            return img, -1, self.t

def get_data(root_images):

    import glob
    files = glob.glob(root_images+'/*/*.JPEG')
    data = {}
    for path in files:
        y = os.path.basename(os.path.dirname(path))
        if y in data:
            data[y].append(path)
        else:
            data[y] = [path]
    return data
    

    # def old_data
        # get test data
        # import pickle
        # self.data = []
        # self.targets = []

        # train_in = open(self.root + "/mini-imagenet-cache-train.pkl", "rb")
        # train = pickle.load(train_in)
        # data_ = train['image_data'].tolist()
        # targets_ = [0 for j in range(len(data_))]
        # y = 0
        # for key in train['class_dict'].keys():
        #     push = int(len(train['class_dict'][key]) * self.train_test_pc)
        #     j = 0
        #     for index in train['class_dict'][key]:
        #         targets_[index] = y
        #         if self.train or self.validation:
        #             if j < push:
        #                 self.data.append(data_[index])
        #                 self.targets.append(targets_[index])
        #         else:
        #             if j >= push:
        #                 self.data.append(data_[index])
        #                 self.targets.append(targets_[index])
        #         j += 1
        #     y += 1

        # test_in = open(self.root + "/mini-imagenet-cache-val.pkl", "rb")
        # test = pickle.load(test_in)
        # data_ = test['image_data'].tolist()
        # targets_ = [0 for j in range(len(data_))]
        # for key in test['class_dict'].keys():
        #     push = int(len(test['class_dict'][key]) * self.train_test_pc)
        #     j = 0
        #     for index in test['class_dict'][key]:
        #         targets_[index] = y
        #         if self.train or self.validation:
        #             if j < push:
        #                 self.data.append(data_[index])
        #                 self.targets.append(targets_[index])
        #         else:
        #             if j >= push:
        #                 self.data.append(data_[index])
        #                 self.targets.append(targets_[index])
        #         j += 1
        #     y += 1
        
        # test_in = open(self.root + "/mini-imagenet-cache-test.pkl", "rb")
        # test = pickle.load(test_in)
        # data_ = test['image_data'].tolist()
        # targets_ = [0 for j in range(len(data_))]
        # for key in test['class_dict'].keys():
        #     push = int(len(test['class_dict'][key]) * self.train_test_pc)
        #     j = 0
        #     for index in test['class_dict'][key]:
        #         targets_[index] = y
        #         if self.train or self.validation:
        #             if j < push:
        #                 self.data.append(data_[index])
        #                 self.targets.append(targets_[index])
        #         else:
        #             if j >= push:
        #                 self.data.append(data_[index])
        #                 self.targets.append(targets_[index])
        #         j += 1
        #     y += 1

        # self.data = np.asarray(self.data)
        # self.targets = np.asarray(self.targets)
        # split = int(len(self.targets) * self.train_test_pc)
        # if self.train or self.validation:
        #     self.data = self.data[:split]
        #     self.targets = self.targets[:split]
        #     print('training')
        #     print(len(self.data))
        #     print(min(self.targets))
        #     print(max(self.targets))
        # else:
        #     self.data = self.data[split:]
        #     self.targets = self.targets[split:]
        #     print('val')
        #     print(len(self.data))

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