import os
import sys
import argparse
import torch
import numpy as np
import random
from random import shuffle
from collections import OrderedDict
import dataloaders
from dataloaders.utils import *
from torch.utils.data import DataLoader
import learners
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class Trainer:

    def __init__(self, args, seed, metric_keys, save_keys):

        # process inputs
        self.seed = seed
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.vis_flag = args.vis_flag == 1
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size
        self.workers = args.workers

        # check if VAE normalization is needed
        if args.learner_type == 'dgr' or args.learner_type == 'dfkd':# or args.learner_type == "deep_inv_gen":
            self.dgr = True
        else:
            self.dgr = False

        # check if external replay data needs loading
        self.dataset_swap = args.dataset_swap
        if args.learner_name == 'KD_EXT' or args.learner_name == 'KD_EXT_HARD' or args.learner_name == 'KD_EXT_GDNODE' or args.learner_name == 'KD_EXT_GD':
            self.ext_data = True
        else:
            self.ext_data = False
        
        # model load directory
        if args.load_model_dir is not None:
            self.model_first_dir = args.load_model_dir
        else:
            self.model_first_dir = args.log_dir
        self.model_top_dir = args.log_dir

        # select dataset
        self.grayscale_vis = False
        self.top_k = 1
        if args.dataset == 'MNIST':
            Dataset = dataloaders.iMNIST
            num_classes = 10
            self.grayscale_vis = True
            self.dataset_size = [28,28,1]
        elif args.dataset == 'CIFAR10':
            Dataset = dataloaders.iCIFAR10
            num_classes = 10
            self.dataset_size = [32,32,3]
        elif args.dataset == 'CIFAR100':
            Dataset = dataloaders.iCIFAR100
            num_classes = 100
            self.dataset_size = [32,32,3]
        elif args.dataset == 'ImageNet32':
            Dataset = dataloaders.iIMAGENETs
            num_classes = 100
            self.dataset_size = [32,32,3]
        elif args.dataset == 'ImageNet84':
            Dataset = dataloaders.iIMAGENETs
            num_classes = 100
            self.dataset_size = [84,84,3]
        elif args.dataset == 'ImageNet':
            Dataset = dataloaders.iIMAGENET
            num_classes = 1000
            self.dataset_size = [224,224,3]
            self.top_k = 5
        elif args.dataset == 'TinyImageNet':
            Dataset = dataloaders.iTinyIMNET
            num_classes = 200
            self.dataset_size = [64,64,3]
        else:
            raise ValueError('Dataset not implemented!')

        # load tasks
        class_order = np.arange(num_classes).tolist()
        class_order_logits = np.arange(num_classes).tolist()
        if self.seed > 0 and args.rand_split:
            print('=============================================')
            print('Shuffling....')
            print('pre-shuffle:' + str(class_order))
            if args.dataset == 'ImageNet':
                np.random.seed(1993)
                np.random.shuffle(class_order)
            else:
                random.seed(self.seed)
                random.shuffle(class_order)
            print('post-shuffle:' + str(class_order))
            print('=============================================')
        self.tasks = []
        self.tasks_logits = []
        p = 0
        while p < num_classes and (args.max_task == -1 or len(self.tasks) < args.max_task):
            inc = args.other_split_size if p > 0 else args.first_split_size
            self.tasks.append(class_order[p:p+inc])
            self.tasks_logits.append(class_order_logits[p:p+inc])
            p += inc
        self.num_tasks = len(self.tasks)
        self.task_names = [str(i+1) for i in range(self.num_tasks)]

        # number of tasks to perform
        if args.max_task > 0:
            self.max_task = min(args.max_task, len(self.task_names))
        else:
            self.max_task = len(self.task_names)

        # select external dataset (if needed)
        if self.ext_data or self.dataset_swap:
            if args.dataset_ext == 'SAME':
                Dataset_ext  = Dataset
                tasks_ext = [np.arange(num_classes).tolist() for t in range(self.num_tasks)]
            elif args.dataset_ext == 'MNIST':
                Dataset_ext  = dataloaders.iMNIST
                tasks_ext = [np.arange(10).tolist() for t in range(self.num_tasks)]
            elif args.dataset_ext  == 'CIFAR10':
                Dataset_ext  = dataloaders.iCIFAR10
                tasks_ext = [np.arange(10).tolist() for t in range(self.num_tasks)]
            elif args.dataset_ext  == 'ImageNet':
                Dataset_ext  = dataloaders.iIMAGENET
                tasks_ext = [np.arange(100).tolist() for t in range(self.num_tasks)]
            elif args.dataset_ext  == 'CIFAR100':
                Dataset_ext  = dataloaders.iCIFAR100
                tasks_ext = [np.arange(100).tolist() for t in range(self.num_tasks)]
            elif args.dataset_ext  == 'KMNIST':
                Dataset_ext  = dataloaders.iKMNIST
                tasks_ext = [np.arange(10).tolist() for t in range(self.num_tasks)]
            elif args.dataset_ext  == 'SVHN':
                Dataset_ext  = dataloaders.iSVHN
                tasks_ext = [np.arange(10).tolist() for t in range(self.num_tasks)]
            elif args.dataset_ext  == 'SVHNI':
                Dataset_ext  = dataloaders.iSVHNI
                tasks_ext = [np.arange(10).tolist() for t in range(self.num_tasks)]
            elif args.dataset_ext  == 'FakeData':
                Dataset_ext  = dataloaders.iFakeData
                tasks_ext = [np.arange(10).tolist() for t in range(self.num_tasks)]
            elif args.dataset_ext  == 'FashionMNIST':
                Dataset_ext  = dataloaders.iFashionMNIST
                tasks_ext = [np.arange(10).tolist() for t in range(self.num_tasks)] 
            else:
                raise ValueError('External dataset not implemented!')

        # datasets and dataloaders
        k = 1 # number of transforms per image
        train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, dgr=self.dgr, swap=self.dataset_swap)
        train_transform_hard = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, hard_aug=True, dgr=self.dgr, swap=self.dataset_swap)
        test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug, dgr=self.dgr, swap=self.dataset_swap)
        if self.ext_data:
            train_transform_ext = dataloaders.utils.get_transform(dataset=args.dataset_ext, phase='train', aug=args.train_aug, primary_dset = args.dataset, dgr=self.dgr, swap=self.dataset_swap)
            self.ext_dataset = Dataset_ext(args.dataroot, train=True, tasks=tasks_ext,
                                    download_flag=True, transform=TransformK(train_transform_ext, train_transform_ext, k), 
                                    seed=self.seed, rand_split=args.rand_split, validation=args.validation)
        
        # swap datasets if doing OOD CL!
        if not self.dataset_swap:
            self.train_dataset = Dataset(args.dataroot, train=True, lab = True, tasks=self.tasks,
                                download_flag=True, transform=TransformK(train_transform, train_transform, k), 
                                seed=self.seed, rand_split=args.rand_split, validation=args.validation)
            self.test_dataset  = Dataset(args.dataroot, train=False, tasks=self.tasks,
                                    download_flag=False, transform=test_transform, 
                                    seed=self.seed, rand_split=args.rand_split, validation=args.validation)
        else:
            train_transform_ext = dataloaders.utils.get_transform(dataset=args.dataset_ext, phase='train', aug=args.train_aug, primary_dset = args.dataset, dgr=self.dgr, swap=self.dataset_swap)
            self.ext_dataset = Dataset_ext(args.dataroot, train=True, tasks=self.tasks,
                                    download_flag=True, transform=TransformK(train_transform_ext, train_transform_ext, k), 
                                    seed=self.seed, rand_split=args.rand_split, validation=args.validation)
            self.train_dataset = Dataset(args.dataroot, train=True, lab = True, tasks=self.tasks,
                                download_flag=True, transform=TransformK(train_transform, train_transform, k), 
                                seed=self.seed, rand_split=args.rand_split, validation=args.validation, swap_dset = self.ext_dataset)
            test_transform_ext  = dataloaders.utils.get_transform(dataset=args.dataset_ext, phase='ext', aug=args.train_aug, primary_dset = args.dataset, dgr=self.dgr, swap=self.dataset_swap)                   
            ext_dataset_test = Dataset_ext(args.dataroot, train=False, tasks=tasks_ext,
                                    download_flag=True, transform=test_transform_ext, 
                                    seed=self.seed, rand_split=args.rand_split, validation=args.validation)                    
            self.test_dataset  = Dataset(args.dataroot, train=False, tasks=self.tasks,
                                download_flag=False, transform=test_transform, 
                                seed=self.seed, rand_split=args.rand_split, validation=args.validation, swap_dset = ext_dataset_test)
        
        self.train_dataset.simple_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug, dgr=self.dgr, swap=self.dataset_swap)

        # get dataset stats
        if False:
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=8)
            mean = 0.
            std = 0.
            nb_samples = 0.
            for i, (input, target, task) in enumerate(train_loader):
                data = input[0]
                batch_samples = data.size(0)
                data = data.view(batch_samples, data.size(1), -1)
                mean += data.mean(2).sum(0)
                std += data.std(2).sum(0)
                nb_samples += batch_samples
                print(i)
                print(mean / nb_samples)
                print(std / nb_samples)
            mean /= nb_samples
            std /= nb_samples
            print(mean)
            print(std)
            print(done)

        # for oracle
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0

        # Prepare the self.learner (model)
        self.learner_config = {'num_classes': num_classes,
                        'lr': args.lr,
                        'debug_mode': args.debug_mode == 1,
                        'momentum': args.momentum,
                        'weight_decay': args.weight_decay,
                        'schedule': args.schedule,
                        'schedule_type': args.schedule_type,
                        'model_type': args.model_type,
                        'model_name': args.model_name,
                        'gen_model_type': args.gen_model_type,
                        'gen_model_name': args.gen_model_name,
                        'optimizer': args.optimizer,
                        'gpuid': args.gpuid,
                        'memory': args.memory,
                        'temp': args.temp,
                        'out_dim': num_classes,
                        'overwrite': args.overwrite == 1,
                        'mu': args.mu,
                        'beta': args.beta,
                        'eps': args.eps,
                        'DW': args.DW,
                        'KD': args.KD,
                        'batch_size': args.batch_size,
                        'stat_layers': args.stat_layers,
                        'power_iters': args.power_iters,
                        'deep_inv_params': args.deep_inv_params,
                        'refresh_iters': args.refresh_iters,
                        'upper_bound_flag': args.upper_bound_flag,
                        'playground_flag': args.playground_flag,
                        'tasks': self.tasks_logits,
                        'top_k': self.top_k,
                        'block_size': args.block_size,
                        'layer_freeze': args.layer_freeze,
                        }
        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
        # self.learner.print_model()

    def train_vis(self, vis_dir, name, t_index, pre=False, embedding=None):
        
        self.test_dataset.load_dataset(self.num_tasks-1, train=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)

        if self.ext_data:
            if self.grayscale_vis: plt.rc('image', cmap='gray')
            test_loader_ext = DataLoader(self.ext_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
            self.learner.data_visualization(test_loader, test_loader_ext, vis_dir, name, t_index)
        else:
            if self.grayscale_vis: plt.rc('image', cmap='gray')
            self.learner.data_visualization(test_loader, vis_dir, name, t_index)

        # val data
        embedding = self.learner.visualization(test_loader, vis_dir, name, t_index, embedding)
        return embedding

    def task_eval(self, t_index, local=False, task='acc'):

        val_name = self.task_names[t_index]
        print('validation split name:', val_name)
        
        # eval
        self.test_dataset.load_dataset(t_index, train=True)
        test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
        if local:
            return self.learner.validation(test_loader, task_in = self.tasks_logits[t_index], task_metric=task, relabel_clusters = local)
        else:
            return self.learner.validation(test_loader, task_metric=task, relabel_clusters = local)

    def sim_eval(self, t_index, local=False, task='cka'):

        val_name = self.task_names[t_index]
        print('Feature Sim task: ', val_name)
        
        # eval
        if local:
            self.test_dataset.load_dataset(t_index - 1, train=False)
        else:
            self.test_dataset.load_dataset(t_index - 1, train=False)
        
        test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
        
        if task == 'cka':
            return self.learner.cka_eval(test_loader)
        else:
            return -1

    def train(self, avg_metrics):
    
        # temporary results saving
        temp_table = {}
        for mkey in self.metric_keys: temp_table[mkey] = []
        # HERE JAMES
        temp_dir = self.log_dir + '/temp/'
        # temp_dir = self.log_dir + '/'
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)

        # for each task
        for i in range(self.max_task):

            # save current task index
            self.current_t_index = i

            # save name for learner specific eval
            if self.vis_flag:
                vis_dir = self.log_dir + '/visualizations/task-'+self.task_names[i]+'/'
                if not os.path.exists(vis_dir): os.makedirs(vis_dir)
            else:
                vis_dir = None

            # set seeds
            random.seed(self.seed*100 + i)
            np.random.seed(self.seed*100 + i)
            torch.manual_seed(self.seed*100 + i)
            torch.cuda.manual_seed(self.seed*100 + i)

            # print name
            train_name = self.task_names[i]
            print('======================', train_name, '=======================')

            # load dataset for task
            task = self.tasks_logits[i]
            if self.oracle_flag:
                self.train_dataset.load_dataset(i, train=False)
                self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
                self.add_dim += len(task)
            else:
                self.train_dataset.load_dataset(i, train=True)
                self.add_dim = len(task)

            # HERE JAMES
            self.learner.max_task = self.max_task

            # add valid class to classifier
            self.learner.add_valid_output_dim(self.add_dim)

            # load dataset with memory
            self.train_dataset.append_coreset(only=False)

            # load dataloader
            if self.ext_data:
                train_loader_base = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(self.workers/2))
                train_loader_ext = DataLoader(self.ext_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=int(self.workers/2))
                train_loader = dataloaders.DoubleDataLoader(train_loader_base, train_loader_ext)
            else:
                train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(self.workers))

            # # T-sne plots
            # if self.vis_flag:
            #     self.train_vis(vis_dir, 'pre', i, pre=True)
            self.learner.debug_dir = vis_dir
            self.learner.debug_model_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'

            # learn
            self.test_dataset.load_dataset(i, train=False)
            test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
            if i == 0:
                model_save_dir = self.model_first_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            else:
                model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            avg_train_time = self.learner.learn_batch(train_loader, self.train_dataset, model_save_dir, test_loader)

            # save model
            self.learner.save_model(model_save_dir)

            # T-sne plots
            if self.vis_flag:
                self.train_vis(vis_dir, 'post', i)
            
            # evaluate acc
            acc_table = []
            acc_table_ssl = []
            self.reset_cluster_labels = True
            for j in range(i+1):
                acc_table.append(self.task_eval(j))
                acc_table_ssl.append(self.task_eval(j, task='aux_task'))
            temp_table['acc'].append(np.mean(np.asarray(acc_table)))
            temp_table['aux_task'].append(np.mean(np.asarray(acc_table_ssl)))
            temp_table['mem'].append(self.learner.count_memory(self.dataset_size))

            # save temporary results
            for mkey in self.metric_keys:
                save_file = temp_dir + mkey + '.csv'
                np.savetxt(save_file, np.asarray(temp_table[mkey]), delimiter=",", fmt='%.2f')  

            if avg_train_time is not None: avg_metrics['time']['global'][i] = avg_train_time
            avg_metrics['mem']['global'][:] = self.learner.count_memory(self.dataset_size)

            # get other metrics
            if not self.oracle_flag:
                avg_metrics['plastic']['global'][i] = self.task_eval(i, local=True)
                til = 0
                for j in range(i+1):
                    til += self.task_eval(j, local=True)
                avg_metrics['til']['global'][i] = til / (i+1)
                if i > 0 and self.vis_flag: avg_metrics['cka']['global'][i] = self.sim_eval(i, local=False)

            # HERE JAMES
            # print(apple)
        return avg_metrics 
    
    def summarize_acc(self, acc_dict, acc_table, acc_table_pt):

        # unpack dictionary
        avg_acc_all = acc_dict['global']
        avg_acc_pt = acc_dict['pt']
        avg_acc_pt_local = acc_dict['pt-local']

        # Calculate average performance across self.tasks
        # Customize this part for a different performance metric
        avg_acc_history = [0] * self.max_task
        for i in range(self.max_task):
            train_name = self.task_names[i]
            cls_acc_sum = 0
            for j in range(i+1):
                val_name = self.task_names[j]
                cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_pt[j,i,self.seed] = acc_table[val_name][train_name]
                avg_acc_pt_local[j,i,self.seed] = acc_table_pt[val_name][train_name]
            avg_acc_history[i] = cls_acc_sum / (i + 1)

        # Gather the final avg accuracy
        avg_acc_all[:,self.seed] = avg_acc_history

        # repack dictionary and return
        return {'global': avg_acc_all,'pt': avg_acc_pt,'pt-local': avg_acc_pt_local}

    def evaluate(self, avg_metrics):

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

        # store results
        metric_table = {}
        metric_table_local = {}
        for mkey in self.metric_keys:
            metric_table[mkey] = {}
            metric_table_local[mkey] = {}

        for i in range(self.max_task):

            # load model
            
            if i == 0:
                model_save_dir = self.model_first_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            else:
                model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            self.learner.task_count = i 
            self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
            self.learner.pre_steps()
            self.learner.load_model(model_save_dir)

            # evaluate acc
            metric_table['acc'][self.task_names[i]] = OrderedDict()
            metric_table_local['acc'][self.task_names[i]] = OrderedDict()
            self.reset_cluster_labels = True
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table['acc'][val_name][self.task_names[i]] = self.task_eval(j)
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table_local['acc'][val_name][self.task_names[i]] = self.task_eval(j, local=True)

            # evaluate aux_task
            metric_table['aux_task'][self.task_names[i]] = OrderedDict()
            metric_table_local['aux_task'][self.task_names[i]] = OrderedDict()
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table['aux_task'][val_name][self.task_names[i]] = self.task_eval(j, task='aux_task')
                metric_table_local['aux_task'][val_name][self.task_names[i]] = self.task_eval(j, local=True, task='aux_task')

        # # get final tsne embeddings
        # if self.vis_flag:

        #     # get new directory
        #     vis_dir = self.log_dir + '/visualizations/final/'
        #     if not os.path.exists(vis_dir): os.makedirs(vis_dir)

        #     # create embeddings
        #     tsne_embedding = None
        #     for i in reversed(range(self.max_task)):

        #         # load model
        #         if i == 0:
                #     model_save_dir = self.model_first_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
                # else:
                #     model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
        #         self.learner.load_model(model_save_dir)

        #         # evaluate tsne
        #         if tsne_embedding is None:
        #             tsne_embedding = self.train_vis(vis_dir, 'task-'+str(i+1), i)
        #         else:
        #             self.train_vis(vis_dir, 'task-'+str(i+1), i, embedding=tsne_embedding)

        # summarize metrics
        avg_metrics['acc'] = self.summarize_acc(avg_metrics['acc'], metric_table['acc'],  metric_table_local['acc'])
        avg_metrics['aux_task'] = self.summarize_acc(avg_metrics['aux_task'], metric_table['aux_task'],  metric_table_local['aux_task'])

        return avg_metrics