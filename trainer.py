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
        
        # model load directory
        self.model_top_dir = args.log_dir

        # select dataset
        self.grayscale_vis = False
        self.top_k = 1
        if args.dataset == 'CIFAR10':
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
        elif args.dataset == 'ImageNet_R':
            Dataset = dataloaders.iIMAGENET_R
            num_classes = 200
            self.dataset_size = [224,224,3]
            self.top_k = 1
        elif args.dataset == 'DomainNet':
            Dataset = dataloaders.iDOMAIN_NET
            num_classes = 345
            self.dataset_size = [224,224,3]
            self.top_k = 1
        elif args.dataset == 'TinyImageNet':
            Dataset = dataloaders.iTinyIMNET
            num_classes = 200
            self.dataset_size = [64,64,3]
        else:
            raise ValueError('Dataset not implemented!')

        # upper bound flag
        if args.upper_bound_flag:
            args.other_split_size = num_classes
            args.first_split_size = num_classes

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
            self.tasks.append(class_order[p:p+inc]) ## SHAUN : shuffled indices
            self.tasks_logits.append(class_order_logits[p:p+inc]) ## SHAUN : ordered indices
            p += inc
        self.num_tasks = len(self.tasks)
        self.task_names = [str(i+1) for i in range(self.num_tasks)]

        # number of tasks to perform
        if args.max_task > 0:
            self.max_task = min(args.max_task, len(self.task_names))
        else:
            self.max_task = len(self.task_names)

        # datasets and dataloaders
        k = 1 # number of transforms per image
        if args.model_name.startswith('vit'):
            resize_imnet = True
        else:
            resize_imnet = False
        train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, resize_imnet=resize_imnet)
        test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug, resize_imnet=resize_imnet)
        self.train_dataset = Dataset(args.dataroot, train=True, lab = True, tasks=self.tasks,
                            download_flag=True, transform=train_transform, 
                            seed=self.seed, rand_split=args.rand_split, validation=args.validation)
        self.test_dataset  = Dataset(args.dataroot, train=False, tasks=self.tasks,
                                download_flag=False, transform=test_transform, 
                                seed=self.seed, rand_split=args.rand_split, validation=args.validation)

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
                        'freeze_encoder' : args.freeze_encoder,
                        'freeze_last' : args.freeze_last,
                        'schedule_type': args.schedule_type,
                        'model_type': args.model_type,
                        'model_name': args.model_name,
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
                        'batch_size': args.batch_size,
                        'upper_bound_flag': args.upper_bound_flag,
                        'tasks': self.tasks_logits,
                        'top_k': self.top_k,
                        'template_style':args.template_style,
                        'prompt_param':[self.num_tasks,args.prompt_param] #SHAUN : Important step
                        }
        print(self.learner_config)
        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config) ## SHAUNAK : Initialize Learner 
        ##SHAUN : Jump back to run.py

    def train_vis(self, vis_dir, name, t_index, pre=False, embedding=None):
        
        self.test_dataset.load_dataset(self.num_tasks-1, train=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)

        if self.grayscale_vis: plt.rc('image', cmap='gray')
        self.learner.data_visualization(test_loader, vis_dir, name, t_index)

        # val data
        embedding = self.learner.visualization(test_loader, vis_dir, name, t_index, embedding)
        return embedding

    def task_eval(self, t_index, local=False, task='acc', all_tasks=False):

        val_name = self.task_names[t_index]
        print('validation split name:', val_name)
        
        # eval
        if all_tasks:
            self.test_dataset.load_dataset(t_index, train=False)
        else:
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
        temp_dir = self.log_dir + '/csv/'
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)

        # for each task
        for i in range(self.max_task): ## SHAUN : See learner config

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
                self.train_dataset.load_dataset(i, train=True) ##SHAUN : See dataloader.py
                self.add_dim = len(task)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # add valid class to classifier
            self.learner.add_valid_output_dim(self.add_dim)

            # load dataset with memory
            self.train_dataset.append_coreset(only=False)

            # load dataloader
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(self.workers))

            # # T-sne plots
            # if self.vis_flag:
            #     self.train_vis(vis_dir, 'pre', i, pre=True)
            self.learner.debug_dir = vis_dir
            self.learner.debug_model_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'

            # frequency table process
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_frequency()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_frequency()

            # learn
            self.test_dataset.load_dataset(i, train=False) ##SHAUN : loads all tasks seen till now
            test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            
            avg_train_time = self.learner.learn_batch(train_loader, self.train_dataset, model_save_dir, test_loader) ## SHAUN : Jump to LWF

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
            # temp_table['acc'].append(np.mean(np.asarray(acc_table)))
            temp_table['acc'].append(self.task_eval(i, all_tasks=True))
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

        return avg_metrics 
    
    def summarize_acc(self, acc_dict, acc_table, acc_table_pt, final_acc):

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
        # avg_acc_all[:,self.seed] = avg_acc_history
        avg_acc_all[:,self.seed] = np.asarray(final_acc)

        # repack dictionary and return
        return {'global': avg_acc_all,'pt': avg_acc_pt,'pt-local': avg_acc_pt_local}

    def evaluate(self, avg_metrics):

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

        # store results
        metric_table = {}
        metric_table_local = {}
        final_acc = []
        for mkey in self.metric_keys:
            metric_table[mkey] = {}
            metric_table_local[mkey] = {}

        for i in range(self.max_task):

            # load model
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            self.learner.task_count = i 
            self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
            self.learner.pre_steps()
            self.learner.load_model(model_save_dir)

            # frequency table process
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_frequency()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_frequency()

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

            final_acc.append(self.task_eval(i, all_tasks=True))

        # summarize metrics
        avg_metrics['acc'] = self.summarize_acc(avg_metrics['acc'], metric_table['acc'],  metric_table_local['acc'], final_acc)

        return avg_metrics
    
    def evaluate_zs(self, avg_metrics):

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

        # store results
        metric_table = {}
        metric_table_local = {}
        final_acc = []
        for mkey in self.metric_keys:
            metric_table[mkey] = {}
            metric_table_local[mkey] = {}

        for i in range(self.max_task):

            self.learner.task_count = i 
            self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
            self.learner.pre_steps()

            # frequency table process
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_frequency()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_frequency()

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

            final_acc.append(self.task_eval(i, all_tasks=True))

        # summarize metrics
        avg_metrics['acc'] = self.summarize_acc(avg_metrics['acc'], metric_table['acc'],  metric_table_local['acc'], final_acc)

        return avg_metrics