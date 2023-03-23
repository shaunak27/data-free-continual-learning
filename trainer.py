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
import torchvision.datasets as datasets
import learners
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import copy
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
        self.load_model_from = args.load_model_from
        # select dataset
        self.grayscale_vis = False
        self.top_k = 1
        if args.dataset == 'ImageNet32':
            Dataset = dataloaders.iIMAGENETs
            num_classes = 100
            self.dataset_size = [32,32,3]
        elif args.dataset == 'CIFAR10':
            TrainDataset = dataloaders.IMBALANCECIFAR10
            Dataset = dataloaders.iCIFAR10
            num_classes = 10
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
            TrainDataset = dataloaders.IMBALANCEINR
            num_classes = args.end_class - args.start_class
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
        class_order = np.arange(args.start_class,args.end_class).tolist()
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
        
        if args.dataset == 'CIFAR10' or args.dataset == 'ImageNet_R':
            self.train_dataset = TrainDataset(args.dataroot, train=True, tasks=self.tasks,
                                download_flag=True, transform=train_transform, 
                                seed=self.seed)
        else:
            self.train_dataset = Dataset(args.dataroot, train=True, lab = True, tasks=self.tasks,
                                download_flag=True, transform=train_transform, 
                                seed=self.seed, rand_split=args.rand_split, validation=args.validation)
        
        self.test_dataset  = Dataset(args.dataroot, train=False, tasks=self.tasks,
                                download_flag=False, transform=test_transform, 
                                seed=self.seed, rand_split=args.rand_split, validation=args.validation)
        # for oracle
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0
        if args.split is None:
            args.split = 0
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
                        'tasks_real': self.tasks,
                        'top_k': self.top_k,
                        'load_model_from' : self.load_model_from,
                        'template_style':args.template_style,
                        'prompt_param':[self.num_tasks,args.prompt_param], #SHAUN : Important step
                        'split':args.split  
                        }
        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config) ## SHAUNAK : Initialize Learner 
        ##SHAUN : Jump back to run.py


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
            return self.learner.validation(test_loader, task_in = self.tasks_logits[t_index], task_metric=task, relabel_clusters = local,t_idx = t_index)
        else:
            return self.learner.validation(test_loader, task_metric=task, relabel_clusters = local,t_idx=t_index)


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
            if self.load_model_from is not None:
                model_save_dir = self.load_model_from
            else:
                model_save_dir =  self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/' #'/home/shaunak/fed_prompt/data-free-continual-learning/model_kd_2class.pth' #
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            
            avg_train_time = self.learner.learn_batch(train_loader, self.train_dataset, model_save_dir, test_loader) ## SHAUN : Jump to LWF
            # temp_p = torch.randn_like(self.learner.model.state_dict()['module.prompt.e_p_0'],dtype=torch.float32)
            # temp_k = torch.randn_like(self.learner.model.state_dict()['module.prompt.e_k_0'],dtype=torch.float32)
            # # save model
            
            # self.learner.model.state_dict()['module.prompt.e_p_0'].data.copy_(temp_p)
            # self.learner.model.state_dict()['module.prompt.e_k_0'].data.copy_(temp_k)
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



    def latent_gen(self):

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
            # load model
        i = 0
        model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
        self.learner.task_count = i 
        self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
        self.learner.pre_steps()
        self.learner.load_model(model_save_dir)

        val_name = self.task_names[i]
        print('validation split name:', val_name)
        self.train_dataset.load_dataset(i, train=True)
        train_loader  = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
        
        # frequency table process
        if i > 0:
            try:
                if self.learner.model.module.prompt is not None:
                    self.learner.model.module.prompt.process_frequency()
            except:
                if self.learner.model.prompt is not None:
                    self.learner.model.prompt.process_frequency()

        
        self.learner.generate_kd_data(train_loader)
            
        return  

    def train_kd(self):

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
            # load model
        i = 0
        model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
        self.learner.task_count = i 
        self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
        self.learner.pre_steps()
        self.learner.load_model(model_save_dir)

        val_name = self.task_names[i]
        print('validation split name:', val_name)
        #train_loader  = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
        dataset = dataloaders.KD_Dataset()
        train_loader = DataLoader(dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.workers, drop_last=False)
        # frequency table process
        if i > 0:
            try:
                if self.learner.model.module.prompt is not None:
                    self.learner.model.module.prompt.process_frequency()
            except:
                if self.learner.model.prompt is not None:
                    self.learner.model.prompt.process_frequency()

        
        self.learner.train_prompts(train_loader)
            
        return
    
    def unify_classifiers(self):

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
        model_save_dir = '/nethome/shalbe3/fed_prompt/data-free-continual-learning/_outputs/l2p_singlelayer_vit_0to120/ImageNet_R/10-task/vit/l2p_single-layer/models/repeat-1/task-1/class.pth'
        self.learner.load_model(model_save_dir)
        client1_config = copy.deepcopy(self.learner_config)
        client2_config = copy.deepcopy(self.learner_config)
        client3_config = copy.deepcopy(self.learner_config)
        client1_config['out_dim'] = 100
        client2_config['out_dim'] = 30
        client3_config['out_dim'] = 60
        client1 = learners.__dict__[self.learner_type].__dict__[self.learner_name](client1_config)
        client2 = learners.__dict__[self.learner_type].__dict__[self.learner_name](client2_config)
        client3 = learners.__dict__[self.learner_type].__dict__[self.learner_name](client3_config)
        client1.load_model('/nethome/shalbe3/fed_prompt/data-free-continual-learning/_outputs/l2p_singlelayer_vit_0to100/ImageNet_R/10-task/vit/l2p_single-layer/models/repeat-1/task-1/class.pth')
        client2.load_model('/nethome/shalbe3/fed_prompt/data-free-continual-learning/_outputs/l2p_singlelayer_vit_50to80/ImageNet_R/10-task/vit/l2p_single-layer/models/repeat-1/task-1/class.pth')
        client3.load_model('/nethome/shalbe3/fed_prompt/data-free-continual-learning/_outputs/l2p_singlelayer_vit_60to120/ImageNet_R/10-task/vit/l2p_single-layer/models/repeat-1/task-1/class.pth')

        last_w = torch.zeros_like(self.learner.model.state_dict()['module.last.weight'],dtype=torch.float32)
        last_b = torch.zeros_like(self.learner.model.state_dict()['module.last.bias'],dtype=torch.float32)
        prompt = torch.zeros_like(self.learner.model.state_dict()['module.prompt.e_p_0'],dtype=torch.float32)
        key = torch.zeros_like(self.learner.model.state_dict()['module.prompt.e_k_0'],dtype=torch.float32)
        
        
        # prompt = 0.33*client1.model.state_dict()['module.prompt.e_p_0'].data + 0.33*client2.model.state_dict()['module.prompt.e_p_0'].data + 0.33*client3.model.state_dict()['module.prompt.e_p_0'].data
        # key = 0.33*client1.model.state_dict()['module.prompt.e_k_0'].data + 0.33*client2.model.state_dict()['module.prompt.e_k_0'].data + 0.33*client3.model.state_dict()['module.prompt.e_k_0'].data
        


        last_w[:50] += client1.model.state_dict()['module.last.weight'].data[:50]
        last_w[50:60] += 0.5*client1.model.state_dict()['module.last.weight'].data[50:60]
        last_w[60:80] += 0.33*client1.model.state_dict()['module.last.weight'].data[60:80]
        last_w[80:100] += 0.5*client1.model.state_dict()['module.last.weight'].data[80:100]
        last_w[50:60] += 0.5*client2.model.state_dict()['module.last.weight'].data[:10]
        last_w[60:80] += 0.33*client2.model.state_dict()['module.last.weight'].data[10:]
        last_w[60:80] += 0.33*client3.model.state_dict()['module.last.weight'].data[:20]
        last_w[80:100] += 0.5*client3.model.state_dict()['module.last.weight'].data[20:40]
        last_w[100:120] += client3.model.state_dict()['module.last.weight'].data[40:60]

        last_b[:50] += client1.model.state_dict()['module.last.bias'].data[:50]
        last_b[50:60] += 0.5*client1.model.state_dict()['module.last.bias'].data[50:60]
        last_b[60:80] += 0.33*client1.model.state_dict()['module.last.bias'].data[60:80]
        last_b[80:100] += 0.5*client1.model.state_dict()['module.last.bias'].data[80:100]
        last_b[50:60] += 0.5*client2.model.state_dict()['module.last.bias'].data[:10]
        last_b[60:80] += 0.33*client2.model.state_dict()['module.last.bias'].data[10:]
        last_b[60:80] += 0.33*client3.model.state_dict()['module.last.bias'].data[:20]
        last_b[80:100] += 0.5*client3.model.state_dict()['module.last.bias'].data[20:40]
        last_b[100:120] += client3.model.state_dict()['module.last.bias'].data[40:60]

        # last_w[:100] += 0.333*client1.model.state_dict()['module.last.weight'].data
        # last_w[50:80] += 0.333*client2.model.state_dict()['module.last.weight'].data
        # last_w[60:120] += 0.333*client3.model.state_dict()['module.last.weight'].data
        # last_b[:100] += 0.333*client1.model.state_dict()['module.last.bias'].data   #Equal Merging
        # last_b[50:80] += 0.333*client2.model.state_dict()['module.last.bias'].data
        # last_b[60:120] += 0.333*client3.model.state_dict()['module.last.bias'].data
        
        self.learner.model.state_dict()['module.last.weight'].data.copy_(last_w)
        self.learner.model.state_dict()['module.last.bias'].data.copy_(last_b)
        # self.learner.model.state_dict()['module.prompt.e_p_0'].data.copy_(prompt)
        # self.learner.model.state_dict()['module.prompt.e_k_0'].data.copy_(key)

        i = 0
        self.learner.task_count = i 
        self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
        self.learner.pre_steps()
        
        

        val_name = self.task_names[i]
        print('validation split name:', val_name)
        cwd = os.getcwd()
        path = os.path.join(cwd,'data/kd_data/queries_uni/')
        dataset = dataloaders.QPdata_multi(path)
        train_loader = DataLoader(dataset, batch_size=self.batch_size,shuffle=True,num_workers=self.workers, drop_last=False)
        # frequency table process
        if i > 0:
            try:
                if self.learner.model.module.prompt is not None:
                    self.learner.model.module.prompt.process_frequency()
            except:
                if self.learner.model.prompt is not None:
                    self.learner.model.prompt.process_frequency()

        #model_save_dir = '/nethome/shalbe3/fed_prompt/data-free-continual-learning/_outputs/onlylastlayer_vit_0to100/ImageNet_R/10-task/vit/models/repeat-1/task-1/class.pth'
        #self.learner.load_model(model_save_dir)
        self.learner.train_prompts(train_loader)
        #self.learner.load_model(self.load_model_from)
        #self.learner.train_classifiers(train_loader)
        return