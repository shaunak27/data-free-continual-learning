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
import time
import copy
import wandb
os.environ["WANDB_SILENT"] = "true"
from models.zoo_old import Generator, DiversityLoss
from tqdm import tqdm
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
        self.n_clients = args.n_clients
        self.n_rounds = args.n_rounds
        self.kl = args.kl
        self.hepco = args.hepco
        self.imbalance = args.imbalance
        self.percent = args.percent
        # model load directory
        self.model_top_dir = args.log_dir
        self.cutoff = args.cutoff
        self.ignore_past_server = args.ignore_past_server
        # select dataset
        self.grayscale_vis = False
        self.top_k = 1
        wandb.init(project="hepcov3.0",name=args.wandb_name)
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
        elif args.dataset == 'IMBALANCEINR':
            Dataset = dataloaders.IMBALANCEINR
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
        self.num_classes = num_classes
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
        self.train_datasets = []
        self.test_datasets = []
        for i in range(self.n_clients):
            self.train_datasets.append(Dataset(args.dataroot, train=True, tasks=self.tasks,
                                download_flag=True, transform=train_transform, 
                                seed=self.seed, rand_split=args.rand_split, validation=args.validation,client_idx=i,imb_factor=self.imbalance,percent=self.percent))
            self.test_datasets.append(Dataset(args.dataroot, train=False, tasks=self.tasks,
                                    download_flag=False, transform=test_transform, 
                                    seed=self.seed, rand_split=args.rand_split, validation=args.validation,client_idx=i))

        self.server_test_dataset = Dataset(args.dataroot, train=False, tasks=self.tasks,
                                    download_flag=False, transform=test_transform, 
                                    seed=self.seed, rand_split=args.rand_split, validation=args.validation,client_idx=-1)

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

        self.server_config = {'num_classes': num_classes,
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
                        'template_style':args.template_style,
                        'prompt_param':[self.num_tasks,args.prompt_param] #SHAUN : Important step
                        }             
        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        #self.learners = [learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config) for i in range(self.n_clients)] ## SHAUNAK : Initialize Learner 
        self.server = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.server_config) ## SHAUNAK : Initialize Server
        ##SHAUN : Jump back to run.py
        # self.generator = Generator().cuda()
        # self.generator = torch.nn.DataParallel(self.generator, device_ids=[0,1], output_device=0)
        # device = torch.device(f"cuda:{self.server_config['gpuid']}")
        # self.generator = self.generator.to(device)
        self.prev_server = None

    def train_generator(self,round=0,task=0):
        n_epochs_new = 100
        batch_size = 32
        cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        kl_div = torch.nn.KLDivLoss(reduction='none')
        mse = torch.nn.MSELoss(reduction='none')
        diversity_loss = DiversityLoss(metric='l2')

        self.generator_new = Generator().cuda()
        self.generator_new = torch.nn.DataParallel(self.generator_new, device_ids=[0,1], output_device=0)

        self.generator_old = Generator().cuda()
        self.generator_old = torch.nn.DataParallel(self.generator_old, device_ids=[0,1], output_device=0)

        optimizer_new = torch.optim.Adam(self.generator_new.parameters(), lr=1e-4,betas=(0.9, 0.999),
            eps=1e-08, amsgrad=False)

        optimizer_old = torch.optim.Adam(self.generator_old.parameters(), lr=1e-4,betas=(0.9, 0.999),
            eps=1e-08, amsgrad=False)

        self.generator_new.train()
        self.generator_old.train()
        self.server.model.eval()
        if self.prev_server is not None:
            self.prev_server.model.eval()
        for learner in self.learners:
            learner.model.eval()

        pbar = tqdm(range(n_epochs_new),total=n_epochs_new)
        tasks_till_now = [j for sub in self.server.tasks_real[:self.current_t_index+1] for j in sub]

        for i in pbar:
            #ENSURE RANDOMNESS !!!
            labels = torch.from_numpy(np.random.choice(self.num_classes, batch_size, p=np.array(list(self.label_counts_round.values())) / np.array(list(self.label_counts_round.values())).sum())).cuda()
            eps = torch.randn(labels.shape[0], 32).cuda()
            z = self.generator_new(labels,eps)
            xent_loss = 0
            kl_loss = 0
            div_loss = 0
            for j in range(self.n_clients):
                ## ADD PROVISION IN CASE label counts sum is zero (mainly for prev tasks)
                xent_loss += ((torch.tensor(list(self.label_counts[j].values())) / torch.tensor(list(self.label_counts[j].values())).sum()).cuda()[labels] * cross_entropy(self.learners[j].model.forward(x=None,z=z)[:,tasks_till_now], labels)).sum()
                if self.kl:
                    kl_loss += ((torch.tensor(list(self.label_counts[j].values())) / torch.tensor(list(self.label_counts[j].values())).sum()).cuda()[labels][:,None] * kl_div(torch.log_softmax(self.server.model.forward(x=None,z=z)[:,tasks_till_now], dim=1),torch.softmax(self.learners[j].model.forward(x=None,z=z)[:,tasks_till_now], dim=1))).sum()
                else :
                    kl_loss += ((torch.tensor(list(self.label_counts[j].values())) / torch.tensor(list(self.label_counts[j].values())).sum()).cuda() * mse(torch.softmax(self.server.model.forward(x=None,z=z), dim=1)[:,tasks_till_now],torch.softmax(self.learners[j].model.forward(x=None,z=z)[:,tasks_till_now], dim=1))).sum()
                div_loss += diversity_loss(eps,z)
            
            total_loss = xent_loss - kl_loss + div_loss 
            
            total_loss.backward()
            optimizer_new.step()
            optimizer_new.zero_grad()
            pbar.set_description('Epoch: {}, Total Loss: {}, Xent Loss: {}, KL Loss: {}, Div Loss: {}'.format(i, total_loss, xent_loss, kl_loss, div_loss))
            #wandb.log({'xent_loss':xent_loss, 'kl_loss':kl_loss})
        
        if self.prev_server is not None:
            pbar = tqdm(range(n_epochs_new),total=n_epochs_new)
            xent_loss = 0
            kl_loss = 0
            div_loss = 0
            for i in pbar:
                #ENSURE RANDOMNESS !!!
                labels = torch.from_numpy(np.random.choice(self.num_classes, batch_size, p=np.array(list(self.label_counts_server_last_task.values())) / np.array(list(self.label_counts_server_last_task.values())).sum())).cuda()
                eps = torch.randn(labels.shape[0], 32).cuda()
                z = self.generator_old(labels,eps)
                
                div_loss = diversity_loss(eps,z)
                xent_loss = ((torch.tensor(list(self.label_counts_server_last_task.values())) / torch.tensor(list(self.label_counts_server_last_task.values())).sum()).cuda()[labels] * cross_entropy(self.prev_server.model.forward(x=None,z=z)[:,tasks_till_now], labels)).sum()
                kl_loss = ((torch.tensor(list(self.label_counts_server_last_task.values())) / torch.tensor(list(self.label_counts_server_last_task.values())).sum()).cuda()[labels][:,None] * kl_div(torch.log_softmax(self.server.model.forward(x=None,z=z)[:,tasks_till_now], dim=1),torch.softmax(self.prev_server.model.forward(x=None,z=z)[:,tasks_till_now], dim=1))).sum()
                
                total_loss = xent_loss - kl_loss + div_loss 
                
                total_loss.backward()
                optimizer_old.step()
                optimizer_old.zero_grad()
                pbar.set_description('Epoch: {}, Total Loss: {}, Xent Loss: {}, KL Loss: {}, Div Loss: {}'.format(i, total_loss, xent_loss, kl_loss, div_loss))
                
        self.server.model.train()
        if self.prev_server is not None:
            self.prev_server.model.train()
        for learner in self.learners:
            learner.model.train()
        
        #save generator

        # model_state = self.generator.state_dict()
        # for key in model_state.keys():  # Always save it to cpu
        #     model_state[key] = model_state[key].cpu()
        # model_save_dir = self.model_top_dir + '/generator/'
        # if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
        # print('=> Saving class model to:', os.path.join(model_save_dir,f'generator_{task}_{round}.pth'))
        # torch.save(model_state, os.path.join(model_save_dir,f'generator_{task}_{round}.pth'))
        # print('=> Save Done')
        #Now that the generator is trained, we can generate the latents and corresponding prompts and perform KD
        #Can have a similar class count weighted loss for prompt KD and classifier KD

    def knowledge_distillation(self):
        num_epochs = 200
        batch_size = 32
        old_batch_size = 4
        mse = torch.nn.MSELoss()
        cross_entropy = torch.nn.CrossEntropyLoss()
        optimizer1 = torch.optim.Adam(self.server.model.module.prompt.parameters(), lr=1e-4,betas=(0.9, 0.999),
            eps=1e-08, amsgrad=False)
        optimizer2 = torch.optim.Adam(self.server.model.module.last.parameters(), lr=1e-4,betas=(0.9, 0.999),
            eps=1e-08, amsgrad=False)
        
        self.generator_new.eval()
        self.generator_old.eval()
        if self.prev_server is not None:
            self.prev_server.model.eval()
        self.server.model.train()
        for learner in self.learners:
            learner.model.eval()

        pb1 = tqdm(range(num_epochs),total=num_epochs)
        pb2 = tqdm(range(num_epochs),total=num_epochs)
        tasks_till_now = [j for sub in self.server.tasks_real[:self.current_t_index+1] for j in sub]
        
        for it in pb1:
            mse_loss = 0
            with torch.no_grad():
                labels_new = torch.from_numpy(np.random.choice(self.num_classes, batch_size, p=np.array(list(self.label_counts_round.values())) / np.array(list(self.label_counts_round.values())).sum())).cuda()
                eps_new = torch.randn(labels_new.shape[0], 32).cuda() 
                z_new = self.generator_new(labels_new,eps_new)
            for j in range(self.n_clients):
                for layer in self.server.model.module.prompt.e_layers:
                    mse_loss += ((torch.tensor(list(self.label_counts[j].values())) / torch.tensor(list(self.label_counts[j].values())).sum()).cuda()[labels_new] * mse(self.server.model.module.prompt.forward(z_new,layer,None,train=True,hepco=True), self.learners[j].model.module.prompt.forward(z_new,layer,None,train=True,hepco=True))).sum() 
            
            if self.prev_server is not None:
                labels_old = torch.from_numpy(np.random.choice(self.num_classes, old_batch_size, p=np.array(list(self.label_counts_server_last_task.values())) / np.array(list(self.label_counts_server_last_task.values())).sum())).cuda()
                eps_old = torch.randn(labels_old.shape[0], 32).cuda()
                z_old = self.generator_old(labels_old,eps_old)
                for layer in self.server.model.module.prompt.e_layers:
                        mse_loss += ((torch.tensor(list(self.label_counts_server_last_task.values())) / torch.tensor(list(self.label_counts_server_last_task.values())).sum()).cuda()[labels_old] * mse(self.server.model.module.prompt.forward(z_old,layer,None,train=True,hepco=True), self.prev_server.model.module.prompt.forward(z_old,layer,None,train=True,hepco=True))).sum()

            optimizer1.zero_grad()
            mse_loss.backward()
            optimizer1.step()
            pb1.set_description('Epoch: {}, MSE Loss: {}'.format(it, mse_loss))
        xent_loss = 0
        for it in pb2: 
            with torch.no_grad():
                labels_new = torch.from_numpy(np.random.choice(self.num_classes, batch_size, p=np.array(list(self.label_counts_round.values())) / np.array(list(self.label_counts_round.values())).sum())).cuda()
                eps_new = torch.randn(labels_new.shape[0], 32).cuda() 
                z_new = self.generator_new(labels_new,eps_new)
                if self.prev_server is not None:
                    labels_old = torch.from_numpy(np.random.choice(self.num_classes, old_batch_size, p=np.array(list(self.label_counts_server_last_task.values())) / np.array(list(self.label_counts_server_last_task.values())).sum())).cuda()
                    eps_old = torch.randn(labels_old.shape[0], 32).cuda()
                    z_old = self.generator_old(labels_old,eps_old)
                    z_combined = torch.cat((z_new,z_old),dim=0)
                    labels_combined = torch.cat((labels_new,labels_old),dim=0)
                else:
                    z_combined = z_new
                    labels_combined = labels_new         
            xent_loss = cross_entropy(self.server.model(x=None,z=z_combined)[:,tasks_till_now], labels_combined)            
            optimizer2.zero_grad()
            xent_loss.backward()
            optimizer2.step()
            pb2.set_description('Epoch: {}, Xent Loss: {}'.format(it, xent_loss))

        #set clients and generator to train mode
        self.generator_new.train()
        self.generator_old.train()
        for learner in self.learners:
            learner.model.train()

    def communicate(self):
        with torch.no_grad():
            for key in self.server.model.state_dict().keys():
                if 'prompt' in key or 'last' in key: #retract : 'last in key
                    temp = torch.zeros_like(self.server.model.state_dict()[key],dtype=torch.float32)
                    for i in range(self.n_clients):
                        temp += (1/self.n_clients)*self.learners[i].model.state_dict()[key]
                    self.server.model.state_dict()[key].data.copy_(temp)
        print('Communication successful !')

    def train_vis(self, vis_dir, name, t_index, pre=False, embedding=None):
        
        self.test_dataset.load_dataset(self.num_tasks-1, train=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)

        if self.grayscale_vis: plt.rc('image', cmap='gray')
        self.learner.data_visualization(test_loader, vis_dir, name, t_index)

        # val data
        embedding = self.learner.visualization(test_loader, vis_dir, name, t_index, embedding)
        return embedding

    def task_eval(self, t_index, local=False, task='acc', all_tasks=False,client_idx=0):

        val_name = self.task_names[t_index]
        print('validation split name:', val_name)
        
        # eval
        if all_tasks:
            self.test_datasets[client_idx].load_dataset(t_index, train=False)
        else:
            self.test_datasets[client_idx].load_dataset(t_index, train=True)
        test_loader  = DataLoader(self.test_datasets[client_idx], batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
        if local:
            return self.learners[client_idx].validation(test_loader, task_in = self.tasks_logits[t_index], task_metric=task, relabel_clusters = local,t_idx = t_index)
        else:
            return self.learners[client_idx].validation(test_loader, task_metric=task, relabel_clusters = local,t_idx=t_index)

    def server_task_eval(self, t_index, local=False, task='acc', all_tasks=False):

        val_name = self.task_names[t_index]
        print('SERVER EVALUATION')
        print('validation split name:', val_name)
        
        # eval
        if all_tasks:
            self.server_test_dataset.load_dataset(t_index, train=False)
        else:
            self.server_test_dataset.load_dataset(t_index, train=True)
        test_loader  = DataLoader(self.server_test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
        if local:
            return self.server.validation(test_loader, task_in = self.tasks_logits[t_index], task_metric=task, relabel_clusters = local,t_idx = t_index)
        else:
            return self.server.validation(test_loader, task_metric=task, relabel_clusters = local,t_idx=t_index)

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
        temp_table = []
        temp_dir = []
        #for idx in range(self.n_clients*self.n_rounds):    #uncomment for learners
            # temporary results saving
            #temp_table.append({})
            #for mkey in self.metric_keys: temp_table[idx][mkey] = []
            #temp_dir.append(self.log_dir + f'_client_{idx}/csv/')
            #if not os.path.exists(temp_dir[idx]): os.makedirs(temp_dir[idx])

            #print('======================',f'Client {idx+1}' , '=======================')
            # for each task
        server_temp_table = {}
        for mkey in self.metric_keys: server_temp_table[mkey] = []
        server_temp_dir = self.log_dir + f'_server/csv/'
        if not os.path.exists(server_temp_dir): os.makedirs(server_temp_dir)

        #initialize a dictionary to store the counts of labels seen by each client
        
        #initialize a dictionary to store the counts of labels
        self.label_counts_server = {}
        for j in range(self.num_classes):
            self.label_counts_server[j] = 0

        #initialize a dictionary to store the counts of labels seen by server until last task
        self.label_counts_server_last_task = {}
        for j in range(self.num_classes):
            self.label_counts_server_last_task[j] = 0

        avg_acc = 0
        # for each task
        for i in range(self.max_task): ## SHAUN : See learner config
            random.seed(self.seed*100 + i)
            np.random.seed(self.seed*100 + i)
            torch.manual_seed(self.seed*100 + i)
            torch.cuda.manual_seed(self.seed*100 + i)
            train_name = self.task_names[i]
            task = self.tasks_logits[i]
            self.add_dim = len(task)

            # set task id for model (needed for prompting)
            try:
                self.server.model.module.task_id = i
            except:
                self.server.model.task_id = i

            # add valid class to classifier
            self.server.add_valid_output_dim(self.add_dim)
            self.server.task_count = i
 
            print('======================', train_name, '=======================')   
            for r in range(self.n_rounds):
                self.learners = [copy.deepcopy(self.server) for _ in range(self.n_clients)]
                self.label_counts = {}
                for x in range(self.n_clients):
                    self.label_counts[x] = {}
                    for p in range(self.num_classes):
                        self.label_counts[x][p] = 0

                self.label_counts_round = {}
                for jo in range(self.num_classes):
                    self.label_counts_round[jo] = 0

                for idx in range(self.n_clients):
                # save current task index
                    self.current_t_index = i 
                    # save name for learner specific eval
                    if self.vis_flag:
                        vis_dir = self.log_dir + '/visualizations/task-'+self.task_names[i]+'/'
                        if not os.path.exists(vis_dir): os.makedirs(vis_dir)
                    else:
                        vis_dir = None

                    # set seeds
                    print('======================',f'Client {idx+1 + self.n_clients*r}, Task {train_name}' , '=======================')

                    # load dataset for task
                    self.label_counts[idx] = self.train_datasets[idx].load_dataset(i, train=True, label_counts = self.label_counts[idx],seed=idx+1 + self.n_clients*r + self.n_rounds*i*self.n_clients,cutoff=self.cutoff) ##SHAUN : See dataloader.py
                    # load dataset with memory
                    self.train_datasets[idx].append_coreset(only=False)

                    # load dataloader
                    train_loader = DataLoader(self.train_datasets[idx], batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=int(self.workers))

                    # # T-sne plots
                    # if self.vis_flag:
                    #     self.train_vis(vis_dir, 'pre', i, pre=True)

                    # self.learners[idx].debug_dir = vis_dir #uncomment for learners
                    # self.learners[idx].debug_model_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'

                    # frequency table process
                    if i > 0:
                        try:
                            if self.learners[idx].model.module.prompt is not None:
                                self.learners[idx].model.module.prompt.process_frequency()
                        except:
                            if self.learners[idx].model.prompt is not None:
                                self.learners[idx].model.prompt.process_frequency()

                    # learn
                    self.test_datasets[idx].load_dataset(i, train=True) ##SHAUN : loads all tasks seen till now
                    test_loader  = DataLoader(self.test_datasets[idx], batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
                    
                    model_save_dir = self.model_top_dir + f'_client_{idx + self.n_clients*r}/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
                    if not os.path.exists(model_save_dir): os.makedirs(model_save_dir) #uncomment for learners
                    
                    server_model_save_dir = self.model_top_dir + f'_server_{r}/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
                    prev_server_model_save_dir = self.model_top_dir + f'_server_{self.n_rounds-1}/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i-1]+'/'
                    try:
                        self.server.load_model(server_model_save_dir)
                        if i > 0:
                            self.prev_server.load_model(prev_server_model_save_dir)
                    except:
                        avg_train_time = self.learners[idx].learn_batch(train_loader, self.train_datasets[idx], model_save_dir, test_loader) ## SHAUN : Jump to LWF

                    # save model
                    #self.learners[idx].save_model(model_save_dir) 

                    # T-sne plots
                    if self.vis_flag:
                        self.train_vis(vis_dir, 'post', i)
                    
                    # evaluate acc
                    acc_table = []
                    self.reset_cluster_labels = True
                    for j in range(i+1):
                        pass#acc_table.append(self.task_eval(j,client_idx=idx))
                    # temp_table['acc'].append(np.mean(np.asarray(acc_table)))
                    # temp_table[idx + self.n_clients*r]['acc'].append(self.task_eval(i, all_tasks=True,client_idx=idx))
                    # temp_table[idx + self.n_clients*r]['mem'].append(self.learners[idx].count_memory(self.dataset_size)) #uncomment for learners

                    # save temporary results
                    # for mkey in self.metric_keys:
                    #     save_file = temp_dir[idx] + mkey + '.csv'
                    #     np.savetxt(save_file, np.asarray(temp_table[idx + self.n_clients*r][mkey]), delimiter=",", fmt='%.2f')  #uncomment for learners

                    # if avg_train_time is not None: avg_metrics['time']['global'][i][idx + self.n_clients*r] = avg_train_time
                    # avg_metrics['mem']['global'][:][idx + self.n_clients*r] = self.learners[idx].count_memory(self.dataset_size) #uncomment for learners

                    # get other metrics
                    # if not self.oracle_flag: #uncomment for learners
                    #     #avg_metrics['plastic']['global'][i][idx] = self.task_eval(i, local=True,client_idx=idx)
                    #     til = 0
                    #     for j in range(i+1):
                    #         pass#til += self.task_eval(j, local=True,client_idx=idx)
                    #     avg_metrics['til']['global'][i][idx + self.n_clients*r] = til / (i+1)
                    #     if i > 0 and self.vis_flag: avg_metrics['cka']['global'][i][idx + self.n_clients*r] = self.sim_eval(i, local=False,client_idx=idx)
                
                #aggregate label counts to server label counts per class
                for row in range(self.num_classes):
                    for idx1 in range(self.n_clients):
                        self.label_counts_round[row] += self.label_counts[idx1][row]

                try:
                    self.server.load_model(server_model_save_dir)
                    if i > 0:
                        self.prev_server.load_model(prev_server_model_save_dir)
                except:
                    self.communicate()
                    if self.hepco:
                        print('Pre distillation:')
                        server_temp_table['predisacc'].append(self.server_task_eval(i, all_tasks=True))         
                        # if i >0:
                        #     print('Pre distillation last acc:')
                        #     server_temp_table['predislastacc'].append(self.server_task_eval(i-1, all_tasks=True))
                        if self.ignore_past_server:
                            self.prev_server = None
                        self.train_generator(round=r,task=i)
                        self.knowledge_distillation()
                    
                if not os.path.exists(server_model_save_dir): os.makedirs(server_model_save_dir)
                self.server.save_model(server_model_save_dir)

                for row in range(self.num_classes):
                    for idx1 in range(self.n_clients):
                        self.label_counts_server[row] += self.label_counts[idx1][row]

                server_temp_table['acc'].append(self.server_task_eval(i, all_tasks=True))
                server_temp_table['plastic'].append(self.server_task_eval(i, all_tasks=False))
                if i >0:
                    print('Last acc:')
                    server_temp_table['lastacc'].append(self.server_task_eval(i-1, all_tasks=True))
                
                wandb.log({'server_acc':server_temp_table['acc'][-1],'server_plastic':server_temp_table['plastic'][-1]})
                for mkey in self.metric_keys:
                    save_file = server_temp_dir + mkey + '.csv'
                    np.savetxt(save_file, np.asarray(server_temp_table[mkey]), delimiter=",", fmt='%.2f')

            self.prev_server = copy.deepcopy(self.server)
            for cls in range(self.num_classes):
                self.label_counts_server_last_task[cls] = self.label_counts_server[cls]

            self.server.last_valid_out_dim = self.server.valid_out_dim
            avg_acc += server_temp_table['acc'][-1]
            # save temporary results
            for mkey in self.metric_keys:
                save_file = server_temp_dir + mkey + '.csv'
                np.savetxt(save_file, np.asarray(server_temp_table[mkey]), delimiter=",", fmt='%.2f')
        server_temp_table['acc'].append(f'Average Accuracy : {avg_acc/(self.max_task)}')
        np.savetxt(server_temp_dir + 'acc.csv', np.asarray(server_temp_table['acc']), delimiter=",", fmt='%s')
        print('Average accuracy :', avg_acc/(self.max_task))
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