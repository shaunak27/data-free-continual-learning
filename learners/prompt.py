from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
import scipy as sp
import scipy.linalg as linalg
from utils.visualization import tsne_eval, confusion_matrix_vis, pca_eval
from torch.optim import Optimizer
import contextlib
import os
from .default import NormalNN, weight_reset, accumulate_acc, loss_fn_kd, Teacher
from .kd import LWF, get_one_hot
import copy
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.schedulers import CosineSchedule
from torch.autograd import Variable, Function

class DualPrompt(LWF):

    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        super(DualPrompt, self).__init__(learner_config)
        

    # update model - add dual prompt loss
    def update_model(self, inputs, targets, target_KD = None):
        
        # let model know that it is time to train
        if len(self.config['gpuid']) > 1:
            self.model.module.train_flag = True
        else:
            self.model.train_flag = True

        # logits
        logits, prompt_loss = self.model(inputs)
        logits = logits[:,:self.valid_out_dim]

        # # bce
        # target_mod = get_one_hot(targets-self.last_valid_out_dim, self.valid_out_dim-self.last_valid_out_dim)
        # total_loss = self.ce_loss(torch.sigmoid(logits[:,self.last_valid_out_dim:self.valid_out_dim]), target_mod)

        # standard ce
        logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # done training
        if len(self.config['gpuid']) > 1:
            self.model.module.train_flag = False
        else:
            self.model.train_flag = False

        return total_loss.detach(), prompt_loss.sum().detach(), torch.zeros((1,), requires_grad=True).cuda().detach(), logits


    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())
        # params_to_opt = list(self.model.module.prompt.parameters())
        # print(params_to_opt)
        # print(count_paramters(params_to_opt))
        optimizer_arg = {'params':params_to_opt,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)

    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'dual', prompt_param=self.prompt_param)

        return model

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        # self.model.prompt = self.model.prompt.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

class L2P(DualPrompt):

    def __init__(self, learner_config):
        super(L2P, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'l2p',prompt_param=self.prompt_param)

        return model

class DualAdapter(DualPrompt):

    def __init__(self, learner_config):
        super(DualAdapter, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'dualadapter',prompt_param=self.prompt_param)

        return model


class Finetune(LWF):

    def __init__(self, learner_config):
        super(Finetune, self).__init__(learner_config)

    def update_model(self, inputs, targets, target_KD = None):

        # get output
        logits = self.forward(inputs)

        # # bce
        # target_mod = get_one_hot(targets-self.last_valid_out_dim, self.valid_out_dim-self.last_valid_out_dim)
        # total_loss = self.ce_loss(torch.sigmoid(logits[:,self.last_valid_out_dim:self.valid_out_dim]), target_mod)

        # standard ce
        logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), total_loss.detach(), torch.zeros((1,), requires_grad=True).cuda().detach(), logits

class Linear(Finetune):

    def __init__(self, learner_config):
        super(Linear, self).__init__(learner_config)

    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        optimizer_arg = {'params':self.model.module.last.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)


def count_parameters(params):
    return sum(p.numel() for p in params if p.requires_grad)