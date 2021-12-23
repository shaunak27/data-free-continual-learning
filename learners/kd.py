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
from models.layers import CosineScaling
from models.resnet import BiasLayer
from utils.visualization import tsne_eval, confusion_matrix_vis, pca_eval
from torch.optim import Optimizer
import contextlib
import os
from .default import NormalNN, weight_reset, accumulate_acc, loss_fn_kd, Teacher
import copy
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.schedulers import CosineSchedule
from torch.autograd import Variable, Function


class LWF(NormalNN):

    def __init__(self, learner_config):
        super(LWF, self).__init__(learner_config)
        self.previous_teacher = None
        self.replay = False
        self.past_tasks = []
        self.bic_layers = None
        self.first_task = True
        self.ce_loss = nn.BCELoss(reduction='sum')
        self.ete_flag = False

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # trains
        if need_train:
            if self.reset_optimizer:  # Reset optimizer before learning each task
                self.log('Optimizer is reset!')
                self.init_optimizer()

            # data weighting
            self.data_weighting(train_dataset)
            
            # Evaluate the performance of current task
            self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
            if val_loader is not None:
                self.validation(val_loader)
        
            losses = [AverageMeter() for l in range(3)]
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            for epoch in range(self.config['schedule'][-1]):
                self.epoch=epoch

                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task)  in enumerate(train_loader):

                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x = [x[k].cuda() for k in range(len(x))]
                        y = y.cuda()
                    
                    # if KD
                    if self.KD and self.replay:
                        allowed_predictions = list(range(self.last_valid_out_dim))
                        y_hat, _ = self.previous_teacher.generate_scores(x[0], allowed_predictions=allowed_predictions)
                    else:
                        y_hat = None

                    # model update - training data
                    loss, loss_class, loss_distill, output= self.update_model(x[0], y, y_hat)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc()) 

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses[0].update(loss,  y.size(0)) 
                    losses[1].update(loss_class,  y.size(0)) 
                    losses[2].update(loss_distill,  y.size(0)) 
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses[0],acc=acc))
                self.log(' * Class Loss {loss.avg:.3f} | KD Loss {lossb.avg:.3f}'.format(loss=losses[1],lossb=losses[2]))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = [AverageMeter() for l in range(3)]
                acc = AverageMeter()

        self.model.eval()

        self.past_tasks.append(np.arange(self.last_valid_out_dim,self.valid_out_dim))
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        if self.ete_flag:

            # for eval
            if self.previous_teacher is not None:
                self.previous_previous_teacher = self.previous_teacher

            # new teacher
            teacher = Teacher(solver=self.model)
            self.previous_teacher = copy.deepcopy(teacher)
            self.replay = True

            # Extend memory
            self.task_count += 1
            if self.memory_size > 0:
                train_dataset.update_coreset_ete(self.memory_size, np.arange(self.last_valid_out_dim), teacher)
        elif self.bic_layers is None:
            # Extend memory
            self.task_count += 1
            if self.memory_size > 0:
                train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

            # for eval
            if self.previous_teacher is not None:
                self.previous_previous_teacher = self.previous_teacher

            # new teacher
            teacher = Teacher(solver=self.model)
            self.previous_teacher = copy.deepcopy(teacher)
            self.replay = True
            if len(self.config['gpuid']) > 1:
                self.previous_linear = copy.deepcopy(self.model.module.last)
            else:
                self.previous_linear = copy.deepcopy(self.model.last)
        else:
            # for eval
            if self.previous_teacher is not None:
                self.previous_previous_teacher = self.previous_teacher

            # new teacher
            teacher = TeacherBiC(solver=self.model, bic_layers = self.bic_layers)
            self.previous_teacher = copy.deepcopy(teacher)
            self.replay = True

            # Extend memory
            self.task_count += 1
            if self.memory_size > 0:
                train_dataset.update_coreset_ic(self.memory_size, np.arange(self.last_valid_out_dim), teacher)

        try:
            return batch_time.avg
        except:
            return None

    def update_model(self, inputs, targets, target_KD = None):
        
        total_loss = torch.zeros((1,), requires_grad=True).cuda()

        if self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        logits = self.forward(inputs)
        loss_class = self.criterion(logits, targets.long(), dw_cls)
        total_loss += loss_class

        # KD
        if self.KD and target_KD is not None:
            dw_KD = self.dw_k[-1 * torch.ones(len(target_KD),).long()]
            logits_KD = logits
            loss_distill = loss_fn_kd(logits_KD, target_KD, dw_KD, np.arange(self.last_valid_out_dim).tolist(), self.DTemp)
            total_loss += self.mu * loss_distill
        else:
            loss_distill = torch.zeros((1,), requires_grad=True).cuda()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_distill.detach(), logits

    def combine_data(self, data):
        x, y = [],[]
        for i in range(len(data)):
            x.append(data[i][0])
            y.append(data[i][1])
        x, y = torch.cat(x), torch.cat(y)
        return x, y

    ##########################################
    #             MODEL EVAL                 #
    ##########################################

    ##########################################
    #             MODEL UTILS                #
    ##########################################

    def combine_data(self, data):
        x, y = [],[]
        for i in range(len(data)):
            x.append(data[i][0])
            y.append(data[i][1])
        x, y = torch.cat(x), torch.cat(y)
        return x, y


class LWF_MC(LWF):

    def __init__(self, learner_config):
        super(LWF_MC, self).__init__(learner_config)
        

    def update_model(self, inputs, targets, target_KD = None):
        
        # get output
        logits = self.forward(inputs)

        # class loss
        if self.KD and target_KD is not None:
            target_mod = get_one_hot(targets, self.valid_out_dim)
            target_mod[:, :self.last_valid_out_dim] = torch.sigmoid(target_KD)
            total_loss = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)
        else:
            target_mod = get_one_hot(targets, self.valid_out_dim)
            total_loss = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), total_loss.detach(), torch.zeros((1,), requires_grad=True).cuda().detach(), logits

class LWF_NOISE_H(LWF):

    def __init__(self, learner_config):
        super(LWF_NOISE_H, self).__init__(learner_config)
        

    def update_model(self, inputs, targets, target_KD = None):
        
        feat_out = self.model.forward(inputs, h=True)
        logits_h = self.model.logits_h(feat_out)
        if not self.first_task:
            with torch.no_grad():
                targets_KD_h = self.previous_teacher.solver.logits_h(self.previous_teacher.solver.forward(inputs, h=True))

        # class loss
        loss_class = torch.zeros((1,), requires_grad=True).cuda()
        for i in range(len(logits_h)):
            
            logits = logits_h[i][:,:self.valid_out_dim]
            if not self.first_task:
                target_KD = targets_KD_h[i][:,:self.last_valid_out_dim]
                target_mod = get_one_hot(targets, self.valid_out_dim)
                target_mod[:, :self.last_valid_out_dim] = torch.sigmoid(target_KD)
                loss = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)
            else:
                target_mod = get_one_hot(targets, self.valid_out_dim)
                loss = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)
            loss_class += loss
                
        # kd feat distil
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        if not self.first_task:
            logits_h_kd = self.previous_teacher.solver.logits_h(feat_out, detach = False)
            nl = len(logits_h_kd)
            for i in range(nl):
                if i > 3:
                    logits_kd = logits_h_kd[i][:, :self.last_valid_out_dim]
                    target_KD = targets_KD_h[i][:,:self.last_valid_out_dim]
                    loss_kd += self.ce_loss(torch.sigmoid(logits_kd), torch.sigmoid(target_KD)) / len(logits_kd)

        total_loss = loss_class + loss_kd

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits 


class LWF_NOISE_HN(LWF):

    def __init__(self, learner_config):
        super(LWF_NOISE_HN, self).__init__(learner_config)
        
    def power_iter(self, x, model, head, n):

        # init optimization
        inputs = torch.randn(x.size(), requires_grad = True, device = "cuda")
        y = torch.randint(low=0, high=self.last_valid_out_dim, size=(len(inputs),)).cuda()
        model = model.cuda()
        head = head.cuda()
        optimizer_di = torch.optim.Adam([inputs], lr=0.1)

        # Create hooks for feature statistics catching
        loss_r_feature_layers = []
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module))
        loss_r_feature_layers = loss_r_feature_layers

        for i in range(n):
            optimizer_di.zero_grad()
            model.zero_grad()
            outputs = model(inputs)
            outputs = F.avg_pool2d(outputs, outputs.size()[3])
            outputs = head(outputs.view(len(outputs),-1))[:,:self.last_valid_out_dim]

            target_mod = get_one_hot(y, self.last_valid_out_dim)
            loss = self.ce_loss(torch.sigmoid(outputs), target_mod) / len(outputs)

            # R_feature loss
            for mod in loss_r_feature_layers: 
                loss_distr = mod.r_feature * 100 / len(loss_r_feature_layers)
                loss = loss + loss_distr

            loss.backward()
            optimizer_di.step()

        return x * 0 + inputs.detach()

    def update_model(self, inputs, targets, target_KD = None):
        
        feat_out = self.model.forward(inputs, h=True)
        logits_h = self.model.logits_h(feat_out)
        if not self.first_task:
            with torch.no_grad():
                targets_KD_h = self.previous_teacher.solver.logits_h(self.previous_teacher.solver.forward(inputs, h=True))

        # class loss
        loss_class = torch.zeros((1,), requires_grad=True).cuda()
        for i in range(len(logits_h)):
            
            logits = logits_h[i][:,:self.valid_out_dim]
            if not self.first_task:

                # if i == len(logits_h) - 1:
                #     feat = feat_out[-1]
                #     logits = torch.cat([self.previous_teacher.solver.last(feat.view(feat.size(0),-1))[:, :self.last_valid_out_dim], logits[:, self.last_valid_out_dim:self.valid_out_dim]], dim=1)

                target_KD = targets_KD_h[i][:,:self.last_valid_out_dim]
                target_mod = get_one_hot(targets, self.valid_out_dim)
                target_mod[:, :self.last_valid_out_dim] = torch.sigmoid(target_KD)
                loss = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)
            else:
                target_mod = get_one_hot(targets, self.valid_out_dim)
                loss = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)
            loss_class += loss
                
        # kd with noise
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        if not self.first_task:
            logits_h_kd = self.previous_teacher.solver.logits_h(feat_out, detach = False)
            nl = len(logits_h_kd)
            for i in range(nl):
                logits_kd = logits_h_kd[i][:, :self.last_valid_out_dim]
                target_KD = targets_KD_h[i][:,:self.last_valid_out_dim]
                loss_kd += self.ce_loss(torch.sigmoid(logits_kd), torch.sigmoid(target_KD)) / len(logits_kd)

                # include some noise here too :)
                if i > 0:
                    if i == 1:
                        mod1 = self.model.layer1
                        mod2 = self.previous_teacher.solver.layer1
                        head = self.previous_teacher.solver.last_1
                    if i == 2:
                        mod1 = self.model.layer2
                        mod2 = self.previous_teacher.solver.layer2
                        head = self.previous_teacher.solver.last_2
                    if i == 3:
                        mod1 = self.model.layer3
                        mod2 = self.previous_teacher.solver.layer3
                        head = self.previous_teacher.solver.last
                    mod1.eval()
                    noise_size = self.model.size_array[i-1]
                    x_ = torch.randn(noise_size, requires_grad = True).cuda()
                    x_ = self.power_iter(x_, copy.deepcopy(mod2), copy.deepcopy(head), 10)
                    x = mod1.forward(x_)
                    x = F.avg_pool2d(x, x.size()[3])
                    y = head.forward(x.view(len(x),-1))
                    x_hat = mod2.forward(x_)
                    x_hat = F.avg_pool2d(x_hat, x_hat.size()[3])
                    y_hat = head.forward(x_hat.view(len(x_hat),-1))
                    loss_kd += self.mu * self.ce_loss(torch.sigmoid(y), torch.sigmoid(y_hat).detach()) / len(logits_kd)
                    mod1.train()

        total_loss = loss_class + loss_kd

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits 






class LWF_FRB_DFC(LWF):

    def __init__(self, learner_config):
        super(LWF_FRB_DFC, self).__init__(learner_config)
        self.kd_criterion = nn.MSELoss(reduction='mean')
        self.mmd_loss = MMD_loss()
        self.beta = learner_config['beta']

    def update_model(self, inputs, targets, target_KD = None):

        # train other heads
        loss_class = 0
        feat_out = self.model.forward_h(inputs)
        logits_h = self.model.logits_h(feat_out, detach=True)
        if not self.first_task:
            with torch.no_grad():
                targets_KD_h = self.previous_teacher.solver.logits_h(self.previous_teacher.solver.forward_h(inputs))

        # class loss
        for i in range(len(logits_h)):
            
            logits_ = logits_h[i][:,:self.valid_out_dim]
            if not self.first_task:
                target_KD = targets_KD_h[i][:,:self.last_valid_out_dim]
                target_mod = get_one_hot(targets, self.valid_out_dim)
                target_mod[:, :self.last_valid_out_dim] = torch.sigmoid(target_KD)
                loss = self.ce_loss(torch.sigmoid(logits_), target_mod) / len(logits_)
            else:
                target_mod = get_one_hot(targets, self.valid_out_dim)
                loss = self.ce_loss(torch.sigmoid(logits_), target_mod) / len(logits_)
            loss_class = loss + loss_class

        if not self.first_task:

            # helper = copy.deepcopy(self.model)
            # feat = self.model.pen_forward_helped(x=inputs, helper=helper)
            # logits_kd = helper.logits(feat.view(feat.size(0),-1))[:, :self.last_valid_out_dim]
            # target_mod = torch.sigmoid(target_KD)
            # loss_class += self.ce_loss(torch.sigmoid(logits_kd), target_mod) / len(logits_kd)

            intermediate_feat_out, feat, bd = self.model.forward(x=inputs, pen=True, div = True)
            logits = self.model.logits(feat)[:,:self.valid_out_dim]
            target_mod = get_one_hot(targets-self.last_valid_out_dim, self.valid_out_dim-self.last_valid_out_dim)
            # loss_class += self.ce_loss(torch.sigmoid(logits[:,self.last_valid_out_dim:self.valid_out_dim]), target_mod) / len(logits)
           
        else:
            intermediate_feat_out, feat, bd = self.model.forward(x=inputs, pen=True, div = True)
            logits = self.model.logits(feat)[:,:self.valid_out_dim]
            target_mod = get_one_hot(targets, self.valid_out_dim)
            loss_class += self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)

        # kd feat distil
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        if not self.first_task:
            logits_h_kd = self.previous_teacher.solver.logits_h(feat_out, detach = False)
            nl = len(logits_h_kd)
            for i in range(nl):
                if i == 0:#or i == nl-1:
                    logits_kd = logits_h_kd[i][:, :self.last_valid_out_dim]
                    target_KD = targets_KD_h[i][:,:self.last_valid_out_dim]
                    loss_kd += self.ce_loss(torch.sigmoid(logits_kd), torch.sigmoid(target_KD)) / len(logits_kd)

        # Constrain to distribution
        loss_dist = torch.zeros((1,), requires_grad=True).cuda()
        nl = len(intermediate_feat_out)
        next_layer_in = {}
        for l in range(nl):
            h = len(intermediate_feat_out[l][0])
            bd_i = 0
            layer_out = [[],[]]
            for b in range(bd):

                # get output to layer
                block_out = intermediate_feat_out[l][:,bd_i:bd_i+int(h/bd)]
                
                # first loss is distribution constrain
                loss_dist = self.dist_loss(block_out) + loss_dist

                # second loss is sampling, after first task
                if not self.first_task:
                    block_in = torch.randn(self.model.size_array[l]).cuda()
                    module = self.model.get_layer(l+1)[b]
                    module_past = self.previous_teacher.solver.get_layer(l+1)[b]

                    # freeze bn
                    # for m in module: m.bn_freeze = True
                    # for m in module_past: m.bn_freeze = True
                    for m in module:
                        m.eval()
                    for m in module_past:
                        m.eval()

                    current = module.forward(block_in)
                    with torch.no_grad():
                        past = module_past.forward(block_in)
                    layer_out[0].append(current)
                    layer_out[1].append(past)
                    
                    # # unfreeze bn
                    # for m in module: m.bn_freeze = False
                    for m in module:
                        m.train()

                # save output as input to next layer
                next_layer_in[b] = block_out.size()

                # increment block
                bd_i += int(h/bd)

            if not self.first_task:
                weight_m = self.previous_teacher.solver.get_linear_layer(l+1)
                current = torch.cat(layer_out[0], dim=1)
                current = F.avg_pool2d(current, current.size()[3])
                current = torch.sigmoid(weight_m.forward(current.view(current.size(0), -1)))
                with torch.no_grad():
                    past = torch.cat(layer_out[1], dim=1)
                    past = F.avg_pool2d(past, past.size()[3])
                    past = torch.sigmoid(weight_m.forward(past.view(past.size(0), -1)))
                loss_kd = self.ce_loss(current, past) / len(current) + loss_kd

        # # total loss
        # loss_dist = loss_dist / (bd * nl)
        # loss_kd =   loss_kd   / (bd * nl)

        # scaling
        loss_dist = loss_dist * self.beta
        loss_kd =   loss_kd * self.mu 
        
        if self.first_task:
            total_loss = loss_class + loss_dist
        else:
            total_loss = loss_class + loss_dist + loss_kd

        # if self.epoch == 150: print(apple)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits 


    def n_task_class_loss(self, inputs, logits, targets, target_KD):
        target_mod = get_one_hot(targets, self.valid_out_dim)
        target_mod[:, :self.last_valid_out_dim] = torch.sigmoid(target_KD)
        loss_class = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)
        return loss_class

    def dist_loss(self,block_out):
        sample_ind = np.random.choice(block_out.size(1), 100)
        block_out = block_out[:,sample_ind]
        block_target = torch.randn(block_out.size()).cuda()
        loss_dist = self.mmd_loss(block_out,block_target)
        return loss_dist










class LWF_FRB_DFB(LWF):

    def __init__(self, learner_config):
        super(LWF_FRB_DFB, self).__init__(learner_config)
        self.kd_criterion = nn.MSELoss(reduction='mean')
        self.mmd_loss = MMD_loss()
        self.beta = learner_config['beta']

    def update_model(self, inputs, targets, target_KD = None):

        # dirty freezing
        if not self.first_task:
            
            intermediate_feat_out, feat, bd = self.model.forward(x=inputs, pen=True, div = True)
            # logits = torch.cat([self.previous_teacher.solver.logits(feat.view(feat.size(0),-1))[:, :self.last_valid_out_dim], self.model.logits(feat.view(feat.size(0),-1))[:, self.last_valid_out_dim:self.valid_out_dim]], dim=1)
            logits = self.model.logits(feat)[:,:self.valid_out_dim]
            target_mod = get_one_hot(targets, self.valid_out_dim)
            target_mod[:, :self.last_valid_out_dim] = torch.sigmoid(target_KD)
            loss_class = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)
        
        else:
            intermediate_feat_out, feat, bd = self.model.forward(x=inputs, pen=True, div = True)
            logits = self.model.logits(feat)[:,:self.valid_out_dim]
            target_mod = get_one_hot(targets, self.valid_out_dim)
            loss_class = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)

        # Constrain to distribution
        loss_dist = torch.zeros((1,), requires_grad=True).cuda()
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        nl = len(intermediate_feat_out)
        next_layer_in = {}
        for l in range(nl):
            h = len(intermediate_feat_out[l][0])
            bd_i = 0
            for b in range(bd):

                # get output to layer
                block_out = intermediate_feat_out[l][:,bd_i:bd_i+int(h/bd)]
                
                # first loss is distribution constrain
                loss_dist = self.dist_loss(block_out) + loss_dist

                # second loss is sampling, after first task
                if not self.first_task:
                    block_in = torch.randn(self.model.size_array[l]).cuda()
                    module = self.model.get_layer(l+1)[b]
                    module_past = self.previous_teacher.solver.get_layer(l+1)[b]

                    # freeze bn
                    # for m in module: m.bn_freeze = True
                    # for m in module_past: m.bn_freeze = True
                    for m in module:
                        m.eval()
                    for m in module_past:
                        m.eval()

                    
                    current = module.forward(block_in)
                    with torch.no_grad():
                        past = module_past.forward(block_in)
                    loss_kd = self.kd_criterion(current.view(current.size(0), -1), past.view(past.size(0), -1))/len(block_in) + loss_kd

                    # # unfreeze bn
                    # for m in module: m.bn_freeze = False
                    for m in module:
                        m.train()

                # save output as input to next layer
                next_layer_in[b] = block_out.size()

                # increment block
                bd_i += int(h/bd)

        # # total loss
        # loss_dist = loss_dist / (bd * nl)
        # loss_kd =   loss_kd   / (bd * nl)

        # scaling
        loss_dist = loss_dist * self.beta
        loss_kd =   loss_kd * self.mu 
        
        if self.first_task:
            total_loss = loss_class + loss_dist
        else:
            total_loss = loss_class + loss_dist + loss_kd

        # if self.epoch == 150: print(apple)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits 


    def n_task_class_loss(self, inputs, logits, targets, target_KD):
        target_mod = get_one_hot(targets, self.valid_out_dim)
        target_mod[:, :self.last_valid_out_dim] = torch.sigmoid(target_KD)
        loss_class = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)
        return loss_class

    def dist_loss(self,block_out):
        sample_ind = np.random.choice(block_out.size(1), 100)
        block_out = block_out[:,sample_ind]
        block_target = torch.randn(block_out.size()).cuda()
        loss_dist = self.mmd_loss(block_out,block_target)
        return loss_dist


























class LWF_FRB_DF(LWF):

    def __init__(self, learner_config):
        super(LWF_FRB_DF, self).__init__(learner_config)
        self.kd_criterion = nn.MSELoss(reduction='mean')
        self.mmd_loss = MMD_loss()
        self.l_freeze = learner_config['layer_freeze']
        self.beta = learner_config['beta']

    def update_model(self, inputs, targets, target_KD = None):

        if self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]

        # dirty freezing
        if not self.first_task:
            
            # current_pen, bd = self.model.forward(x=inputs, pen=True, div = True, l_freeze=self.l_freeze)
            # feat = current_pen[-1]
            # logits = self.model.logits(feat.view(feat.size(0),-1))[:, :self.valid_out_dim]

            # current_pen, bd = self.model.forward(x=inputs, pen=True, div = True, l_freeze=-1)
            # feat = current_pen[-1]
            # logits = torch.cat([self.previous_teacher.solver.logits(feat.view(feat.size(0),-1))[:, :self.last_valid_out_dim], self.model.logits(feat.view(feat.size(0),-1))[:, self.last_valid_out_dim:self.valid_out_dim]], dim=1)

            current_pen, bd = self.model.forward(x=inputs, pen=True, div = True)
            logits = self.model.logits(current_pen[-1])
            target_mod = get_one_hot(targets, self.valid_out_dim)
            target_mod[:, :self.last_valid_out_dim] = torch.sigmoid(target_KD)
            loss_class = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)
        
        else:
            logits = self.forward(inputs)
            target_mod = get_one_hot(targets, self.valid_out_dim)
            loss_class = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)
            current_pen, bd = self.model.forward(x=inputs, pen=True, div = True)

        # Constrain to distribution
        loss_dist = torch.zeros((1,), requires_grad=True).cuda()
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        nl = len(current_pen)
        next_layer_in = {}
        for l in range(0,nl):
            h = len(current_pen[l][0])
            bd_i = 0
            for b in range(bd):

                # get output to layer
                block_out = current_pen[l][:,bd_i:bd_i+int(h/bd)]

                # get input to layer
                if l > 1:
                    block_in_dim = next_layer_in[b]
                
                # first loss is distribution constrain
                if l >= self.l_freeze and l < nl-1:
                    loss_dist = self.dist_loss(block_out) + loss_dist

                if l >= self.l_freeze + 1:
                    # second loss is sampling, after first task
                    if not self.first_task:
                        block_in = torch.randn(self.model.size_array[l-1]).cuda()
                        module = self.model.get_layer(l)[b]
                        current = module.forward(block_in)
                        past = self.previous_teacher.solver.get_layer(l)[b].forward(block_in)
                        loss_kd = self.kd_criterion(current.view(current.size(0), -1), past.view(past.size(0), -1))/len(block_in) + loss_kd

                # save output as input to next layer
                next_layer_in[b] = block_out.size()

                # increment block
                bd_i += int(h/bd)

        # total loss
        loss_dist = loss_dist / (bd * nl)
        loss_kd =   loss_kd   / (bd * nl)

        # scaling
        loss_dist = loss_dist * self.beta
        loss_kd =   loss_kd * self.mu 
        
        if self.first_task:
            total_loss = loss_class + loss_dist
        else:
            total_loss = loss_class + loss_dist + loss_kd

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits 


    def n_task_class_loss(self, inputs, logits, targets, target_KD):
        target_mod = get_one_hot(targets, self.valid_out_dim)
        target_mod[:, :self.last_valid_out_dim] = torch.sigmoid(target_KD)
        loss_class = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)
        return loss_class

    def dist_loss(self,block_out):

        return torch.zeros((1,), requires_grad=True).cuda()

class LWF_FRB_DF_MMD(LWF_FRB_DF):

    def __init__(self, learner_config):
        super(LWF_FRB_DF_MMD, self).__init__(learner_config)

    def dist_loss(self,block_out):
        sample_ind = np.random.choice(block_out.size(1), 100)
        block_out = block_out[:,sample_ind]
        block_target = torch.randn(block_out.size()).cuda()
        loss_dist = self.mmd_loss(block_out,block_target)
        return loss_dist

class LWF_FRB_DF_COV(LWF_FRB_DF):

    def __init__(self, learner_config):
        super(LWF_FRB_DF_COV, self).__init__(learner_config)

    def dist_loss(self,block_out):
        block_cov = cov(block_out)
        loss_dist = torch.pow(block_out.mean(dim=0),2).sum()/len(block_out) + torch.pow(block_cov - torch.eye(block_cov.size(0)).cuda(),2).sum()/len(block_out) 
        return loss_dist


class SSIL(LWF):

    def __init__(self, learner_config):
        super(SSIL, self).__init__(learner_config)

    def update_model(self, inputs, targets, target_KD = None):
        
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        logits = self.forward(inputs)

        if not self.first_task:
            mask_p = targets < self.last_valid_out_dim
            mask_ind_p = mask_p.nonzero().view(-1) 
            mask_c = targets >= self.last_valid_out_dim
            mask_ind_c = mask_c.nonzero().view(-1) 
            loss_class_p = self.criterion(logits[mask_ind_p][:,:self.last_valid_out_dim], targets[mask_ind_p].long(), dw_cls[mask_ind_p]) * len(mask_ind_p)
            loss_class_c = self.criterion(logits[mask_ind_c][:,self.last_valid_out_dim:self.valid_out_dim], targets[mask_ind_c].long() - self.last_valid_out_dim, dw_cls[mask_ind_c]) * len(mask_ind_c)
            loss_class = (loss_class_p + loss_class_c) * (1 / (len(targets)))
            total_loss = loss_class

        else:
            loss_class = self.criterion(logits, targets, dw_cls)
            total_loss = loss_class

        # KD
        if self.KD and target_KD is not None:

            dw_KD = self.dw_k[-1 * torch.ones(len(target_KD),).long()]
            logits_KD = logits
            for task_l in self.past_tasks:
                loss_distill = loss_fn_kd(logits_KD, target_KD, dw_KD, task_l.tolist(), self.DTemp)
                total_loss += self.mu * loss_distill * (len(task_l) / self.last_valid_out_dim)
        else:
            loss_distill = torch.zeros((1,), requires_grad=True).cuda()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_distill.detach(), logits


class BIC(LWF):

    def __init__(self, learner_config):
        super(BIC, self).__init__(learner_config)
        self.bic_layers = []

    def forward(self, x):
        y_hat = self.model.forward(x)[:, :self.valid_out_dim]

        # forward with bic
        for i in range(len(self.bic_layers)):
            y_hat[:,self.bic_layers[i][0]] = self.bic_layers[i][1](y_hat[:,self.bic_layers[i][0]])

        return y_hat

    def update_model(self, inputs, targets, target_KD = None):
        
        # # if self.dw:
        # #     dw_cls = self.dw_k[targets.long()]
        # # else:
        # #     dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        # # class balancing
        # mappings = torch.ones(targets.size(), dtype=torch.float32)
        # if self.gpu:
        #     mappings = mappings.cuda()
        # rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
        # mappings[:self.last_valid_out_dim] = 2.0 * rnt
        # mappings[self.last_valid_out_dim:] = 2.0 * (1-rnt)
        # dw_cls = mappings[targets.long()]
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]

        logits = self.forward(inputs)
        loss_class = self.criterion(logits, targets.long(), dw_cls)
        total_loss = loss_class

        # KD
        if self.KD and target_KD is not None:

            mu = self.last_valid_out_dim / self.valid_out_dim
            total_loss = (1 - mu) * total_loss

            dw_KD = self.dw_k[-1 * torch.ones(len(target_KD),).long()]
            logits_KD = logits
            loss_distill = loss_fn_kd(logits_KD, target_KD, dw_KD, np.arange(0,self.last_valid_out_dim), self.DTemp)
            total_loss += mu * loss_distill
        else:
            loss_distill = torch.zeros((1,), requires_grad=True).cuda()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_distill.detach(), logits


    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        if self.task_count == 0:
            return super(BIC, self).learn_batch(train_loader, train_dataset, model_save_dir, val_loader)

        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # trains
        if need_train:
            if self.reset_optimizer:  # Reset optimizer before learning each task
                self.log('Optimizer is reset!')
                self.init_optimizer()

            # dataset start
            train_dataset.load_dataset(train_dataset.t, train=True)
            train_dataset.load_bic_dataset()
            train_dataset.append_coreset_ic()


            try: 
                self.load_model(model_save_dir, class_only = True)
            except:
                # data weighting
                self.data_weighting(train_dataset)
                
                # Evaluate the performance of current task
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
                if val_loader is not None:
                    self.validation(val_loader)
            
                losses = [AverageMeter() for l in range(3)]
                acc = AverageMeter()
                batch_time = AverageMeter()
                batch_timer = Timer()
                for epoch in range(self.config['schedule'][-1]):
                    self.epoch=epoch

                    if epoch > 0: self.scheduler.step()
                    for param_group in self.optimizer.param_groups:
                        self.log('LR:', param_group['lr'])
                    batch_timer.tic()
                    for i, (x, y, task)  in enumerate(train_loader):

                        # verify in train mode
                        self.model.train()

                        # send data to gpu
                        if self.gpu:
                            x = [x[k].cuda() for k in range(len(x))]
                            y = y.cuda()
                        
                        # if KD
                        if self.KD and self.replay:
                            allowed_predictions = list(range(self.last_valid_out_dim))
                            y_hat, _ = self.previous_teacher.generate_scores(x[0], allowed_predictions=allowed_predictions)
                        else:
                            y_hat = None

                        # model update - training data
                        loss, loss_class, loss_distill, output= self.update_model(x[0], y, y_hat)

                        # measure elapsed time
                        batch_time.update(batch_timer.toc()) 

                        # measure accuracy and record loss
                        y = y.detach()
                        accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                        losses[0].update(loss,  y.size(0)) 
                        losses[1].update(loss_class,  y.size(0)) 
                        losses[2].update(loss_distill,  y.size(0)) 
                        batch_timer.tic()

                    # eval update
                    self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                    self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses[0],acc=acc))
                    self.log(' * Class Loss {loss.avg:.3f} | KD Loss {lossb.avg:.3f}'.format(loss=losses[1],lossb=losses[2]))

                    # Evaluate the performance of current task
                    if val_loader is not None:
                        self.validation(val_loader)

                    # reset
                    losses = [AverageMeter() for l in range(3)]
                    acc = AverageMeter()

            # save halfway point
            self.model.eval()
            self.save_model(model_save_dir, class_only = True)

            # part b
            # dataset tune
            train_dataset.load_dataset(train_dataset.t, train=True)
            train_dataset.load_bic_dataset(post=True)
            train_dataset.append_coreset_ic(post=True)
            print(len(train_dataset.data))
            print(len(train_dataset.targets))

            # bias correction layer
            self.bic_layers.append([np.arange(self.last_valid_out_dim,self.valid_out_dim),BiasLayer().cuda()])
            self.config['lr'] = self.config['lr'] / 1e2
            self.optimizer, self.scheduler = self.new_optimizer(self.bic_layers[-1][1])
            self.config['lr'] = self.config['lr'] * 1e2
            
            # data weighting
            self.data_weighting(train_dataset)

            # Evaluate the performance of current task
            self.log('Balance Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
            if val_loader is not None:
                self.validation(val_loader)
        
            losses = [AverageMeter() for l in range(1)]
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            
            for epoch in range(self.config['schedule'][-1]):
                self.epoch=epoch

                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task)  in enumerate(train_loader):

                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x = [x[k].cuda() for k in range(len(x))]
                        y = y.cuda()

                    # model update - training data
                    dw_cls = self.dw_k[-1 * torch.ones(y.size()).long()]
                    output = self.forward(x[0])
                    loss = self.criterion(output, y.long(), dw_cls)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # measure elapsed time
                    batch_time.update(batch_timer.toc()) 

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses[0].update(loss.detach(),  y.size(0)) 
                    batch_timer.tic()

                # eval update
                self.log('Balance Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses[0],acc=acc))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = [AverageMeter() for l in range(1)]
                acc = AverageMeter()

            # dataset final
            train_dataset.load_dataset(train_dataset.t, train=True)
            train_dataset.append_coreset(only=False)

        self.model.eval()
        self.past_tasks.append(np.arange(self.last_valid_out_dim,self.valid_out_dim))
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # for eval
        if self.previous_teacher is not None:
            self.previous_previous_teacher = self.previous_teacher

        # new teacher
        teacher = TeacherBiC(solver=self.model, bic_layers = self.bic_layers)
        self.previous_teacher = copy.deepcopy(teacher)
        self.replay = True

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset_ic(self.memory_size, np.arange(self.last_valid_out_dim), teacher)

        try:
            return batch_time.avg
        except:
            return None

    def save_model(self, filename, class_only = False):
        model_state = self.model.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving class model to:', filename)
        torch.save(model_state, filename + 'class.pth')
        self.log('=> Save Done')
        print(self.bic_layers)
        print(len(self.bic_layers))
        if not class_only:
            for tc in range(1,self.task_count):
                tci = tc + 1
                model_state = self.bic_layers[tc-1][1].state_dict()
                for key in model_state.keys():  # Always save it to cpu
                    model_state[key] = model_state[key].cpu()
                    torch.save(model_state, filename + 'BiC-' + str(tci+1) + '.pth')

    def load_model(self, filename, class_only = False):
        self.model.load_state_dict(torch.load(filename + 'class.pth'))
        self.log('=> Load Done')
        if self.gpu:
            self.model = self.model.cuda()
        self.model.eval()

        if not class_only:
            bic_layers = []
            for tc in range(1,self.task_count+1):
                tci = tc + 1
                bic_layers.append([self.tasks[tc],BiasLayer().cuda()])
                bic_layers[tc-1][1].load_state_dict(torch.load(filename + 'BiC-' + str(tci+1) + '.pth'))
        
        self.bic_layers = bic_layers

    def validation(self, dataloader,  task_in = None, task_metric='acc', relabel_clusters = True, verbal = True):

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = self.training
        self.eval()
        for i, (input, target, task) in enumerate(dataloader):
            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            if task_in is None:
                output = self.forward(input)[:, :self.valid_out_dim]
                acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
            else:
                mask = target >= task_in[0]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]

                mask = target < task_in[-1]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]
                
                if len(target) > 1:
                    output = self.forward(input)[:, task_in]
                    acc = accumulate_acc(output, target-task_in[0], task, acc, topk=(self.top_k,))
            
        self.train(orig_mode)

        if verbal:
            self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                    .format(acc=acc, time=batch_timer.toc()))
        return acc.avg

class TeacherBiC(nn.Module):

    def __init__(self, solver, bic_layers):

        super().__init__()
        self.solver = solver
        self.bic_layers = bic_layers

    def generate_scores(self, x, allowed_predictions=None, threshold=None):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x)

        # forward with bic
        for i in range(len(self.bic_layers)):
            y_hat[:,self.bic_layers[i][0]] = self.bic_layers[i][1](y_hat[:,self.bic_layers[i][0]])

        y_hat = y_hat[:, allowed_predictions]

        # set model back to its initial mode
        self.train(mode=mode)

        # # threshold if desired
        # if threshold is not None:
        #     # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
        #     y_hat = F.softmax(y_hat, dim=1)
        #     ymax, y = torch.max(y_hat, dim=1)
        #     thresh_mask = ymax > (threshold)
        #     thresh_idx = thresh_mask.nonzero().view(-1)
        #     y_hat = y_hat[thresh_idx]
        #     y = y[thresh_idx]
        #     return y_hat, y, x[thresh_idx]

        # else:
        # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
        ymax, y = torch.max(y_hat, dim=1)

        return y_hat, y

    def generate_scores_pen(self, x):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x=x, pen=True)

        # set model back to its initial mode
        self.train(mode=mode)

        return y_hat

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.target = None

    def hook_fn(self, module, input, output):

        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False) + 1e-8
        r_feature = torch.log(var**(0.5) / (module.running_var.data.type(var.type()) + 1e-8)**(0.5)).mean() - 0.5 * (1.0 - (module.running_var.data.type(var.type()) + 1e-8 + (module.running_mean.data.type(var.type())-mean)**2)/var).mean()

        self.r_feature = r_feature

            
    def close(self):
        self.hook.remove()

def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()

def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).cuda()
    one_hot[range(target.shape[0]), target]=1
    return one_hot