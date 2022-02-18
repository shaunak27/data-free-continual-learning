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
from .module_helper import ModuleHelper
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
        if not self.overwrite or self.task_count == 0:
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
                    self.step = i

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
        self.last_last_valid_out_dim = self.last_valid_out_dim
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
        
        # prepare dataloaders for block replay methods
        self.accumulate_block_memory(train_loader)

        try:
            return batch_time.avg
        except:
            return None

    def accumulate_block_memory(self, train_loader):
        pass

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

class LWF_MC_ewc(LWF):

    def __init__(self, learner_config):
        super(LWF_MC_ewc, self).__init__(learner_config)
        self.regularization_terms = {}
        

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

        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        if self.KD and target_KD is not None:
            # Calculate the reg_loss only when the regularization_terms exists
            reg_loss = 0
            for i,reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term['importance']
                task_param = reg_term['task_param']
                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
                reg_loss += task_reg_loss
            loss_kd += self.mu * reg_loss / self.task_count

        total_loss = total_loss + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()
        return total_loss.detach(), total_loss.detach() - loss_kd.detach(), loss_kd.detach(), logits

    def accumulate_block_memory(self, train_loader):
        dataloader = train_loader
        # Update the diag fisher information
        # There are several ways to estimate the F matrix.
        # We keep the implementation as simple as possible while maintaining a similar performance to the literature.
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.log('Computing EWC')

        # Initialize the importance matrix
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(0)  # zero initialized

        mode = self.training
        self.eval()

        # Accumulate the square of gradients
        for i, (input, target, task) in enumerate(dataloader):
            if self.gpu:
                input = input[0].cuda()
                target = target.cuda()

            pred = self.model.forward(input)[:,:self.valid_out_dim]
            ind = pred.max(1)[1].flatten()  # Choose the one with max

            # loss = self.criterion(pred, ind, task, regularization=False)
            target_mod = get_one_hot(target, self.valid_out_dim)
            loss = self.ce_loss(torch.sigmoid(pred), target_mod) / len(pred)
            self.model.zero_grad()
            loss.backward()
            for n, p in importance.items():
                if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
                    p += ((self.params[n].grad ** 2) * len(input) / len(dataloader))

        self.train(mode=mode)
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()
        self.regularization_terms[self.task_count] = {'importance':importance, 'task_param':task_param}


class MC_ewc(LWF):

    def __init__(self, learner_config):
        super(MC_ewc, self).__init__(learner_config)
        self.regularization_terms = {}
        

    def update_model(self, inputs, targets, target_KD = None):
        
        # get output
        logits = self.forward(inputs)

        # class loss
        if self.KD and target_KD is not None:
            target_mod = get_one_hot(targets-self.last_valid_out_dim, self.valid_out_dim-self.last_valid_out_dim)
            total_loss = self.ce_loss(torch.sigmoid(logits[:,self.last_valid_out_dim:self.valid_out_dim]), target_mod) / len(logits)
        else:
            target_mod = get_one_hot(targets, self.valid_out_dim)
            total_loss = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)

        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        if self.KD and target_KD is not None:
            # Calculate the reg_loss only when the regularization_terms exists
            reg_loss = 0
            for i,reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term['importance']
                task_param = reg_term['task_param']
                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
                reg_loss += task_reg_loss
            loss_kd += self.mu * reg_loss / self.task_count

        total_loss = total_loss + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()
        return total_loss.detach(), total_loss.detach() - loss_kd.detach(), loss_kd.detach(), logits

    def accumulate_block_memory(self, train_loader):
        dataloader = train_loader
        # Update the diag fisher information
        # There are several ways to estimate the F matrix.
        # We keep the implementation as simple as possible while maintaining a similar performance to the literature.
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.log('Computing EWC')

        # Initialize the importance matrix
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(0)  # zero initialized

        mode = self.training
        self.eval()

        # Accumulate the square of gradients
        for i, (input, target, task) in enumerate(dataloader):
            if self.gpu:
                input = input[0].cuda()
                target = target.cuda()

            pred = self.model.forward(input)[:,:self.valid_out_dim]
            ind = pred.max(1)[1].flatten()  # Choose the one with max

            # loss = self.criterion(pred, ind, task, regularization=False)
            target_mod = get_one_hot(target, self.valid_out_dim)
            loss = self.ce_loss(torch.sigmoid(pred), target_mod) / len(pred)
            self.model.zero_grad()
            loss.backward()
            for n, p in importance.items():
                if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
                    p += ((self.params[n].grad ** 2) * len(input) / len(dataloader))

        self.train(mode=mode)
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()
        self.regularization_terms[self.task_count] = {'importance':importance, 'task_param':task_param}


class LWF_MC_ewc_b(LWF):

    def __init__(self, learner_config):
        super(LWF_MC_ewc_b, self).__init__(learner_config)
        self.regularization_terms = {}
        

    def update_model(self, inputs, targets, target_KD = None):
        
        # get output
        logits = self.forward(inputs)

        # class loss
        if self.KD and target_KD is not None:
            target_mod = get_one_hot(targets-self.last_valid_out_dim, self.valid_out_dim-self.last_valid_out_dim)
            total_loss = self.ce_loss(torch.sigmoid(logits[:,self.last_valid_out_dim:self.valid_out_dim]), target_mod) / len(logits)
            pen = self.model.forward(inputs, pen=True)
            logits_kd = self.previous_teacher.solver.last(pen)[:,:self.last_valid_out_dim]
            total_loss += self.ce_loss(torch.sigmoid(logits_kd), torch.sigmoid(target_KD)) / len(logits)
        else:
            target_mod = get_one_hot(targets, self.valid_out_dim)
            total_loss = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)

        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        if self.KD and target_KD is not None:
            # Calculate the reg_loss only when the regularization_terms exists
            reg_loss = 0
            for i,reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term['importance']
                task_param = reg_term['task_param']
                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
                reg_loss += task_reg_loss
            loss_kd += self.mu * reg_loss / self.task_count

        total_loss = total_loss + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()
        return total_loss.detach(), total_loss.detach() - loss_kd.detach(), loss_kd.detach(), logits

    def accumulate_block_memory(self, train_loader):
        dataloader = train_loader
        # Update the diag fisher information
        # There are several ways to estimate the F matrix.
        # We keep the implementation as simple as possible while maintaining a similar performance to the literature.
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.log('Computing EWC')

        # Initialize the importance matrix
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(0)  # zero initialized

        mode = self.training
        self.eval()

        # Accumulate the square of gradients
        for i, (input, target, task) in enumerate(dataloader):
            if self.gpu:
                input = input[0].cuda()
                target = target.cuda()

            pred = self.model.forward(input)[:,:self.valid_out_dim]
            ind = pred.max(1)[1].flatten()  # Choose the one with max

            # loss = self.criterion(pred, ind, task, regularization=False)
            target_mod = get_one_hot(target, self.valid_out_dim)
            loss = self.ce_loss(torch.sigmoid(pred), target_mod) / len(pred)
            self.model.zero_grad()
            loss.backward()
            for n, p in importance.items():
                if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
                    p += ((self.params[n].grad ** 2) * len(input) / len(dataloader))

        self.train(mode=mode)
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()
        self.regularization_terms[self.task_count] = {'importance':importance, 'task_param':task_param}



































class LWF_FRB_DFC_lwf(LWF):

    def __init__(self, learner_config):
        super(LWF_FRB_DFC_lwf, self).__init__(learner_config)
        self.kd_criterion = nn.MSELoss(reduction='mean')
        self.mmd_loss = MMD_loss()
        self.beta = learner_config['beta']
        self.block_helper_dict = {}
        self.block_reg_term = {}

    def update_model(self, inputs, targets, target_KD = None):
        
        epoch_change = 70
        if not self.first_task:
            if self.epoch == 0 and self.step == 0:
                import copy
                self.model.layer1_pre = copy.deepcopy(self.model.layer1_pre)
                self.model.layer2_pre = copy.deepcopy(self.model.layer2_pre)
                self.model.layer3_pre = copy.deepcopy(self.model.layer3_pre)
                self.model.conv1 = copy.deepcopy(self.model.conv1)
            elif self.epoch == epoch_change and self.step == 0:
                params = list(self.model.layer1_pre.parameters()) + list(self.model.layer2_pre.parameters()) + list(self.model.layer3_pre.parameters())
                optimizer_arg = {'params':params,
                                'lr':self.config['lr'],
                                'weight_decay':self.config['weight_decay'],
                                'momentum':self.config['momentum']}
                self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
                # new teacher
                import copy
                teacher = Teacher(solver=self.model)
                self.current_teacher = copy.deepcopy(teacher)

        # get logits and intermediate features
        intermediate_feat_out, feat, bd = self.model.forward(x=inputs, pen=True, div = True)
        logits = self.model.logits(feat)[:,:self.valid_out_dim]

        # class loss
        if self.KD and target_KD is not None:
            if self.epoch < epoch_change:
                target_mod = get_one_hot(targets - self.last_valid_out_dim, self.valid_out_dim - self.last_valid_out_dim)
                loss_class = self.ce_loss(torch.sigmoid(logits[:,self.last_valid_out_dim:self.valid_out_dim]), target_mod) / len(logits)
                # target_mod = get_one_hot(targets, self.valid_out_dim)
                # target_mod[:, :self.last_valid_out_dim] = torch.sigmoid(target_KD)
                # loss_class = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)
            else:
                allowed_predictions = list(range(self.last_valid_out_dim, self.valid_out_dim))
                y_hat, _ = self.current_teacher.generate_scores(inputs, allowed_predictions=allowed_predictions)
                target_mod = torch.cat([torch.sigmoid(target_KD), torch.sigmoid(y_hat)], dim=1)
                loss_class = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)
        else:
            target_mod = get_one_hot(targets, self.valid_out_dim)
            loss_class = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)
            
        # Constrain to distribution
        loss_dist = torch.zeros((1,), requires_grad=True).cuda()
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        nl = len(intermediate_feat_out)
        if self.epoch < epoch_change:
            for l in range(nl):

                h = len(intermediate_feat_out[l][0])
                bd_i = 0
                for b in range(bd):

                    # get output to layer
                    block_in = intermediate_feat_out[l][:,bd_i:bd_i+int(h/bd)]
                    
                    ##################
                    # CONTRIBUTION 1 #
                    ##################
                    # first loss is distribution constrain
                    # loss_dist = self.dist_loss(block_in) + loss_dist
                    loss_dist = 0 + loss_dist

                    # second loss is sampling kd, after first task
                    if not self.first_task:

                        ##################
                        # CONTRIBUTION 2 #
                        ##################
                        # sample from block for KD
                        # block_in = torch.randn(self.model.size_array[l]).cuda()
                        block_in = self.block_helper_dict[l][b].generate_samples()

                        module = self.model.get_layer(l+1)[b]
                        module_past = self.previous_teacher.solver.get_layer(l+1)[b]

                        # freeze bn
                        for m in module:
                            m.eval()
                        for m in module_past:
                            m.eval()

                        current = module.forward(block_in)
                        with torch.no_grad():
                            past = module_past.forward(block_in)
                        if not self.first_task:
                            loss_kd = self.kd_criterion(current, past) + loss_kd

                            # sample_ind = np.random.choice(block_in.size(1), 100)
                            # loss_kd = self.mmd_loss(current.reshape(len(current),-1)[:,sample_ind], past.reshape(len(past),-1)[:,sample_ind]) + loss_kd

                            # reg_loss = 0
                            # for n, p in self.block_reg_term[l][b]['params'].items():
                            #     importance = self.block_reg_term[l][b]['importance'][n]
                            #     reg_loss += (importance * (p - self.block_reg_term[l][b]['values'][n]) ** 2).sum()
                            # reg_loss += reg_loss
                            # loss_kd = reg_loss + loss_kd
                        
                        # # unfreeze bn
                        for m in module:
                            m.train()

                    # increment block
                    bd_i += int(h/bd)

        # total loss
        # scaling
        loss_dist = loss_dist * self.beta / (nl*bd)
        loss_kd =   loss_kd * self.mu / (nl*bd)
        
        if self.first_task:
            total_loss = loss_class + loss_dist
        else:
            total_loss = loss_class + loss_dist + loss_kd

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits 

    def accumulate_block_memory(self, train_loader):
        
        # verify in eval mode
        self.model.eval()
        for i, (x, y, task)  in enumerate(train_loader):

            # send data to gpu
            if self.gpu:
                x = [x[k].cuda() for k in range(len(x))]
                y = y.cuda()

            # get logits and intermediate features
            intermediate_feat_out, feat, bd = self.model.forward(x=x[0], pen=True, div = True)
            logits = self.model.logits(feat)[:,:self.valid_out_dim]
            logits.pow_(2)
            loss = logits.mean()
            self.model.zero_grad()
            loss.backward()

            # Constrain to distribution
            nl = len(intermediate_feat_out)
            for l in range(nl):
                
                # get helper block layer
                if not l in self.block_helper_dict:
                    self.block_helper_dict[l] = {}
                if not l in self.block_reg_term:
                    self.block_reg_term[l] = {}

                h = len(intermediate_feat_out[l][0])
                bd_i = 0
                for b in range(bd):

                    # get helper block
                    if not b in self.block_helper_dict[l]:
                        self.block_helper_dict[l][b] = ModuleHelper(self.model.size_array[l], bd)

                    # get output to layer
                    block_in = intermediate_feat_out[l][:,bd_i:bd_i+int(h/bd)]

                    # update block helper
                    self.block_helper_dict[l][b].update_memory(block_in)

                    # get module
                    module = self.model.get_layer(l+1)[b]

                    # freeze bn
                    for m in module:
                        m.eval()

                    # get helper block
                    if not b in self.block_reg_term[l]:
                        self.block_reg_term[l][b] = {}
                        self.block_reg_term[l][b]['params'] = {n: p for n, p in module.named_parameters() if p.requires_grad}
                        self.block_reg_term[l][b]['values'] = {}
                        self.block_reg_term[l][b]['importance'] = {}
                        for n, p in self.block_reg_term[l][b]['params'].items():
                            self.block_reg_term[l][b]['values'][n] = p.clone().detach()
                            self.block_reg_term[l][b]['importance'][n] = 0

                    for n, p in self.block_reg_term[l][b]['params'].items():
                        if p.grad is not None:
                            self.block_reg_term[l][b]['importance'][n] = (p.grad.abs() / len(train_loader)) + self.block_reg_term[l][b]['importance'][n]

                    # increment block
                    bd_i += int(h/bd)

        # update block helper loader
        for l in range(nl):
            for b in range(bd):
                self.block_helper_dict[l][b].update_loader(self.task_count)

    def n_task_class_loss(self, inputs, logits, targets, target_KD):
        target_mod = get_one_hot(targets, self.valid_out_dim)
        target_mod[:, :self.last_valid_out_dim] = torch.sigmoid(target_KD)
        loss_class = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)
        return loss_class

    def dist_loss(self,block_in):
        sample_ind = np.random.choice(block_in.size(1), 100)
        block_in = block_in[:,sample_ind]
        block_target = torch.randn(block_in.size()).cuda()
        loss_dist = self.mmd_loss(block_in,block_target)
        return loss_dist

    def visualization(self, dataloader, topdir, name, task, embedding):

        # past vis
        super(LWF_FRB_DFC_lwf, self).visualization(dataloader, topdir, name, task, embedding)

        if self.task_count == 0: return

        # gather data
        orig_mode = self.training
        self.eval()
        metrics = {'CKA':{},'MSE':{},'MMD':{}} # layer, then average (over blocks) for CKA, MSE, and MMD
        metrics_blockmem = {'CKA':{},'MSE':{},'MMD':{}} # layer, then average (over blocks) for CKA, MSE, and MMD
        dist_match = {'CKA':{'NormalG':{},'LearnedG':{}},'MSE':{'NormalG':{},'LearnedG':{}},'MMD':{'NormalG':{},'LearnedG':{}}} # layer, then average (over blocks) for {MSE, MMD} with {normal, learned gaussian}
        for i, (input, target, _) in enumerate(dataloader):
            if self.gpu:
                with torch.no_grad():
                    x = input.cuda()
                
            # get logits and intermediate features
            intermediate_feat_out, feat, bd = self.previous_teacher.solver.forward(x=x, pen=True, div = True)

            # Constrain to distribution
            with torch.no_grad():
                nl = len(intermediate_feat_out)
                first_time = [True for l in range(nl)]
                for l in range(nl):

                    if first_time[l]:
                        metrics['CKA'][l] = 0
                        metrics_blockmem['CKA'][l] = 0
                        dist_match['CKA']['NormalG'][l] = 0
                        dist_match['CKA']['LearnedG'][l] = 0
                        metrics['MSE'][l] = 0
                        metrics_blockmem['MSE'][l] = 0
                        dist_match['MSE']['NormalG'][l] = 0
                        dist_match['MSE']['LearnedG'][l] = 0
                        metrics['MMD'][l] = 0
                        metrics_blockmem['MMD'][l] = 0
                        dist_match['MMD']['NormalG'][l] = 0
                        dist_match['MMD']['LearnedG'][l] = 0
                        first_time[l] = False

                    h = len(intermediate_feat_out[l][0])
                    bd_i = 0
                    for b in range(bd):

                        # get output to layer
                        block_in = self.block_helper_dict[l][b].unflatten(intermediate_feat_out[l][:,bd_i:bd_i+int(h/bd)])

                        # update block helper
                        block_mem = self.block_helper_dict[l][b].generate_samples()

                        module = self.previous_teacher.solver.get_layer(l+1)[b]
                        if self.task_count > 1:
                            module_past = self.previous_previous_teacher.solver.get_layer(l+1)[b]

                        # freeze bn
                        # for m in module:
                        #     m.eval()
                        module.eval()
                        if self.task_count > 1:
                            # for m in module_past:
                            #     m.eval()
                            module_past.eval()

                        current = module.forward(block_in)
                        current_mem = module.forward(block_mem)
                        if self.task_count > 1:
                            past = module_past.forward(block_in)
                            past_mem = module_past.forward(block_mem)

                        # generate metrics
                        metrics['CKA'][l] += 0
                        metrics_blockmem['CKA'][l] += 0
                        dist_match['CKA']['NormalG'][l] += 0
                        dist_match['CKA']['LearnedG'][l] += 0

                        metrics['MSE'][l] += 0
                        metrics_blockmem['MSE'][l] += 0
                        dist_match['MSE']['NormalG'][l] += 0
                        dist_match['MSE']['LearnedG'][l] += 0

                        metrics['MMD'][l] += 0
                        metrics_blockmem['MMD'][l] += 0
                        dist_match['MMD']['NormalG'][l] += 0
                        dist_match['MMD']['LearnedG'][l] += 0
                        # metrics use data from previous tasks (self.last_last_valid_out_dim), dist_match uses data from current task
                        # list.extend(metric.cpu().detach().numpy().tolist()))

                        
                        # unfreeze bn
                        # for m in module:
                        #     m.train()
                        module.train()

                        # increment block
                        bd_i += int(h/bd)

            
        self.train(orig_mode)

        # save results
        savedir = topdir + name + '/block_metrics/'
        if not os.path.exists(savedir): os.makedirs(savedir)
        savename = savedir 
        import yaml
        for metric in ['CKA','MSE','MMD']:

            with open(savedir + 'raw-'+metric, 'w') as yaml_file:
                yaml.dump(metrics[metric], yaml_file, default_flow_style=False)
            with open(savedir + 'blockmem-'+metric, 'w') as yaml_file:
                yaml.dump(metrics_blockmem[metric], yaml_file, default_flow_style=False)
            with open(savedir + 'dist-'+metric, 'w') as yaml_file:
                yaml.dump(dist_match[metric], yaml_file, default_flow_style=False)




class LWF_FRB_DFC_lwfb(LWF):

    def __init__(self, learner_config):
        super(LWF_FRB_DFC_lwfb, self).__init__(learner_config)
        self.kd_criterion = nn.MSELoss(reduction='mean')
        self.mmd_loss = MMD_loss()
        self.beta = learner_config['beta']
        self.block_helper_dict = {}
        self.block_reg_term = {}

    def update_model(self, inputs, targets, target_KD = None):
        

        if not self.first_task:
            import copy
            self.model.conv1 = copy.deepcopy(self.model.conv1)

        # get logits and intermediate features
        intermediate_feat_out, logits, bd = self.model.forward(x=inputs, div = True)
        logits = logits[:,:self.valid_out_dim]

        # class loss
        if self.KD and target_KD is not None:
            target_mod = get_one_hot(targets, self.valid_out_dim)
            target_mod[:, :self.last_valid_out_dim] = torch.sigmoid(target_KD)
            loss_class = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)
        else:
            target_mod = get_one_hot(targets, self.valid_out_dim)
            loss_class = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)
            
        # Constrain to distribution
        loss_dist = torch.zeros((1,), requires_grad=True).cuda()
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        nl = len(intermediate_feat_out)
        for l in range(nl):

            h = len(intermediate_feat_out[l][0])
            bd_i = 0
            for b in range(bd):

                # get output to layer
                block_in = intermediate_feat_out[l][:,bd_i:bd_i+int(h/bd)]
                
                ##################
                # CONTRIBUTION 1 #
                ##################
                # first loss is distribution constrain
                # loss_dist = self.dist_loss(block_in) + loss_dist
                loss_dist = 0 + loss_dist

                # second loss is sampling kd, after first task
                if not self.first_task:

                    ##################
                    # CONTRIBUTION 2 #
                    ##################
                    # sample from block for KD
                    # block_in = torch.randn(self.model.size_array[l]).cuda()
                    block_in = self.block_helper_dict[l][b].generate_samples()

                    module = self.model.get_layer(l+1)[b]
                    module_past = self.previous_teacher.solver.get_layer(l+1)[b]

                    # freeze bn
                    # for m in module:
                    #     m.eval()
                    # for m in module_past:
                    #     m.eval()
                    module.eval()
                    module_past.eval()

                    current = torch.sigmoid(module.forward(block_in))
                    with torch.no_grad():
                        past = torch.sigmoid(module_past.forward(block_in))
                    if not self.first_task:
                        loss_kd = self.kd_criterion(current, past) + loss_kd
                        # loss_kd = loss_kd + self.ce_loss(current, past) / len(current)

                        # sample_ind = np.random.choice(block_in.size(1), 100)
                        # loss_kd = self.mmd_loss(current.reshape(len(current),-1)[:,sample_ind], past.reshape(len(past),-1)[:,sample_ind]) + loss_kd

                        # reg_loss = 0
                        # for n, p in self.block_reg_term[l][b]['params'].items():
                        #     importance = self.block_reg_term[l][b]['importance'][n]
                        #     reg_loss += (importance * (p - self.block_reg_term[l][b]['values'][n]) ** 2).sum()
                        # reg_loss += reg_loss
                        # loss_kd = reg_loss + loss_kd
                    
                    # # # unfreeze bn
                    # for m in module:
                    #     m.train()
                    module.train()

                # increment block
                bd_i += int(h/bd)

        # total loss
        # scaling
        loss_dist = loss_dist * self.beta / (nl*bd)
        loss_kd =   loss_kd * self.mu / (nl*bd)
        
        if self.first_task:
            total_loss = loss_class + loss_dist
        else:
            total_loss = loss_class + loss_dist + loss_kd

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits 

    def accumulate_block_memory(self, train_loader):
        
        # verify in eval mode
        self.model.eval()
        for i, (x, y, task)  in enumerate(train_loader):

            # send data to gpu
            if self.gpu:
                x = [x[k].cuda() for k in range(len(x))]
                y = y.cuda()

            # get logits and intermediate features
            intermediate_feat_out, logits, bd = self.model.forward(x=x[0], div = True)
            logits = logits[:,:self.valid_out_dim]
            logits.pow_(2)
            loss = logits.mean()
            self.model.zero_grad()
            loss.backward()

            # Constrain to distribution
            nl = len(intermediate_feat_out)
            for l in range(nl):
                
                # get helper block layer
                if not l in self.block_helper_dict:
                    self.block_helper_dict[l] = {}
                if not l in self.block_reg_term:
                    self.block_reg_term[l] = {}

                h = len(intermediate_feat_out[l][0])
                bd_i = 0
                for b in range(bd):

                    # get helper block
                    if not b in self.block_helper_dict[l]:
                        self.block_helper_dict[l][b] = ModuleHelper(self.model.size_array[l], bd)

                    # get output to layer
                    block_in = intermediate_feat_out[l][:,bd_i:bd_i+int(h/bd)]

                    # update block helper
                    self.block_helper_dict[l][b].update_memory(block_in)

                    # get module
                    module = self.model.get_layer(l+1)[b]

                    # freeze bn
                    # for m in module:
                    #     m.eval()
                    module.eval()

                    # get helper block
                    if not b in self.block_reg_term[l]:
                        self.block_reg_term[l][b] = {}
                        self.block_reg_term[l][b]['params'] = {n: p for n, p in module.named_parameters() if p.requires_grad}
                        self.block_reg_term[l][b]['values'] = {}
                        self.block_reg_term[l][b]['importance'] = {}
                        for n, p in self.block_reg_term[l][b]['params'].items():
                            self.block_reg_term[l][b]['values'][n] = p.clone().detach()
                            self.block_reg_term[l][b]['importance'][n] = 0

                    for n, p in self.block_reg_term[l][b]['params'].items():
                        if p.grad is not None:
                            self.block_reg_term[l][b]['importance'][n] = (p.grad.abs() / len(train_loader)) + self.block_reg_term[l][b]['importance'][n]

                    # increment block
                    bd_i += int(h/bd)

        # update block helper loader
        for l in range(nl):
            for b in range(bd):
                self.block_helper_dict[l][b].update_loader(self.task_count)

    def n_task_class_loss(self, inputs, logits, targets, target_KD):
        target_mod = get_one_hot(targets, self.valid_out_dim)
        target_mod[:, :self.last_valid_out_dim] = torch.sigmoid(target_KD)
        loss_class = self.ce_loss(torch.sigmoid(logits), target_mod) / len(logits)
        return loss_class

    def dist_loss(self,block_in):
        sample_ind = np.random.choice(block_in.size(1), 100)
        block_in = block_in[:,sample_ind]
        block_target = torch.randn(block_in.size()).cuda()
        loss_dist = self.mmd_loss(block_in,block_target)
        return loss_dist

    def visualization(self, dataloader, topdir, name, task, embedding):

        # past vis
        super(LWF_FRB_DFC_lwfb, self).visualization(dataloader, topdir, name, task, embedding)

        # if self.task_count == 0: return

        # # gather data
        # orig_mode = self.training
        # self.eval()
        # metrics = {'CKA':{},'MSE':{},'MMD':{}} # layer, then average (over blocks) for CKA, MSE, and MMD
        # metrics_blockmem = {'CKA':{},'MSE':{},'MMD':{}} # layer, then average (over blocks) for CKA, MSE, and MMD
        # dist_match = {'CKA':{'NormalG':{},'LearnedG':{}},'MSE':{'NormalG':{},'LearnedG':{}},'MMD':{'NormalG':{},'LearnedG':{}}} # layer, then average (over blocks) for {MSE, MMD} with {normal, learned gaussian}
        # for i, (input, target, _) in enumerate(dataloader):
        #     if self.gpu:
        #         with torch.no_grad():
        #             x = input.cuda()
                
        #     # get logits and intermediate features
        #     intermediate_feat_out, feat, bd = self.previous_teacher.solver.forward(x=x, pen=True, div = True)

        #     # Constrain to distribution
        #     with torch.no_grad():
        #         nl = len(intermediate_feat_out)
        #         first_time = [True for l in range(nl)]
        #         for l in range(nl):

        #             if first_time[l]:
        #                 metrics['CKA'][l] = 0
        #                 metrics_blockmem['CKA'][l] = 0
        #                 dist_match['CKA']['NormalG'][l] = 0
        #                 dist_match['CKA']['LearnedG'][l] = 0
        #                 metrics['MSE'][l] = 0
        #                 metrics_blockmem['MSE'][l] = 0
        #                 dist_match['MSE']['NormalG'][l] = 0
        #                 dist_match['MSE']['LearnedG'][l] = 0
        #                 metrics['MMD'][l] = 0
        #                 metrics_blockmem['MMD'][l] = 0
        #                 dist_match['MMD']['NormalG'][l] = 0
        #                 dist_match['MMD']['LearnedG'][l] = 0
        #                 first_time[l] = False

        #             h = len(intermediate_feat_out[l][0])
        #             bd_i = 0
        #             for b in range(bd):

        #                 # get output to layer
        #                 block_in = self.block_helper_dict[l][b].unflatten(intermediate_feat_out[l][:,bd_i:bd_i+int(h/bd)])

        #                 # update block helper
        #                 block_mem = self.block_helper_dict[l][b].generate_samples()

        #                 module = self.previous_teacher.solver.get_layer(l+1)[b]
        #                 if self.task_count > 1:
        #                     module_past = self.previous_previous_teacher.solver.get_layer(l+1)[b]

        #                 # freeze bn
        #                 for m in module:
        #                     m.eval()
        #                 if self.task_count > 1:
        #                     for m in module_past:
        #                         m.eval()

        #                 current = module.forward(block_in)
        #                 current_mem = module.forward(block_mem)
        #                 if self.task_count > 1:
        #                     past = module_past.forward(block_in)
        #                     past_mem = module_past.forward(block_mem)

        #                 # generate metrics
        #                 metrics['CKA'][l] += 0
        #                 metrics_blockmem['CKA'][l] += 0
        #                 dist_match['CKA']['NormalG'][l] += 0
        #                 dist_match['CKA']['LearnedG'][l] += 0

        #                 metrics['MSE'][l] += 0
        #                 metrics_blockmem['MSE'][l] += 0
        #                 dist_match['MSE']['NormalG'][l] += 0
        #                 dist_match['MSE']['LearnedG'][l] += 0

        #                 metrics['MMD'][l] += 0
        #                 metrics_blockmem['MMD'][l] += 0
        #                 dist_match['MMD']['NormalG'][l] += 0
        #                 dist_match['MMD']['LearnedG'][l] += 0
        #                 # metrics use data from previous tasks (self.last_last_valid_out_dim), dist_match uses data from current task
        #                 # list.extend(metric.cpu().detach().numpy().tolist()))

                        
        #                 # unfreeze bn
        #                 for m in module:
        #                     m.train()

        #                 # increment block
        #                 bd_i += int(h/bd)

            
        # self.train(orig_mode)

        # # save results
        # savedir = topdir + name + '/block_metrics/'
        # if not os.path.exists(savedir): os.makedirs(savedir)
        # savename = savedir 
        # import yaml
        # for metric in ['CKA','MSE','MMD']:

        #     with open(savedir + 'raw-'+metric, 'w') as yaml_file:
        #         yaml.dump(metrics[metric], yaml_file, default_flow_style=False)
        #     with open(savedir + 'blockmem-'+metric, 'w') as yaml_file:
        #         yaml.dump(metrics_blockmem[metric], yaml_file, default_flow_style=False)
        #     with open(savedir + 'dist-'+metric, 'w') as yaml_file:
        #         yaml.dump(dist_match[metric], yaml_file, default_flow_style=False)



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