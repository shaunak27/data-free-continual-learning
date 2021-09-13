from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import libmr
import numpy as np
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

KL = False
class LWF(NormalNN):

    def __init__(self, learner_config):
        super(LWF, self).__init__(learner_config)
        self.previous_teacher = None
        self.replay = False
        self.past_tasks = []
        self.bic_layers = None
        self.ete_flag = False
        
    ##########################################
    #           MODEL TRAINING               #
    ##########################################

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

        # KD
        if self.KD and target_KD is not None:
            target = get_one_hot(targets, self.valid_out_dim)
            
            target_KD = F.softmax(target_KD, dim=1)
            target[:, :self.last_valid_out_dim] = target_KD

            log_logits= F.log_softmax(logits, dim=1)

            KD_loss_unnorm = -(target * log_logits)
            KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)
            total_loss= KD_loss_unnorm.mean()
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
            total_loss = self.criterion(logits, targets.long(), dw_cls)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), total_loss.detach(), torch.zeros((1,), requires_grad=True).cuda().detach(), logits

class ETE(LWF):

    def __init__(self, learner_config):
        super(ETE, self).__init__(learner_config)
        self.ete_flag = True

    def update_model(self, inputs, targets, target_KD = None):

        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]

        logits = self.forward(inputs)
        loss_class = self.criterion(logits, targets.long(), dw_cls)
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

    def update_model_b(self, inputs, targets, target_KD = None, target_KD_B = None):

        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]

        logits = self.forward(inputs)
        loss_class = self.criterion(logits, targets.long(), dw_cls)
        total_loss = loss_class

        # KD
        if self.KD and target_KD is not None:

            dw_KD = self.dw_k[-1 * torch.ones(len(target_KD),).long()]
            logits_KD = logits
            for task_l in self.past_tasks:
                loss_distill = loss_fn_kd(logits_KD, target_KD, dw_KD, task_l.tolist(), self.DTemp)
                total_loss += self.mu * loss_distill * (len(task_l) / self.valid_out_dim)

            # current task
            loss_distill = loss_fn_kd(logits_KD[:, self.last_valid_out_dim:self.valid_out_dim], target_KD, dw_KD, np.arange(self.valid_out_dim-self.last_valid_out_dim), self.DTemp)
            total_loss += self.mu * loss_distill * ((self.valid_out_dim-self.last_valid_out_dim) / self.valid_out_dim)

        else:
            loss_distill = torch.zeros((1,), requires_grad=True).cuda()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_distill.detach(), logits

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        if self.task_count == 0:
            return super(ETE, self).learn_batch(train_loader, train_dataset, model_save_dir, val_loader)

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

        # new teacher
        teacher = Teacher(solver=self.model)
        self.current_teacher = copy.deepcopy(teacher)

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset_ete(self.memory_size, np.arange(self.valid_out_dim), teacher)

        # trains
        if need_train:

            # part b
            # dataset tune
            train_dataset.load_dataset(train_dataset.t, train=True)
            train_dataset.append_coreset(only=True)

            self.config['lr'] = self.config['lr'] / 1e2
            if len(self.config['gpuid']) > 1:
                self.optimizer, self.scheduler = self.new_optimizer(self.model.module.last)
            else:
                self.optimizer, self.scheduler = self.new_optimizer(self.model.last)
            self.config['lr'] = self.config['lr'] * 1e2

            # Evaluate the performance of current task
            self.log('Balance Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
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
                        y_hat_b, _ = self.current_teacher.generate_scores(x[0], allowed_predictions=np.arange(self.last_valid_out_dim, self.valid_out_dim))
                    else:
                        y_hat = None

                    # model update - training data
                    loss, loss_class, loss_distill, output= self.update_model_b(x[0], y, y_hat, y_hat_b)

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
                self.log('Balanced Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses[0],acc=acc))
                self.log(' * Class Loss {loss.avg:.3f} | KD Loss {lossb.avg:.3f}'.format(loss=losses[1],lossb=losses[2]))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = [AverageMeter() for l in range(3)]
                acc = AverageMeter()

            # dataset final
            train_dataset.load_dataset(train_dataset.t, train=True)
            train_dataset.append_coreset(only=False)

        self.model.eval()
        self.past_tasks.append(np.arange(self.last_valid_out_dim,self.valid_out_dim))
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        ## for eval
        if self.previous_teacher is not None:
            self.previous_previous_teacher = self.previous_teacher

        # new teacher
        teacher = Teacher(solver=self.model)
        self.previous_teacher = copy.deepcopy(teacher)
        self.replay = True

        try:
            return batch_time.avg
        except:
            return None

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


def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).cuda()
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot