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
import scipy as sp
import scipy.linalg as linalg
from models.layers import CosineScaling
from models.resnet import BiasLayer, BiasModel
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

# Code from https://github.com/jlezama/OrthogonalLowrankEmbedding/blob/master/pytorch_OLE/OLE.py
# Lezama et al 2017

import torch
from torch.autograd import Variable, Function

import numpy as np
import scipy as sp
import scipy.linalg as linalg

def power_iter(model, x, y, dim, n):
    inputs = torch.randn((len(x),64), requires_grad = True, device = "cuda")
    criterion = nn.CrossEntropyLoss()
    model = model.cuda()
    optimizer_di = torch.optim.Adam([inputs], lr=0.1)
    for i in range(n):
        optimizer_di.zero_grad()
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs[:,dim[0]:dim[1]], y.long() - dim[0])
        loss.backward()
        optimizer_di.step()

    return x * 0 + inputs.detach()

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

# Inherit from Function
class OLELoss(Function):

    @staticmethod
    def forward(ctx, X, y, lambda_=0.25):
        classes = torch.unique(y)
        N, D = X.shape
        DELTA = 1.
        
        # gradients initialization
        Obj_c = 0
        dX_c = torch.zeros((N, D), device=X.device)
        Obj_all = 0;
        dX_all = torch.zeros((N,D), device=X.device)
        eigThd = 1e-6 # threshold small eigenvalues for a better subgradient

        # compute objective and gradient for first term \sum ||TX_c||*
        for c in classes:
            A = X[y==c,:]
            # SVD
            U, S, V = torch.linalg.svd(A.float(), full_matrices=False)
            V = V.T
            nuclear = S.sum()

            ## L_c = max(DELTA, ||TY_c||_*)-DELTA
            if nuclear>DELTA:
              Obj_c += nuclear;
              # discard small singular values
              r = torch.sum(S<eigThd)

              uprod = U[:,0:U.shape[1]-r] @ V[:,0:V.shape[1]-r].T
              dX_c[y==c,:] += uprod
            else:
              Obj_c+= DELTA

        # compute objective and gradient for secon term ||TX||*
        #U, S, V = sp.linalg.svd(X.detach().cpu().numpy(), full_matrices=False)
        U, S, V = torch.linalg.svd(X.float(), full_matrices=False)
        V = V.T
        Obj_all = torch.sum(S)
        r = torch.sum(S<eigThd)
        uprod = U[:,0:U.shape[1]-r] @ V[:,0:V.shape[1]-r].T
        dX_all = uprod

        obj = (Obj_c  - lambda_*Obj_all)/N
        dX = (dX_c  - lambda_*dX_all)/N
        ctx.save_for_backward(dX)

        return obj

    @staticmethod
    def backward(ctx, grad_output):
        # print self.dX
        result, = ctx.saved_tensors
        return result, None, None

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

KL = False
class LWF(NormalNN):

    def __init__(self, learner_config):
        super(LWF, self).__init__(learner_config)
        self.previous_teacher = None
        self.replay = False
        self.past_tasks = []
        self.bic_layers = None
        self.ete_flag = False
        self.first_task = True
        
    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def validation(self, dataloader, model=None, task_in = None, task_metric='acc', relabel_clusters = True, verbal = True, filter_past = False, scale=False):

        if model is None:
            if task_metric == 'acc':
                model = self.model
            # elif task_metric == 'aux_task':
            #     return self.aux_task(dataloader)
            else:
                return -1
        if True:
            # This function doesn't distinguish tasks.
            batch_timer = Timer()
            acc = AverageMeter()
            batch_timer.tic()

            orig_mode = model.training
            model.eval()
            for i, (input, target, task) in enumerate(dataloader):

                if self.gpu:
                    with torch.no_grad():
                        input = input.cuda()
                        target = target.cuda()

                if filter_past:
                    mask = target < self.last_valid_out_dim
                    mask_ind = mask.nonzero().view(-1) 
                    input = input[mask_ind]
                    target = target[mask_ind]
                if len(target) > 1 or not filter_past:
                    if task_in is None:
                        output = model.forward(input)[:, :self.valid_out_dim]
                        acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
                    else:
                        mask = target >= task_in[0]
                        mask_ind = mask.nonzero().view(-1) 
                        input, target = input[mask_ind], target[mask_ind]
                        if len(mask_ind) > 0:

                            mask = target < task_in[-1]
                            mask_ind = mask.nonzero().view(-1) 
                            input, target = input[mask_ind], target[mask_ind]
                            if len(mask_ind) > 0:

                                output = model.forward(input)[:, task_in]
                                acc = accumulate_acc(output, target-task_in[0], task, acc, topk=(self.top_k,))

            if verbal:
                self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                        .format(acc=acc, time=batch_timer.toc()))

        else:
            for scale in [5, 6, 7, 8, 9, 10]:
                # This function doesn't distinguish tasks.
                batch_timer = Timer()
                acc = AverageMeter()
                batch_timer.tic()

                orig_mode = model.training
                model.eval()
                for i, (input, target, task) in enumerate(dataloader):

                    if self.gpu:
                        with torch.no_grad():
                            input = input.cuda()
                            target = target.cuda()

                    if filter_past:
                        mask = target < self.last_valid_out_dim
                        mask_ind = mask.nonzero().view(-1) 
                        input = input[mask_ind]
                        target = target[mask_ind]
                    if len(target) > 1 or not filter_past:
                        output = model.forward(input)[:, :self.valid_out_dim]
                        output[:,:self.last_valid_out_dim] = output[:,:self.last_valid_out_dim] + scale
                        acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
                
                if verbal:
                    self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                            .format(acc=acc, time=batch_timer.toc()))
        model.train(orig_mode)
        return acc.avg

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

            # # JAMES HERE
            if not self.first_task:
                self.model = BiasModel(self.model)

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
                self.validation(val_loader, scale=True)
                self.validation(val_loader, filter_past = True, scale=True)
        
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
                self.log(' * Class Loss {loss.avg:.3f} | Dist Loss {lossb:.3f}| KD Loss {lossc.avg:.3f}'.format(loss=losses[1], lossb=losses[0].avg-losses[1].avg-losses[2].avg, lossc=losses[2]))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)
                    self.validation(val_loader, filter_past = True)

                # reset
                losses = [AverageMeter() for l in range(3)]
                acc = AverageMeter()

        self.model.eval()
        self.KD = True

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

class LWF_FR(LWF):

    def __init__(self, learner_config):
        super(LWF_FR, self).__init__(learner_config)
        self.kd_criterion = nn.MSELoss(reduction='none')

    def update_model(self, inputs, targets, target_KD = None):

        if self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        logits = self.forward(inputs)
        loss_class = self.criterion(logits, targets.long(), dw_cls)
        total_loss = loss_class

        # KD
        if self.KD and target_KD is not None:
            kd_mask = targets < self.last_valid_out_dim
            kd_ind = kd_mask.nonzero().view(-1) 
            current_pen = self.model.forward(x=inputs[kd_ind], pen=True)
            past_pen = self.previous_teacher.solver.forward(x=inputs[kd_ind], pen=True)
            loss_distill = self.kd_criterion(current_pen,past_pen).mean()
            total_loss += self.mu * loss_distill
        else:
            loss_distill = torch.zeros((1,), requires_grad=True).cuda()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_distill.detach(), logits

class LWF_FRB_DF(LWF):

    def __init__(self, learner_config):
        super(LWF_FRB_DF, self).__init__(learner_config)
        self.oc_criterion = OLELoss.apply
        self.kd_criterion = nn.MSELoss()
        self.mmd_loss = MMD_loss()

    def update_model(self, inputs, targets, target_KD = None):

        if self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        
        # freeze layer
        l_freeze = 2

        # dirty freezing
        if not self.first_task:
            # JAMES HERE
            past_pen, bd = self.previous_teacher.solver.forward(x=inputs, pen=True, div = True)
            # past = past_pen[l_freeze].detach()
            # for l in range(l_freeze,3):
            #     past = self.model.get_layer_forward(l+1,past.view(self.previous_teacher.solver.size_array[l]))
            # logits = self.model.last.forward(past.view(past.size(0),-1))[:, :self.valid_out_dim]
            # loss_class = self.n_task_class_loss(inputs, logits, past, targets, target_KD, dw_cls)
            loss_class = self.n_task_class_loss(inputs, -1, -1, targets, -1, dw_cls)
        
        else:
            logits = self.forward(inputs)
            loss_class = self.criterion(logits, targets.long(), dw_cls)

        # Constrain to distribution
        loss_dist = torch.zeros((1,), requires_grad=True).cuda()
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        current_pen, bd = self.model.forward(x=inputs, pen=True, div = True)
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
                if l >= l_freeze:
                    loss_dist = self.dist_loss(block_out) + loss_dist

                if l >= l_freeze + 1:
                    # second loss is sampling, after first task
                    if not self.first_task:
                        block_in = torch.randn(self.model.size_array[l-1]).cuda()
                        current = self.model.get_layer(l)[b].forward(block_in)
                        past = self.previous_teacher.solver.get_layer(l)[b].forward(block_in)
                        loss_kd = self.kd_criterion(current.view(current.size(0), -1), past.view(past.size(0), -1)).mean() + loss_kd
  
                # save output as input to next layer
                next_layer_in[b] = block_out.size()

                # increment block
                bd_i += int(h/bd)

        # total loss
        loss_dist = loss_dist / (bd * nl)
        loss_kd = self.mu * loss_kd / (bd * nl)

        # scaling
        loss_dist = loss_dist * 1e-1

        if self.first_task:
            total_loss = loss_class# + loss_dist
        else:
            total_loss = loss_class# + loss_kd

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # JAMES HERE
        with torch.no_grad():
            logits = self.forward(inputs)
        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits 


    def n_task_class_loss(self, inputs, logits, past, targets, target_KD, dw_cls):
        target = get_one_hot(targets, self.valid_out_dim)
        target_KD = F.softmax(target_KD, dim=1)
        target[:, :self.last_valid_out_dim] = target_KD

        log_logits= F.log_softmax(logits, dim=1)

        KD_loss_unnorm = -(target * log_logits)
        KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)
        loss_class = KD_loss_unnorm.mean()
        return loss_class

    def dist_loss(self,block_out):

        block_target = torch.randn(block_out.size()).cuda()
        loss_dist = self.mmd_loss(block_out,block_target)
        return loss_dist

class LWF_FRB_DF_MMD(LWF_FRB_DF):

    def __init__(self, learner_config):
        super(LWF_FRB_DF_MMD, self).__init__(learner_config)

class LWF_FRB_DF_LOCAL(LWF_FRB_DF):

    def __init__(self, learner_config):
        super(LWF_FRB_DF_LOCAL, self).__init__(learner_config)

    # def n_task_class_loss(self, inputs, logits, past, targets, target_KD, dw_cls):
        
    #     loss_class = self.criterion(logits[:,self.last_valid_out_dim:self.valid_out_dim], targets.long() - self.last_valid_out_dim, dw_cls)
        
    #     # ft loss
    #     x = past.view(past.size(0),-1).detach() + 0
    #     y = torch.randint(low=0, high=self.last_valid_out_dim, size=(len(inputs),)).cuda()
    #     x = power_iter(copy.deepcopy(self.previous_teacher.solver.last), x, y, [0, self.last_valid_out_dim], 10)
    #     # xb = torch.randn((len(inputs),64), requires_grad = True).cuda()
    #     # yb = torch.randint(low=self.last_valid_out_dim, high=self.valid_out_dim, size=(len(inputs),)).cuda()
    #     # xb = power_iter(copy.deepcopy(self.model.last), xb, yb, [self.last_valid_out_dim, self.valid_out_dim], 10)
    #     xb = past.view(past.size(0),-1).detach() + 0
    #     yb = targets

    #     x_com = torch.cat([x, xb])
    #     logits_com = self.model.last.forward(x_com)[:,:self.valid_out_dim]
    #     targets_com = torch.cat([y, yb])
    #     loss_class = loss_class + self.criterion(logits_com, targets_com, self.dw_k[-1 * torch.ones(targets_com.size()).long()])

    #     loss_class = loss_class + self.kd_criterion(self.previous_teacher.solver.last(past.view(past.size(0),-1))[:,:self.last_valid_out_dim], target_KD[:,:self.last_valid_out_dim]).mean()
    #     return loss_class

    def n_task_class_loss(self, inputs, logits, past, targets, target_KD, dw_cls):
        
        # ft loss
        x = torch.randn((len(inputs),64), requires_grad = True).cuda()
        y = torch.randint(low=0, high=self.last_valid_out_dim, size=(len(inputs),)).cuda()
        # x = power_iter(copy.deepcopy(self.previous_teacher.solver.last), x, y, [0, self.last_valid_out_dim], 10)
        xb = torch.randn((len(inputs),64), requires_grad = True).cuda()
        yb = torch.randint(low=self.last_valid_out_dim, high=self.valid_out_dim, size=(len(inputs),)).cuda()
        # xb = power_iter(copy.deepcopy(self.previous_teacher.solver.last), xb, yb, [self.last_valid_out_dim, self.valid_out_dim], 10)

        x_com = torch.cat([x, xb])
        # x_com = self.model.forward(inputs, pen=True)
        logits_com = self.model.logits(self.model.features.last.forward(x_com).detach())[:,:self.valid_out_dim]
        targets_com = torch.cat([y, yb])
        #targets_com = targets
        loss_class = self.criterion(logits_com, targets_com, self.dw_k[-1 * torch.ones(targets_com.size()).long()])

        # target_KD, _ = self.previous_teacher.generate_scores(inputs, allowed_predictions=np.arange(self.valid_out_dim))
        # kd_logits_a = self.previous_teacher.solver.last(past.view(past.size(0),-1))[:,:self.last_valid_out_dim]
        # kd_logits_b = self.previous_teacher.solver.last(past.view(past.size(0),-1))[:,self.last_valid_out_dim:self.valid_out_dim]
        # loss_class = loss_class + loss_fn_kd(kd_logits_a, target_KD[:,:self.last_valid_out_dim], dw_cls, np.arange(self.last_valid_out_dim).tolist(), self.DTemp)
        # loss_class = loss_class + loss_fn_kd(kd_logits_b, target_KD[:,self.last_valid_out_dim:self.valid_out_dim], dw_cls, np.arange(self.valid_out_dim - self.last_valid_out_dim).tolist(), self.DTemp)
        return loss_class

class LWF_FRB_DF_MMD_COV(LWF_FRB_DF):

    def __init__(self, learner_config):
        super(LWF_FRB_DF_MMD_COV, self).__init__(learner_config)

    def dist_loss(self,block_out):
        block_cov = cov(block_out)
        loss_dist = torch.pow(block_out.sum(dim=0),2).mean() + torch.pow(block_cov - torch.eye(block_cov.size(0)).cuda(),2).mean()
        return loss_dist


class LWF_FRB_DF_MMD_VAR(LWF_FRB_DF):

    def __init__(self, learner_config):
        super(LWF_FRB_DF_MMD_VAR, self).__init__(learner_config)

    def dist_loss(self,block_out):
        loss_dist = torch.pow(block_out.mean(dim=0),2).mean() + torch.pow(block_out.var(dim=0),2).mean()
        return loss_dist

class LWF_FRB_DF_MMD_PRO(LWF_FRB_DF):

    def __init__(self, learner_config):
        super(LWF_FRB_DF_MMD_PRO, self).__init__(learner_config)

    def n_task_class_loss(self, inputs, logits, past, targets, target_KD, dw_cls):
            
        # loss class
        logits_append = []
        targets_append = []
        npc = int(len(targets) / (self.valid_out_dim - self.last_valid_out_dim))
        for c in range(self.last_valid_out_dim):
            targets_append.append(c * torch.ones((npc,)).cuda())
            logits_to_append = (torch.randn((npc,64), requires_grad = True).cuda() - 0.5)*1e-0 + self.previous_teacher.solver.last.weight.data[c]
            # logits_to_append = self.model.last.forward(logits_to_append.view(logits_to_append.size(0),-1))
            logits_append.append(logits_to_append)
        
        logits_append = torch.cat(logits_append)
        logits_append_out = self.model.last.forward(logits_append.view(logits_append.size(0),-1))
        targets_append = torch.cat(targets_append)
        loss_class = self.criterion(torch.cat([logits, logits_append_out]), torch.cat([targets.long(), targets_append.long()]), torch.ones((len(targets)+len(targets_append),)).cuda())
        loss_class = loss_class + self.kd_criterion(logits_append_out[:self.last_valid_out_dim], self.previous_teacher.solver.last.forward(logits_append.view(logits_append.size(0),-1))[:self.last_valid_out_dim]).mean()
        return loss_class

























































class LWF_FRB(LWF):

    def __init__(self, learner_config):
        super(LWF_FRB, self).__init__(learner_config)
        self.kd_criterion = nn.MSELoss(reduction='none')

    def update_model(self, inputs, targets, target_KD = None):

        if self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        logits = self.forward(inputs)
        loss_class = self.criterion(logits, targets.long(), dw_cls)
        total_loss = loss_class

        # KD
        if self.KD and target_KD is not None:

            kd_mask = targets < self.last_valid_out_dim
            kd_ind = kd_mask.nonzero().view(-1) 
            current_pen, bd = self.model.forward(x=inputs[kd_ind], pen=True, div = True)
            past_pen, bd = self.previous_teacher.solver.forward(x=inputs[kd_ind], pen=True, div = True)
            loss_distill = 0
            nl = len(current_pen)
            for l in range(nl):
                bd_i = 0
                h = len(current_pen[l][0])
                for b in range(bd):
                    loss_distill = self.kd_criterion(current_pen[l][:,bd_i:bd_i+int(h/bd)],past_pen[l][:,bd_i:bd_i+int(h/bd)]).mean() / bd + loss_distill
                    bd_i += int(h/bd)
            total_loss += self.mu * loss_distill / nl
        else:
            loss_distill = torch.zeros((1,), requires_grad=True).cuda()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_distill.detach(), logits

class LWF_FRB_ABLATE(LWF):

    def __init__(self, learner_config):
        super(LWF_FRB_ABLATE, self).__init__(learner_config)
        self.kd_criterion = nn.MSELoss(reduction='none')

    def update_model(self, inputs, targets, target_KD = None):

        if self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        logits = self.forward(inputs)
        loss_class = self.criterion(logits, targets.long(), dw_cls)
        total_loss = loss_class

        # KD
        if self.KD and target_KD is not None:

            kd_mask = targets < self.last_valid_out_dim
            kd_ind = kd_mask.nonzero().view(-1) 
            current_pen, bd = self.model.forward(x=inputs[kd_ind], pen=True, div = True)
            past_pen, bd = self.previous_teacher.solver.forward(x=inputs[kd_ind], pen=True, div = True)
            loss_distill = 0
            nl = len(current_pen)
            for l in range(nl):
                h = len(current_pen[l][0])
                loss_distill = self.kd_criterion(current_pen[l],past_pen[l]).mean() + loss_distill
            total_loss += self.mu * loss_distill / nl
        else:
            loss_distill = torch.zeros((1,), requires_grad=True).cuda()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_distill.detach(), logits

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