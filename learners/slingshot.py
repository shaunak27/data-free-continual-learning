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
from utils.visualization import tsne_eval, confusion_matrix_vis, pca_eval, tsne_eval_new, cka_plot, calculate_cka
from torch.optim import Optimizer
import contextlib
import os
from .slingshot_helper import Sampler
from .default import NormalNN, weight_reset, accumulate_acc, loss_fn_kd, Teacher
import copy
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.optim import SGD, Adam
from utils.schedulers import CosineSchedule
from utils.distance import MMD


class Slingshot(NormalNN):

    def __init__(self, learner_config):
        super(Slingshot, self).__init__(learner_config)
        self.inversion_replay = False
        self.previous_teacher = None
        self.dw = self.config['DW']
        self.device = 'cuda' if self.gpu else 'cpu'
        self.debug_dir = None
        self.power_iters = self.config['power_iters']
        self.deep_inv_params = self.config['deep_inv_params']
        self.kd_criterion = nn.MSELoss(reduction='none')
        
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
            
            try: 
                self.load_model(model_save_dir + '-halfway')
            except:
            
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

            # save the halfway point
            self.save_model(model_save_dir + '-halfway')

        if self.task_count > 0:

            # new teacher
            teacher = Teacher(solver=self.model)
            self.current_teacher = copy.deepcopy(teacher)

            # sample new synthetic coreset
            net_inv = Sampler(self.previous_teacher.solver, img_shape = (-1, train_dataset.nch,train_dataset.im_size, train_dataset.im_size),
                                iters = self.power_iters, deep_inv_params = self.deep_inv_params, class_idx = np.arange(self.last_valid_out_dim))
            x_replay, y_replay = net_inv.train_samples(copy.deepcopy(self.model), self.valid_out_dim)
            if self.debug_dir is not None: self.visualize_gen(x_replay, y_replay, self.debug_dir)

            # trains
            if need_train:

                # part b
                self.init_optimizer()

                # data weighting
                self.data_weighting(train_dataset)

                # Evaluate the performance of current task
                self.log('Distillation Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
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
                        if self.replay:
                            allowed_predictions = list(range(self.last_valid_out_dim))
                            y_hat, _ = self.previous_teacher.generate_scores(x_replay, allowed_predictions=allowed_predictions)
                            y_hat_b, _ = self.current_teacher.generate_scores(x[0], allowed_predictions=np.arange(self.valid_out_dim))

                        # model update - training data
                        loss, loss_class, loss_distill, output= self.update_model_b(x[0], y, y_hat_b, x_replay, y_replay, y_hat)

                        # measure elapsed time
                        batch_time.update(batch_timer.toc()) 

                        # measure accuracy and record loss
                        y = torch.cat([y,y_replay]).detach()
                        accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                        losses[0].update(loss,  y.size(0)) 
                        losses[1].update(loss_class,  y.size(0)) 
                        losses[2].update(loss_distill,  y.size(0)) 
                        batch_timer.tic()

                    # eval update
                    self.log('Distillation Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                    self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses[0],acc=acc))
                    self.log(' * Class Loss {loss.avg:.3f} | KD Loss {lossb.avg:.3f}'.format(loss=losses[1],lossb=losses[2]))

                    # Evaluate the performance of current task
                    if val_loader is not None:
                        self.validation(val_loader)

                    # reset
                    losses = [AverageMeter() for l in range(3)]
                    acc = AverageMeter()

        self.model.eval()
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False
        self.task_count += 1

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

    def update_model_b(self, inputs, targets, target_KD, inputs_B, targets_B, target_KD_B):

        dw_cls = self.dw_k[-1 * torch.ones((len(targets)*2,)).long()]
        dw_cls[:len(targets)] = (self.valid_out_dim-self.last_valid_out_dim) / self.valid_out_dim
        dw_cls[len(targets):] = self.last_valid_out_dim / self.valid_out_dim

        logits = self.forward(torch.cat([inputs, inputs_B]))
        # loss_class = self.criterion(logits, torch.cat([targets, targets_B]).long(), dw_cls)
        
        # classification 
        class_idx = np.arange(self.batch_size)

        # local classification
        loss_class = self.criterion(logits[:len(targets)][:,self.last_valid_out_dim:self.valid_out_dim], (targets-self.last_valid_out_dim).long(), dw_cls[:len(targets)]) 

        # ft classification               
        feat_class = self.model.forward(x=torch.cat([inputs, inputs_B]), pen=True).detach()
        loss_class += self.criterion(self.model.last(feat_class), torch.cat([targets, targets_B]).long(), dw_cls)
        total_loss = loss_class

        # KD
        dw_KD = self.dw_k[-1 * torch.ones(len(target_KD),).long()]
        loss_distill_a = loss_fn_kd(logits[:len(targets)][:, :self.valid_out_dim], target_KD, dw_KD, np.arange(self.valid_out_dim), self.DTemp)
        loss_distill_b = loss_fn_kd(logits[len(targets):][:, :self.last_valid_out_dim], target_KD_B, dw_KD, np.arange(self.last_valid_out_dim), self.DTemp)
        loss_distill = ((self.valid_out_dim-self.last_valid_out_dim) / self.valid_out_dim) * loss_distill_a + (self.last_valid_out_dim / self.valid_out_dim) * loss_distill_b
        total_loss += self.mu * loss_distill

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_distill.detach(), logits

    def visualize_gen(self, x, y, topdir):

        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        # this way normalizes per image
        savedir = topdir + '/images/'
        if not os.path.exists(savedir): os.makedirs(savedir)  
        num_show = 50
        cols = 10
        rows = math.ceil(num_show/cols)
        for j in range(num_show):
            plt.subplot(rows, cols, j + 1)
            im_show = np.squeeze(x[j].transpose((1,2,0)))
            im_min = np.amin(im_show)
            im_max = np.amax(im_show)
            im_show = (im_show - im_min) / ((im_max - im_min))
            plt.imshow((im_show * 255).astype(np.uint8)); plt.axis('off')
            plt.title(str(y[j]),fontsize=6)
        save_name = savedir + 'synthetic_images.png'
        plt.savefig(save_name) 
        plt.close()


def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).cuda()
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot