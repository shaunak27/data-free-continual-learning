from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from models.layers import CosineScaling
from utils.visualization import tsne_eval, confusion_matrix_vis, pca_eval, tsne_eval_new, cka_plot, calculate_cka
from torch.optim import Optimizer
import contextlib
import os
from .deep_inv_helper_gen import Teacher
from .default import NormalNN, weight_reset, accumulate_acc, loss_fn_kd
import copy
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.optim import SGD, Adam
from utils.schedulers import CosineSchedule

FREEZE_BN = False

# HERE JAMES
AUX_TASK = False
# STEPS_SAVE = [1, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000]
STEPS_SAVE = [1, 100, 500, 1000, 4000, 10000]

class DeepInversionGenBN(NormalNN):

    def __init__(self, learner_config):
        super(DeepInversionGenBN, self).__init__(learner_config)
        self.inversion_replay = False
        self.previous_teacher = None
        self.dw = self.config['DW']
        self.device = 'cuda' if self.gpu else 'cpu'
        self.debug_dir = None
        self.aux_task_complete = False
        self.power_iters = self.config['power_iters']
        self.deep_inv_params = self.config['deep_inv_params']
        self.kd_criterion = nn.MSELoss(reduction='none')
        self.refresh_iters = self.config['refresh_iters']
        self.balanced_linear = None
        self.bic = False

        # upper bound dictionary to hold past task data
        self.upper_bound = {}
        self.upper_bound_flag = self.config['upper_bound_flag']
        self.cp = self.deep_inv_params[7]
        self.playground_flag = self.config['playground_flag']

        # gen parameters
        self.generator = self.create_generator()
        self.generator_optimizer = Adam(params=self.generator.parameters(), lr=self.deep_inv_params[1])
        # self.generator_optimizer = self.new_optimizer(self.generator)
        self.beta = self.config['beta']

        self.cka_holder = [[],[], []]

        # repeat call for generator network
        if self.gpu:
            self.cuda_gen()
        
    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        self.pre_steps()

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

            # visualize
            if self.inversion_replay:
                self.epoch = 0
                if (self.epoch) % self.refresh_iters == 0 or self.epoch == 0:
                    if self.debug_dir is not None:
                        x_replay, y_replay, y_replay_hat = self.sample(self.previous_teacher, self.model, 2*self.batch_size, self.device, True, step=True)
                        if self.inversion_replay: self.visualize_gen(val_loader, self.debug_dir, "inversions", gen_images = (x_replay, y_replay), gen_text = str(0), show_change=True)

            losses = [AverageMeter() for i in range(3)]
            acc = AverageMeter()
            accg = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            self.save_gen = False
            self.save_gen_later = False
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

                    # update style images
                    if len(x[0]) == self.batch_size: self.style_imgs = x[0]

                    # data replay
                    if self.inversion_replay:
                        x_replay, y_replay, y_replay_hat = self.sample(self.previous_teacher, self.model, len(x[0]), self.device, True, step=True)

                    # if KD
                    if self.inversion_replay:
                        y_hat = self.previous_teacher.generate_scores(x[0], allowed_predictions=np.arange(self.last_valid_out_dim))
                        _, y_hat_com = self.combine_data(((x[0], y_hat),(x_replay, y_replay_hat)))
                    else:
                        y_hat_com = None

                    # combine inputs and generated samples for classification
                    if self.inversion_replay:
                        x_com, y_com = self.combine_data(((x[0], y),(x_replay, y_replay)))
                    else:
                        x_com, y_com = x[0], y

                    # sd data weighting (NOT online learning compatible)
                    if self.dw:
                        dw_cls = self.dw_k[y_com.long()]
                    else:
                        #dw_cls = self.dw_k[-1 * torch.ones(y_com.size()).long()]

                        # mappings = torch.ones(y_com.size(), dtype=torch.float32)
                        # if self.gpu:
                        #     mappings = mappings.cuda()
                        # rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
                        # mappings[:self.last_valid_out_dim] = rnt
                        # mappings[self.last_valid_out_dim:] = 1-rnt
                        # dw_cls = mappings[y_com.long()]
                        dw_cls = None

                    # freeze bn
                    if self.task_count > 0 and FREEZE_BN:
                        for name ,child in (self.model.named_children()):
                            if isinstance(child, nn.BatchNorm2d):
                                child.eval()

                    # model update
                    loss, loss_class, loss_kd, output= self.update_model(x_com, y_com, y_hat_com, dw_force = dw_cls, kd_index = np.arange(len(x[0]), len(x_com)))

                    # update stats
                    self.update_stats(x[0], y)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc()) 

                    # measure accuracy and record loss
                    y_com = y_com.detach()
                    accumulate_acc(output[:self.batch_size], y_com[:self.batch_size], task, acc, topk=(self.top_k,))
                    if self.inversion_replay: accumulate_acc(output[self.batch_size:], y_com[self.batch_size:], task, accg, topk=(self.top_k,))
                    losses[0].update(loss,  y_com.size(0)) 
                    losses[1].update(loss_class,  y_com.size(0))
                    losses[2].update(loss_kd,  y_com.size(0))
                    batch_timer.tic()

                # eval update
                self.swap_model(True)

                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | CE Loss {lossb.avg:.3f} | KD Loss {lossc.avg:.3f} | Conf Loss {lossd:.3f}'.format(loss=losses[0],lossb=losses[1],lossc=losses[2], lossd=losses[0].avg-losses[1].avg-losses[2].avg))
                self.log(' * Train Acc {acc.avg:.3f} | Train Acc Gen {accg.avg:.3f}'.format(acc=acc,accg=accg))

                # Evaluate the performance of current task
                # print(' * All')
                if val_loader is not None:
                    self.validation(val_loader)
                    self.validation(val_loader, filter_past = True)
                # print(' * Past Tasks')
                # if val_loader is not None:
                #     self.validation(val_loader, filter_past = True)

                # reset
                losses = [AverageMeter() for i in range(3)]
                acc = AverageMeter()
                accg = AverageMeter()

                self.swap_model(False)

                # visualize
                if self.inversion_replay:
                    if (self.epoch+1) % self.refresh_iters == 0 and self.epoch > 0:
                        if self.debug_dir is not None:
                            x_replay, y_replay, y_replay_hat = self.sample(self.previous_teacher, self.model, 2*self.batch_size, self.device, True, step=True)
                            if self.inversion_replay: self.visualize_gen(val_loader, self.debug_dir, "inversions", gen_images = (x_replay, y_replay), gen_text = str(self.epoch + 1), show_change=True)

        if self.inversion_replay and self.debug_dir is not None: self.visualize_gen(val_loader, self.debug_dir, "final")

        self.model.eval()
        self.last_last_valid_out_dim = self.last_valid_out_dim
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # for eval
        if self.previous_teacher is not None:
            self.previous_previous_teacher = self.previous_teacher

        # HERE JAMES
        if (self.out_dim == self.valid_out_dim): need_train = False
        
        # new teacher
        # self.generator = self.create_generator()
        self.previous_teacher = Teacher(solver=copy.deepcopy(self.model), generator=self.generator, gen_opt = self.generator_optimizer, img_shape = (-1, train_dataset.nch,train_dataset.im_size, train_dataset.im_size), iters = self.power_iters, deep_inv_params = self.deep_inv_params, class_idx = np.arange(self.valid_out_dim), var = self.variance_torch, mean = self.centers_torch, train = need_train, config = self.config)
        self.sample(self.previous_teacher, None, self.batch_size, self.device, False)
        self.inversion_replay = True
        if len(self.config['gpuid']) > 1:
            self.previous_linear = copy.deepcopy(self.model.module.last)
        else:
            self.previous_linear = copy.deepcopy(self.model.last)
        self.upper_bound_past = copy.deepcopy(self.upper_bound)
        # self.centers_torch_past = copy.deepcopy(self.centers_torch)
        # self.variance_torch_past = copy.deepcopy(self.variance_torch)

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            if self.bic:
                train_dataset.update_coreset_ic(self.memory_size, np.arange(self.last_valid_out_dim), self.previous_teacher)
            else:
                train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        self.swap_model(True)

        try:
            return batch_time.avg
        except:
            return None

    def visualization(self, dataloader, topdir, name, task, embedding):
        # # save generated images
        # self.debug_dir = topdir
        # if self.inversion_replay: self.visualize_gen(dataloader, topdir, name)
        #
        # tsne
        #
        # setup files
        savedir = topdir + name + '/'
        if not os.path.exists(savedir): os.makedirs(savedir)
        savename = savedir 

        # gather data
        X = []
        y_pred = []
        y_true = []
        y_act = [[] for k in range(self.valid_out_dim)]
        logit_out = [[] for k in range(self.valid_out_dim)]
        orig_mode = self.training
        self.eval()
        for i, (input, target, _) in enumerate(dataloader):
            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            output = self.predict(input)
            penultimate = self.model.forward(x=input, pen=True)
            X.extend(penultimate.cpu().detach().tolist())
            y_pred.extend(np.argmax(output.cpu().detach().numpy(), axis = 1).tolist())
            y_true.extend(target.cpu().detach())
            for k in range(self.valid_out_dim):
                y_act[k].extend(output[:,k].cpu().detach().numpy().tolist())
                y_ind = np.where(target.cpu().detach().numpy() == k)[0]
                logit_out[k].extend(output[y_ind,:].cpu().detach().numpy().tolist())
        self.train(orig_mode)

        # # debug - print activation stats
        # logit_out = [np.asarray(logit_out[k]) for k in range(self.valid_out_dim)]
        # print('acutal')
        # for k in range(self.valid_out_dim):
        #     print(np.mean(logit_out[k], axis=0))
        #     print(np.std(logit_out[k], axis=0))
        #     print(np.cov(logit_out[k].transpose()))

        # save activations
        y_act = np.asarray(y_act)
        cm_array = np.zeros((self.valid_out_dim,2))
        cm_array[:,0] = np.mean(y_act, axis=1)
        cm_array[:,1] = np.std(y_act, axis=1)
        np.savetxt(savename+'mean_std_act_per_task.csv', cm_array, delimiter=",", fmt='%.0f')

        # convert to arrays
        X = np.asarray(X)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # confusion matrix
        title = 'CM - Task ' + str(task+1)
        ood_cuttoff = self.valid_out_dim
        confusion_matrix_vis(y_pred[y_true < ood_cuttoff], y_true[y_true < ood_cuttoff], savename, title)

        # tsne for in and out of distribution data
        title = 'TSNE - Task ' + str(task+1)
        if self.centers_torch is None:
            tsne_eval(X[y_true < ood_cuttoff], y_true[y_true < ood_cuttoff], savename, title, self.out_dim)
        else:
            tsne_eval(X[y_true < ood_cuttoff], y_true[y_true < ood_cuttoff], savename, title, self.out_dim, clusters = self.centers_torch.cpu().detach().numpy())


        # pca for in and out of distribution data
        title = 'PCA - Task ' + str(task+1)
        embedding = pca_eval(X[y_true < ood_cuttoff], y_true[y_true < ood_cuttoff], savename, title, self.out_dim, embedding)

        if self.task_count > 1 and self.task_count < 3:
            
            sample_ind = np.random.choice(len(X), self.batch_size * 2, replace=False)
            X = X[sample_ind]; y_pred = y_pred[sample_ind]; y_true = y_true[sample_ind]

            # extend
            X = np.concatenate((X, self.x_gen), axis=0)
            y_true = np.concatenate((y_true, self.y_gen), axis=0)
            y_pred = np.concatenate((y_pred, self.y_gen_pred), axis=0)


            title = 'tasks'
            markers = ['Current Task - Real','','', 'Previous Task - Real', 'Previous Task - Synthesized']
            tsne_text = ''
            num_color = 3

            # to and from tsne
            from tsne import bh_sne
            X = X[y_true < ood_cuttoff]; y_pred = y_pred[y_true < ood_cuttoff]; y_true = y_true[y_true < ood_cuttoff]
            X = bh_sne(X.astype('float64'))
            X[:,0] = (X[:,0] - min(X[:,0])) / (max(X[:,0]) - min(X[:,0]))
            X[:,1] = (X[:,1] - min(X[:,1])) / (max(X[:,1]) - min(X[:,1]))

            # Gen
            len_gen = len(y_true[y_true < 0])
            data_gen = [X[y_true < 0], -1*y_true[y_true < 0], 4*np.ones(len_gen), y_pred[y_true < 0]]

            # past
            X = X[y_true >= 0]; y_pred = y_pred[y_true >= 0]; y_true = y_true[y_true >= 0]
            len_past = len(y_true[y_true < self.last_last_valid_out_dim])
            data_past= [X[y_true < self.last_last_valid_out_dim], y_true[y_true < self.last_last_valid_out_dim], 3*np.ones(len_past), y_pred[y_true < self.last_last_valid_out_dim]]

            # current
            len_current = len(y_true[y_true >= self.last_last_valid_out_dim])
            data_current = [X[y_true >= self.last_last_valid_out_dim], y_true[y_true >= self.last_last_valid_out_dim], 0*np.ones(len_current), y_pred[y_true >= self.last_last_valid_out_dim]]

            # acutal tsne
            savedir = topdir + name + '/tsne/'
            savedirb = savedir + 'class-color/'
            savedirc = savedir + 'errors/'
            savedir = savedir + 'standard-color/'
            if not os.path.exists(savedir): os.makedirs(savedir)  
            if not os.path.exists(savedirb): os.makedirs(savedirb)   
            if not os.path.exists(savedirc): os.makedirs(savedirc) 
            savename = savedir + tsne_text
            savenameb = savedirb + tsne_text
            savenamec = savedirc + tsne_text
            # X_tsne = np.concatenate((data_current[0], data_past[0], data_future[0], data_gen[0]), axis=0)
            # y_tsne = np.concatenate((data_current[1], data_past[1], data_future[1], data_gen[1]), axis=0).astype(int)
            # y_tsne_pred = np.concatenate((data_current[3], data_past[3], data_future[3], data_gen[3]), axis=0).astype(int)
            # ym_tsne = np.concatenate((data_current[2], data_past[2], data_future[2], data_gen[2]), axis=0).astype(int)
            X_tsne = np.concatenate((data_current[0], data_past[0], data_gen[0]), axis=0)
            y_tsne = np.concatenate((data_current[1], data_past[1],  data_gen[1]), axis=0).astype(int)
            y_tsne_pred = np.concatenate((data_current[3], data_past[3], data_gen[3]), axis=0).astype(int)
            ym_tsne = np.concatenate((data_current[2], data_past[2],  data_gen[2]), axis=0).astype(int)
            tsne_eval_new(X_tsne, y_tsne, ym_tsne, y_tsne_pred, markers, savename, savenameb, savenamec, title, num_color)
        
        return embedding


    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):

        loss_kd = torch.zeros((1,), requires_grad=True).cuda()

        if dw_force is not None:
            dw_cls = dw_force
        elif self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]

        # forward pass
        logits = self.forward(inputs)

        # # classification 
        class_idx = np.arange(self.batch_size)
        loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])

        # KD old
        if target_scores is not None:
            loss_kd = self.mu * loss_fn_kd(logits[class_idx], target_scores[class_idx], dw_cls[class_idx], np.arange(self.last_valid_out_dim).tolist(), self.DTemp)

        # KD new
        if target_scores is not None:
            target_scores = F.softmax(target_scores[:, :self.last_valid_out_dim] / self.DTemp, dim=1)
            target_scores = [target_scores]
            target_scores.append(torch.zeros((len(targets),self.valid_out_dim-self.last_valid_out_dim), requires_grad=True).cuda())
            target_scores = torch.cat(target_scores, dim=1)
            loss_kd += self.mu * loss_fn_kd(logits[kd_index], target_scores[kd_index], dw_cls[kd_index], np.arange(self.valid_out_dim).tolist(), self.DTemp, soft_t = True)

        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits

    def validation(self, dataloader, model=None, task_in = None, task_metric='acc', relabel_clusters = True, verbal = True, filter_past = False):

        if model is None:
            if task_metric == 'acc':
                model = self.model
            # elif task_metric == 'aux_task':
            #     return self.aux_task(dataloader)
            else:
                return -1

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
            
        model.train(orig_mode)

        if verbal:
            self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                    .format(acc=acc, time=batch_timer.toc()))
        return acc.avg

    def visualize_gen(self, dataloader, topdir, name, model=None, gen_text = "", gen_images = None, show_change = False):

        orig_mode = self.training
        self.eval()

        if model is None: model = self.previous_teacher

        num_show = 50
        cols = 10
        input_gen = []
        x_real = []
        y_real = []
        y_pred = []
        x_pen_past_real = []
        x_pen_past_real_score = []
        x_pen_cur_real = []
        for i, (input, target, _) in enumerate(dataloader):
            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            input_gen.append(input)
            x_real.extend(input.detach().cpu().numpy())
            y_real.extend(target.detach().cpu().numpy())
            output = self.model.forward(input)[:, :self.valid_out_dim]
            max_score, pl = output.max(dim=1)
            y_pred.extend(pl.detach().cpu().numpy())
            x_pen_past_real.extend(self.previous_teacher.generate_scores_pen(input).cpu().detach().tolist())
            x_pen_past_real_score.extend(F.softmax(self.previous_teacher.generate_scores(input), dim=2).cpu().detach().tolist())
            x_pen_cur_real.extend(self.model.forward(x=input, pen=True).cpu().detach().tolist())
            # update style images
            if len(input) == self.batch_size: self.style_imgs = input
        # sample_ind = np.random.choice(len(y_real), 2*self.batch_size)
        input_gen = torch.cat(input_gen)
        # input_gen = input_gen[sample_ind]
        x_real, y_real, y_pred = np.asarray(x_real), np.asarray(y_real), np.asarray(y_pred)
        # x_real, y_real, y_pred = x_real[sample_ind], y_real[sample_ind], y_pred[sample_ind]
        min_x, max_x = np.amin(x_real), np.amax(x_real)
        x_pen_past_real, x_pen_cur_real = np.asarray(x_pen_past_real), np.asarray(x_pen_cur_real)
        # x_pen_past_real, x_pen_cur_real = x_pen_past_real[sample_ind], x_pen_cur_real[sample_ind]
        x_pen_past_real_score = np.asarray(x_pen_past_real_score)
        # x_pen_past_real_score = x_pen_past_real_score[sample_ind]

        if gen_images is None:
            tsne = False
            x_gen, y_gen = self.sample(model, None, self.batch_size, self.device, False)
        else:
            tsne = True
            x_gen, y_gen = gen_images
        output = self.model.forward(x_gen)[:, :self.valid_out_dim]
        max_score, pl = output.max(dim=1)
        y_gen_pred = pl.detach().cpu().numpy()
        x_gen_cur = self.model.forward(x=x_gen, pen=True).detach().cpu().numpy()
        x_gen_past = self.previous_teacher.generate_scores_pen(x_gen).detach().cpu().numpy()
        x_gen_past_score = F.softmax(self.previous_teacher.generate_scores(x_gen), dim=2).detach().cpu().numpy()
        x_gen, y_gen = x_gen.detach().cpu().numpy(), y_gen.detach().cpu().numpy()
        

        # cka eval
        index_r_p = np.where(y_real < self.last_valid_out_dim)[0]
        index_r_n = np.where(y_real >= self.last_valid_out_dim)[0]
        X_real_a = x_pen_cur_real[index_r_p]
        X_real_b = x_pen_cur_real[index_r_n]
        # X_gen_a = x_pen_past_real[index_r_p]
        X_gen_a = x_gen_cur

        # sample
        n_sample = min(len(X_real_a), len(X_real_b), len(X_gen_a))
        sample_ind = np.random.choice(len(X_real_a), n_sample, replace=False)
        X_real_a = X_real_a[sample_ind]
        # X_gen_a = X_gen_a[sample_ind]
        sample_ind = np.random.choice(len(X_real_b), n_sample, replace=False)
        X_real_b = X_real_b[sample_ind]
        sample_ind = np.random.choice(len(X_gen_a), n_sample, replace=False)
        X_gen_a = X_gen_a[sample_ind]
        

        # return cka score
        # cka_domain = calculate_cka(X_real_a, X_gen_a)
        # cka_real = calculate_cka(X_real_a, X_real_b)
        # print('sanity check a - ' + str(calculate_cka(X_real_a, X_real_a[::-1])))
        # print('sanity check b - ' + str(calculate_cka(X_gen_a, X_gen_a)))

        cka_domain = np.linalg.norm((np.mean(X_real_a, axis=0) - np.mean(X_gen_a, axis=0))/np.std(X_real_a, axis=0)) # / np.linalg.norm(np.mean(X_real_a, axis=0))
        cka_real = np.linalg.norm((np.mean(X_real_a, axis=0) - np.mean(X_real_b, axis=0))/np.std(X_real_a, axis=0)) # / np.linalg.norm(np.mean(X_real_a, axis=0))
        print('domain - ' + str(cka_domain))
        print('content - ' + str(cka_real))
        
        # append to long term array
        if self.epoch == 0:
            epoch_x = 0
        else:
            epoch_x = self.epoch+1
        self.cka_holder[0].append(epoch_x)
        self.cka_holder[1].append(cka_domain)
        self.cka_holder[2].append(cka_real)

        # save in csv
        ckadir = topdir + name + '/cka/'
        if not os.path.exists(ckadir): os.makedirs(ckadir)
        if len(self.cka_holder[0]) > 1:
            np.savetxt(ckadir + 'real.csv', np.asarray(self.cka_holder[2]), delimiter=",", fmt='%.6f')  
            np.savetxt(ckadir + 'domain.csv', np.asarray(self.cka_holder[1]), delimiter=",", fmt='%.6f')  
        # self.cka_holder[2] = np.loadtxt(ckadir + 'real.csv')
        # self.cka_holder[1] = np.loadtxt(ckadir + 'domain.csv')
        # self.cka_holder[0] = [0, 50, 100, 150, 200, 250]

        # plot
        cka_plot(ckadir+'cka.png', np.asarray(self.cka_holder[0]), [np.asarray(self.cka_holder[1]),np.asarray(self.cka_holder[2])], ['Real T1 vs Synthetic T1','Real T1 vs Real T2'])
        # print(done)
        # # turn off
        # if self.epoch >= 99: print(apple)

        # for vis
        self.x_gen = x_gen_cur; self.y_gen = -1 - y_gen; self.y_gen_pred = y_gen_pred

        # this way normalizes per image
        savedir = topdir + name + '/images/'
        if not os.path.exists(savedir): os.makedirs(savedir)  
        rows = math.ceil(num_show/cols)
        if len(gen_text) > 0:
            data_loop = [[(x_gen, y_gen), 'gen']]
        else:
            data_loop = [[(x_real, y_real), 'real'], [(x_gen, y_gen), 'gen']]
        for data in data_loop:
            x,y = data[0]
            for j in range(num_show):
                plt.subplot(rows, cols, j + 1)
                im_show = np.squeeze(x[j].transpose((1,2,0)))
                im_min = np.amin(im_show)
                im_max = np.amax(im_show)
                min_scale = im_min / min_x
                max_scale = im_max / max_x
                im_show = (im_show - im_min) / ((im_max - im_min))
                plt.imshow((im_show * 255).astype(np.uint8)); plt.axis('off')
                plt.title(str(y[j]) + ' ({min:.1f},{max:.1f})'.format(min=min_scale,max=max_scale),fontsize=6)
            save_name = savedir + data[1] + '_images' + gen_text + '.png'
            plt.savefig(save_name) 
            plt.close()

        self.train(orig_mode)

    def update_stats(self, x, y):
    
        pass

    def swap_model(self, val):
        pass

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

    def save_model(self, filename):
        
        model_state = self.generator.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving generator model to:', filename)
        torch.save(model_state, filename + 'generator.pth')
        super(DeepInversionGenBN, self).save_model(filename)

    def load_model(self, filename):
        self.generator.load_state_dict(torch.load(filename + 'generator.pth'))
        if self.gpu:
            self.generator = self.generator.cuda()
        self.generator.eval()
        super(DeepInversionGenBN, self).load_model(filename)

    def create_generator(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        generator = models.__dict__[cfg['gen_model_type']].__dict__[cfg['gen_model_name']]()
        return generator

    def print_model(self):
        super(DeepInversionGenBN, self).print_model()
        self.log(self.generator)
        self.log('#parameter of generator:', self.count_parameter_gen())
    
    def reset_model(self):
        super(DeepInversionGenBN, self).reset_model()
        self.generator.apply(weight_reset)

    def count_parameter_gen(self):
        return sum(p.numel() for p in self.generator.parameters())

    def count_memory(self, dataset_size):
        return self.count_parameter() + self.count_parameter_gen() + self.memory_size * dataset_size[0]*dataset_size[1]*dataset_size[2]

    def cuda_gen(self):
        self.generator = self.generator.cuda()
        return self

    def sample(self, model, model_student, dim, device, return_scores, step=False):

        return model.sample(dim, device, model_student, self.balanced_linear, self.valid_out_dim, return_scores=return_scores, step=step)

class DeepInversionLWF(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(DeepInversionLWF, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()

        if dw_force is not None:
            dw_cls = dw_force
        elif self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]

        # forward pass
        logits = self.forward(inputs)

        # classification 
        class_idx = np.arange(self.batch_size)
        loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])

        # KD old
        if target_scores is not None:
            loss_kd = self.mu * loss_fn_kd(logits, target_scores, dw_cls, np.arange(self.last_valid_out_dim).tolist(), self.DTemp)

        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits

class DeepInversionGenBN_agem(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(DeepInversionGenBN_agem, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        loss_gem = torch.zeros((1,), requires_grad=True).cuda()
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]

        # forward pass
        logits = self.forward(inputs)

        # classification 
        class_idx = np.arange(self.batch_size)
        if self.playground_flag and self.inversion_replay:
            logits_class = [self.previous_linear(self.model.forward(x=inputs[class_idx], pen=True))[:,:self.last_valid_out_dim]]
            logits_class.append(logits[class_idx][:,self.last_valid_out_dim:self.valid_out_dim])
            loss_class = self.criterion(torch.cat(logits_class, dim=1), targets[class_idx].long(), dw_cls[class_idx])
        else:
            loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])

        # KD new
        if target_scores is not None:

            # hard - linear
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            logits_KD = self.previous_linear(self.model.forward(x=inputs[kd_index], pen=True))[:,:self.last_valid_out_dim]
            logits_KD_past = self.previous_linear(self.previous_teacher.generate_scores_pen(inputs[kd_index]))[:,:self.last_valid_out_dim]
            loss_kd = self.mu * (self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1) * dw_KD).mean() / (logits_KD.size(1))
        
        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()

        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits

class DeepInversionGenBN_agem_global(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(DeepInversionGenBN_agem_global, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        loss_gem = torch.zeros((1,), requires_grad=True).cuda()
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        
        # first, get gradient for constraint loss
        grad_rep = None
        if self.inversion_replay:
            logits_gem = self.forward(inputs[kd_index])
            loss_gem = self.criterion(logits_gem, targets[kd_index].long(), dw_cls[kd_index])

            # backwards and store
            self.optimizer.zero_grad()
            loss_gem.backward()
            grad_rep = []
            for p in self.model.parameters():
                if p.requires_grad:
                    grad_rep.append(p.grad.view(-1))
            grad_rep = torch.cat(grad_rep)

        # forward pass
        logits = self.forward(inputs)

        # classification 
        class_idx = np.arange(self.batch_size)
        if self.playground_flag and self.inversion_replay:
            logits_class = [self.previous_linear(self.model.forward(x=inputs[class_idx], pen=True))[:,:self.last_valid_out_dim]]
            logits_class.append(logits[class_idx][:,self.last_valid_out_dim:self.valid_out_dim])
            loss_class = self.criterion(torch.cat(logits_class, dim=1), targets[class_idx].long(), dw_cls[class_idx])
        else:
            loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])

        # KD new
        if target_scores is not None:

            # hard - linear
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            logits_KD = self.previous_linear(self.model.forward(x=inputs[kd_index], pen=True))[:,:self.last_valid_out_dim]
            logits_KD_past = self.previous_linear(self.previous_teacher.generate_scores_pen(inputs[kd_index]))[:,:self.last_valid_out_dim]
            loss_kd = self.mu * (self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1) * dw_KD).mean() / (logits_KD.size(1))
        
        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()

        # apply gem fix
        if grad_rep is not None:
            grad_cur = []
            for p in self.model.parameters():
                if p.requires_grad:
                    grad_cur.append(p.grad.view(-1))
            grad_cur = torch.cat(grad_cur)
            angle = (grad_cur*grad_rep).sum()
            if angle < 0:
                length_rep = (grad_rep*grad_rep).sum()
                grad_proj = grad_cur-(angle/length_rep)*grad_rep
                index = 0
                for p in self.model.parameters():
                    if p.requires_grad:
                        n_param = p.numel()  # number of parameters in [p]
                        p.grad.copy_(grad_proj[index:index+n_param].view_as(p))
                        index += n_param

        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits

class DeepInversionGenBN_agem_l2(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(DeepInversionGenBN_agem_l2, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        loss_gem = torch.zeros((1,), requires_grad=True).cuda()
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        
        # class balancing
        mappings = torch.ones(targets.size(), dtype=torch.float32)
        if self.gpu:
            mappings = mappings.cuda()
        rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
        mappings[:self.last_valid_out_dim] = rnt
        mappings[self.last_valid_out_dim:] = 1-rnt
        dw_cls = mappings[targets.long()]

        # forward pass
        logits_pen = self.model.forward(x=inputs, pen=True)
        if len(self.config['gpuid']) > 1:
            logits = self.model.module.last(logits_pen)
        else:
            logits = self.model.last(logits_pen)
        

        # classification 
        class_idx = np.arange(self.batch_size)
        if self.playground_flag and self.inversion_replay:

            # local classification
            loss_class = self.criterion(logits[class_idx,self.last_valid_out_dim:self.valid_out_dim], (targets[class_idx]-self.last_valid_out_dim).long(), dw_cls[class_idx]) 

            # ft classification  
            with torch.no_grad():             
                feat_class = self.model.forward(x=inputs, pen=True).detach()
            if len(self.config['gpuid']) > 1:
                loss_class += self.criterion(self.model.module.last(feat_class), targets.long(), dw_cls)
            else:
                loss_class += self.criterion(self.model.last(feat_class), targets.long(), dw_cls)
            
        else:
            loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])

        # KD new
        if target_scores is not None:

            # hard - linear
            kd_index = np.arange(2 * self.batch_size)
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            logits_KD = self.previous_linear(logits_pen[kd_index])[:,:self.last_valid_out_dim]
            logits_KD_past = self.previous_linear(self.previous_teacher.generate_scores_pen(inputs[kd_index]))[:,:self.last_valid_out_dim]
            loss_kd = self.mu * (self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1) * dw_KD).mean() / (logits_KD.size(1))
            
        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()

        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits


class DeepInversionGenBN_agem_l2_ablate_a(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(DeepInversionGenBN_agem_l2_ablate_a, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        loss_gem = torch.zeros((1,), requires_grad=True).cuda()
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]

        # forward pass
        logits = self.forward(inputs)

        # classification 
        class_idx = np.arange(self.batch_size)
        if self.playground_flag and self.inversion_replay:

            # local classification
            loss_class = self.criterion(logits[class_idx,self.last_valid_out_dim:self.valid_out_dim], (targets[class_idx]-self.last_valid_out_dim).long(), dw_cls[class_idx]) 

            # ft classification               
            feat_class = self.model.forward(x=inputs, pen=True).detach()
            loss_class += self.criterion(self.model.last(feat_class), targets.long(), dw_cls)

        else:
            loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])

        # KD new
        if target_scores is not None:

            # hard - linear
            kd_index = np.arange(2 * self.batch_size)
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            logits_KD = self.previous_linear(self.model.forward(x=inputs[kd_index], pen=True))[:,:self.last_valid_out_dim]
            logits_KD_past = self.previous_linear(self.previous_teacher.generate_scores_pen(inputs[kd_index]))[:,:self.last_valid_out_dim]
            loss_kd = self.mu * (self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1) * dw_KD).mean() / (logits_KD.size(1))
            
        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()

        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits

class DeepInversionGenBN_agem_l2_ablate_b(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(DeepInversionGenBN_agem_l2_ablate_b, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        loss_gem = torch.zeros((1,), requires_grad=True).cuda()
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        
        # class balancing
        mappings = torch.ones(targets.size(), dtype=torch.float32)
        if self.gpu:
            mappings = mappings.cuda()
        rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
        mappings[:self.last_valid_out_dim] = rnt
        mappings[self.last_valid_out_dim:] = 1-rnt
        dw_cls = mappings[targets.long()]

        # forward pass
        logits = self.forward(inputs)

        # classification 
        class_idx = np.arange(self.batch_size)
        if self.playground_flag and self.inversion_replay:

            # local classification
            loss_class = self.criterion(logits[class_idx,self.last_valid_out_dim:self.valid_out_dim], (targets[class_idx]-self.last_valid_out_dim).long(), dw_cls[class_idx]) 

            # ft classification               
            feat_class = self.model.forward(x=inputs, pen=True).detach()
            loss_class += self.criterion(self.model.last(feat_class), targets.long(), dw_cls)

        else:
            loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])

        # KD new
        if target_scores is not None:

            # hard - linear
            kd_index = np.arange(self.batch_size)
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            logits_KD = self.previous_linear(self.model.forward(x=inputs[kd_index], pen=True))[:,:self.last_valid_out_dim]
            logits_KD_past = self.previous_linear(self.previous_teacher.generate_scores_pen(inputs[kd_index]))[:,:self.last_valid_out_dim]
            loss_kd = self.mu * (self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1) * dw_KD).mean() / (logits_KD.size(1))
            
        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()

        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits

class DeepInversionGenBN_agem_l2_ablate_c(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(DeepInversionGenBN_agem_l2_ablate_c, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        loss_gem = torch.zeros((1,), requires_grad=True).cuda()
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        
        # class balancing
        mappings = torch.ones(targets.size(), dtype=torch.float32)
        if self.gpu:
            mappings = mappings.cuda()
        rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
        mappings[:self.last_valid_out_dim] = rnt
        mappings[self.last_valid_out_dim:] = 1-rnt
        dw_cls = mappings[targets.long()]

        # forward pass
        logits = self.forward(inputs)

        # classification 
        class_idx = np.arange(self.batch_size)
        if self.playground_flag and self.inversion_replay:

            # local classification
            loss_class = self.criterion(logits[class_idx,self.last_valid_out_dim:self.valid_out_dim], (targets[class_idx]-self.last_valid_out_dim).long(), dw_cls[class_idx]) 

            # ft classification               
            feat_class = self.model.forward(x=inputs, pen=True).detach()
            loss_class += self.criterion(self.model.last(feat_class), targets.long(), dw_cls)

        else:
            loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])

        # KD new
        if target_scores is not None:

            # hard - linear
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            logits_KD = self.previous_linear(self.model.forward(x=inputs[kd_index], pen=True))[:,:self.last_valid_out_dim]
            logits_KD_past = self.previous_linear(self.previous_teacher.generate_scores_pen(inputs[kd_index]))[:,:self.last_valid_out_dim]
            loss_kd = self.mu * (self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1) * dw_KD).mean() / (logits_KD.size(1))
            
        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()

        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits

class DeepInversionGenBN_agem_l2_ablate_d(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(DeepInversionGenBN_agem_l2_ablate_d, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        loss_gem = torch.zeros((1,), requires_grad=True).cuda()
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        
        # class balancing
        mappings = torch.ones(targets.size(), dtype=torch.float32)
        if self.gpu:
            mappings = mappings.cuda()
        rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
        mappings[:self.last_valid_out_dim] = rnt
        mappings[self.last_valid_out_dim:] = 1-rnt
        dw_cls = mappings[targets.long()]

        # forward pass
        logits = self.forward(inputs)

        # classification 
        class_idx = np.arange(self.batch_size)
        if self.playground_flag and self.inversion_replay:

            # local classification
            loss_class = self.criterion(logits[class_idx,self.last_valid_out_dim:self.valid_out_dim], (targets[class_idx]-self.last_valid_out_dim).long(), dw_cls[class_idx]) 

        else:
            loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])

        # KD new
        if target_scores is not None:

            # hard - linear
            kd_index = np.arange(2 * self.batch_size)
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            logits_KD = self.previous_linear(self.model.forward(x=inputs[kd_index], pen=True))[:,:self.last_valid_out_dim]
            logits_KD_past = self.previous_linear(self.previous_teacher.generate_scores_pen(inputs[kd_index]))[:,:self.last_valid_out_dim]
            loss_kd = self.mu * (self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1) * dw_KD).mean() / (logits_KD.size(1))
            
        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()

        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits

class DeepInversionGenBN_agem_l2_ablate_e(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(DeepInversionGenBN_agem_l2_ablate_e, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        loss_gem = torch.zeros((1,), requires_grad=True).cuda()
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        
        # class balancing
        mappings = torch.ones(targets.size(), dtype=torch.float32)
        if self.gpu:
            mappings = mappings.cuda()
        rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
        mappings[:self.last_valid_out_dim] = rnt
        mappings[self.last_valid_out_dim:] = 1-rnt
        dw_cls = mappings[targets.long()]

        # forward pass
        logits = self.forward(inputs)

        # classification 
        class_idx = np.arange(self.batch_size)
        loss_class = self.criterion(logits, targets.long(), dw_cls)

        # KD new
        if target_scores is not None:

            # hard - linear
            kd_index = np.arange(2 * self.batch_size)
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            logits_KD = self.previous_linear(self.model.forward(x=inputs[kd_index], pen=True))[:,:self.last_valid_out_dim]
            logits_KD_past = self.previous_linear(self.previous_teacher.generate_scores_pen(inputs[kd_index]))[:,:self.last_valid_out_dim]
            loss_kd = self.mu * (self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1) * dw_KD).mean() / (logits_KD.size(1))
            
        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()

        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits

class DeepInversionGenBN_agem_l2_ablate_f(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(DeepInversionGenBN_agem_l2_ablate_f, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        loss_gem = torch.zeros((1,), requires_grad=True).cuda()
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        
        # class balancing
        mappings = torch.ones(targets.size(), dtype=torch.float32)
        if self.gpu:
            mappings = mappings.cuda()
        rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
        mappings[:self.last_valid_out_dim] = rnt
        mappings[self.last_valid_out_dim:] = 1-rnt
        dw_cls = mappings[targets.long()]

        # forward pass
        logits = self.forward(inputs)

        # classification 
        class_idx = np.arange(self.batch_size)
        if self.playground_flag and self.inversion_replay:

            # local classification
            loss_class = self.criterion(logits[class_idx,self.last_valid_out_dim:self.valid_out_dim], (targets[class_idx]-self.last_valid_out_dim).long(), dw_cls[class_idx]) 

            # ft classification               
            feat_class = self.model.forward(x=inputs, pen=True).detach()
            loss_class += self.criterion(self.model.last(feat_class), targets.long(), dw_cls)

        else:
            loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])

        # KD new
        if target_scores is not None:

            # hard - linear
            kd_index = np.arange(2 * self.batch_size)
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            logits_KD = self.model.forward(x=inputs[kd_index], pen=True)
            logits_KD_past = self.previous_teacher.generate_scores_pen(inputs[kd_index])
            loss_kd = self.mu * (self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1) * dw_KD).mean() / (logits_KD.size(1))
            
        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()

        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits

class DeepInversionGenBN_agem_l2_ablate_g(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(DeepInversionGenBN_agem_l2_ablate_g, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        loss_gem = torch.zeros((1,), requires_grad=True).cuda()
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        
        # class balancing
        mappings = torch.ones(targets.size(), dtype=torch.float32)
        if self.gpu:
            mappings = mappings.cuda()
        rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
        mappings[:self.last_valid_out_dim] = rnt
        mappings[self.last_valid_out_dim:] = 1-rnt
        dw_cls = mappings[targets.long()]

        # forward pass
        logits = self.forward(inputs)

        # classification 
        class_idx = np.arange(self.batch_size)
        if self.playground_flag and self.inversion_replay:

            # local classification
            loss_class = self.criterion(logits[class_idx,self.last_valid_out_dim:self.valid_out_dim], (targets[class_idx]-self.last_valid_out_dim).long(), dw_cls[class_idx]) 

            # ft classification               
            feat_class = self.model.forward(x=inputs, pen=True).detach()
            loss_class += self.criterion(self.model.last(feat_class), targets.long(), dw_cls)

        else:
            loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])

        # KD new
        if target_scores is not None:

            # soft
            kd_index = np.arange(2 * self.batch_size)
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            loss_kd = self.mu * loss_fn_kd(logits[kd_index], target_scores[kd_index], dw_KD, np.arange(self.last_valid_out_dim).tolist(), self.DTemp)
            
        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()

        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits

class DeepInversionGenBN_agem_l2_ablate_h(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(DeepInversionGenBN_agem_l2_ablate_h, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        loss_gem = torch.zeros((1,), requires_grad=True).cuda()
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        
        # class balancing
        mappings = torch.ones(targets.size(), dtype=torch.float32)
        if self.gpu:
            mappings = mappings.cuda()
        rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
        mappings[:self.last_valid_out_dim] = rnt
        mappings[self.last_valid_out_dim:] = 1-rnt
        dw_cls = mappings[targets.long()]

        # forward pass
        logits_pen = self.model.forward(x=inputs, pen=True)
        if len(self.config['gpuid']) > 1:
            logits = self.model.module.last(logits_pen)
        else:
            logits = self.model.last(logits_pen)
        

        # classification 
        class_idx = np.arange(self.batch_size)
        if self.playground_flag and self.inversion_replay:

            # local classification
            loss_class = self.criterion(logits[class_idx,self.last_valid_out_dim:self.valid_out_dim], (targets[class_idx]-self.last_valid_out_dim).long(), dw_cls[class_idx]) 

            # ft classification  
            with torch.no_grad():             
                feat_class = self.model.forward(x=inputs, pen=True).detach()
            if len(self.config['gpuid']) > 1:
                loss_class += self.criterion(self.model.module.last(feat_class), targets.long(), dw_cls)
            else:
                loss_class += self.criterion(self.model.last(feat_class), targets.long(), dw_cls)
            
        else:
            loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])

        # KD new
        if target_scores is not None:

            # hard - linear
            kd_index = np.arange(2 * self.batch_size)
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            logits_KD = self.model.module.last(logits_pen[kd_index])[:,:self.last_valid_out_dim]
            logits_KD_past = self.previous_linear(self.previous_teacher.generate_scores_pen(inputs[kd_index]))[:,:self.last_valid_out_dim]
            loss_kd = self.mu * (self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1) * dw_KD).mean() / (logits_KD.size(1))
            
        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()

        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits

class DeepInversionGenBN_bic(DeepInversionGenBN):

    def __init__(self, learner_config):
        super(DeepInversionGenBN_bic, self).__init__(learner_config)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
        self.bic = True

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        loss_kd = torch.zeros((1,), requires_grad=True).cuda()
        loss_gem = torch.zeros((1,), requires_grad=True).cuda()
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]

        # forward pass
        logits = self.forward(inputs)

        # classification 
        class_idx = np.arange(self.batch_size)
        loss_class = self.criterion(logits[class_idx], targets[class_idx].long(), dw_cls[class_idx])

        # KD new
        if target_scores is not None:

            # hard - linear
            kd_index = np.arange(2 * self.batch_size)
            dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()]
            logits_KD = self.previous_linear(self.model.forward(x=inputs[kd_index], pen=True))[:,:self.last_valid_out_dim]
            logits_KD_past = self.previous_linear(self.previous_teacher.generate_scores_pen(inputs[kd_index]))[:,:self.last_valid_out_dim]
            loss_kd = self.mu * (self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1) * dw_KD).mean() / (logits_KD.size(1))

            # extra weighting
            mu_c = self.last_valid_out_dim / self.valid_out_dim
            loss_class = (1 - mu_c) * loss_class
            loss_kd = mu_c * loss_kd
            
        total_loss = loss_class + loss_kd
        self.optimizer.zero_grad()
        total_loss.backward()

        # step
        self.optimizer.step()

        return total_loss.detach(), loss_class.detach(), loss_kd.detach(), logits

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        self.pre_steps()

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

            losses = [AverageMeter() for i in range(3)]
            acc = AverageMeter()
            accg = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            self.save_gen = False
            self.save_gen_later = False
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

                    # update style images
                    if len(x[0]) == self.batch_size: self.style_imgs = x[0]

                    # gen vis logic
                    if (self.epoch) in STEPS_SAVE:
                        self.save_gen = True

                    # data replay
                    if (self.epoch) % self.refresh_iters == 0 or self.epoch == 0:
                        if not self.inversion_replay:
                            x_replay = None   #-> if no replay
                        else:
                            # visualize
                            if self.debug_dir is not None and self.epoch > 0 and self.save_gen_later:
                                if self.inversion_replay: self.visualize_gen(val_loader, self.debug_dir, "inversions", gen_images = (x_replay, y_replay), gen_text = str(self.epoch + 1) + '-steps_before-gen-update', show_change=True)
                                self.save_gen_later = False
                            x_replay, y_replay, y_replay_hat = self.sample(self.previous_teacher, self.model, len(x[0]), self.device, True, step=True)
                            # visualize
                            if self.debug_dir is not None and self.save_gen:
                                if self.inversion_replay: self.visualize_gen(val_loader, self.debug_dir, "inversions", gen_images = (x_replay, y_replay), gen_text = str(self.epoch+ 1) + '-steps_post-gen-update')
                                self.save_gen_later = True
                                self.save_gen = False

                    # if KD
                    if self.inversion_replay:
                        y_hat = self.previous_teacher.generate_scores(x[0], allowed_predictions=np.arange(self.last_valid_out_dim))
                        _, y_hat_com = self.combine_data(((x[0], y_hat),(x_replay, y_replay_hat)))
                    else:
                        y_hat_com = None

                    # combine inputs and generated samples for classification
                    if self.inversion_replay:
                        x_com, y_com = self.combine_data(((x[0], y),(x_replay, y_replay)))
                    else:
                        x_com, y_com = x[0], y

                    # model update
                    loss, loss_class, loss_kd, output= self.update_model(x_com, y_com, y_hat_com, kd_index = np.arange(len(x[0]), len(x_com)))

                    # update stats
                    self.update_stats(x[0], y)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc()) 

                    # measure accuracy and record loss
                    y_com = y_com.detach()
                    accumulate_acc(output[:self.batch_size], y_com[:self.batch_size], task, acc, topk=(self.top_k,))
                    if self.inversion_replay: accumulate_acc(output[self.batch_size:], y_com[self.batch_size:], task, accg, topk=(self.top_k,))
                    losses[0].update(loss,  y_com.size(0)) 
                    losses[1].update(loss_class,  y_com.size(0))
                    losses[2].update(loss_kd,  y_com.size(0))
                    batch_timer.tic()

                # eval update
                self.swap_model(True)

                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | CE Loss {lossb.avg:.3f} | KD Loss {lossc.avg:.3f} | Conf Loss {lossd:.3f}'.format(loss=losses[0],lossb=losses[1],lossc=losses[2], lossd=losses[0].avg-losses[1].avg-losses[2].avg))
                self.log(' * Train Acc {acc.avg:.3f} | Train Acc Gen {accg.avg:.3f}'.format(acc=acc,accg=accg))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)
                    self.validation(val_loader, filter_past = True)

                # reset
                losses = [AverageMeter() for i in range(3)]
                acc = AverageMeter()
                accg = AverageMeter()

                self.swap_model(False)

            # fine-tuning phase!!
            # data weighting
            self.data_weighting(train_dataset)

            # new optimizer
            self.optimizer, self.scheduler = self.new_optimizer(self.model.last)

            # Evaluate the performance of current task
            self.log('FT Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
            if val_loader is not None:
                self.validation(val_loader)

            losses = [AverageMeter() for i in range(1)]
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
                    dw_cls = self.dw_k[y.long()]
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
                self.log('FT Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses[0],acc=acc))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = [AverageMeter() for l in range(1)]
                acc = AverageMeter()

        if self.inversion_replay and self.debug_dir is not None: self.visualize_gen(val_loader, self.debug_dir, "final")

        self.model.eval()
        self.last_last_valid_out_dim = self.last_valid_out_dim
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # for eval
        if self.previous_teacher is not None:
            self.previous_previous_teacher = self.previous_teacher

        # HERE JAMES
        if (self.out_dim == self.valid_out_dim): need_train = False
        
        # new teacher
        self.previous_teacher = Teacher(solver=copy.deepcopy(self.model), generator=self.generator, gen_opt = self.generator_optimizer, img_shape = (-1, train_dataset.nch,train_dataset.im_size, train_dataset.im_size), iters = self.power_iters, deep_inv_params = self.deep_inv_params, class_idx = np.arange(self.valid_out_dim), var = self.variance_torch, mean = self.centers_torch, train = need_train, config = self.config)
        self.sample(self.previous_teacher, None, self.batch_size, self.device, False)
        self.inversion_replay = True
        if len(self.config['gpuid']) > 1:
            self.previous_linear = copy.deepcopy(self.model.module.last)
        else:
            self.previous_linear = copy.deepcopy(self.model.last)
        self.upper_bound_past = copy.deepcopy(self.upper_bound)
        # self.centers_torch_past = copy.deepcopy(self.centers_torch)
        # self.variance_torch_past = copy.deepcopy(self.variance_torch)

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            if self.bic:
                train_dataset.update_coreset_ic(self.memory_size, np.arange(self.last_valid_out_dim), self.previous_teacher)
            else:
                train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        self.swap_model(True)

        try:
            return batch_time.avg
        except:
            return None