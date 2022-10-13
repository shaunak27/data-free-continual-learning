import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
import numpy as np
import math

# def ortho_penalty(t):
#     return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean() * 1e-6

def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    nn.init.uniform_(p)
    # nn.init.zeros_(p)
    return p

# class HLoss(nn.Module):
#     def __init__(self):
#         super(HLoss, self).__init__()

#     def forward(self, x):
#         b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
#         b = -1.0 * b.sum(dim=1).mean()
#         return b

# class DLoss(nn.Module):
#     def __init__(self):
#         super(DLoss, self).__init__()

#     def forward(self, x):
#         b = F.softmax(x,dim=1).mean(dim=0) 
#         # loss = 1.0 + (b * torch.log(b) / math.log(x.size(1))).sum()
#         a = b.mean()
#         loss = torch.abs(a - b).mean()
#         return loss

DEBUG_METRICS=True

class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count_f = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.expand_and_freeze = False
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)
        self.counter = 0
        # self.h_loss = HLoss()
        # self.d_loss = DLoss()
        self.rank = 20

        # e prompt init
        if DEBUG_METRICS: self.metrics = {'attention':{},'keys':{}}
        for e in self.e_layers:
            if e < 2:
                e_l = 6
            else:
                e_l = self.e_p_length
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)
            p1 = tensor_prompt(self.e_pool_size, self.rank)
            p2 = tensor_prompt(self.rank, e_l * emb_d)
            setattr(self, f'e_p_{e}_1',p1)
            setattr(self, f'e_p_{e}_2',p2)

            if DEBUG_METRICS:
                self.metrics['keys'][e] = torch.zeros((self.e_pool_size,))

    def _init_smart(self, emb_d, prompt_param):
        self.task_id_bootstrap = False

        # prompt locations
        self.e_layers = [0,1,3,4,5]

        if prompt_param[3] == 3:
            self.expand_and_freeze = True

        # prompt pool size
        self.e_p_length = prompt_param[1]
        self.e_pool_size = prompt_param[0]
        self.ortho_mu = prompt_param[2]

    def process_frequency(self):
        self.task_count_f += 1
        for e in self.e_layers:
            if e < 2:
                e_l = 6
            else:
                e_l = self.e_p_length

            # # between task paramaeter sharpening
            # # get matrix components
            # p1 = getattr(self, f'e_p_{e}_1')
            # p2 = getattr(self, f'e_p_{e}_2')

            # # freeze/control past tasks
            # # prompt pool
            # pt = int(self.e_pool_size / (self.n_tasks))
            # s = int(self.task_count_f * pt)
            # f = int((self.task_count_f + 1) * pt)
            # # rank
            # pt_rank = int(self.rank / (self.n_tasks))
            # s_rank = int(self.task_count_f * pt_rank)
            # f_rank = int((self.task_count_f + 1) * pt_rank)
            # for i in range(s,f):
            #     for j in range(s_rank):
            #         print(i)
            #         print(j)
            #         print(p1.size())
            #         p1.data[i,j] = 0
            # for j in range(s_rank):
            #     p2.data[j,:] = p2.data[j,:] * 0

            # # set the bad boys back
            # setattr(self, f'e_p_{e}_1',p1)
            # setattr(self, f'e_p_{e}_2',p2)

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            if l < 2:
                e_l = 6
            else:
                e_l = self.e_p_length
            e_valid = True
            B, C = x_querry.shape

            # get matrix components
            K = getattr(self,f'e_k_{l}')
            A = getattr(self,f'e_a_{l}')
            p1 = getattr(self, f'e_p_{l}_1')
            p2 = getattr(self, f'e_p_{l}_2')

            # expand and freeze
            if self.expand_and_freeze:
                # freeze/control past tasks
                # prompt pool
                pt = int(self.e_pool_size / (self.n_tasks))
                s = int(self.task_count_f * pt)
                f = int((self.task_count_f + 1) * pt)
                # rank
                pt_rank = int(self.e_pool_size / (self.rank))
                s_rank = int(self.task_count_f * pt_rank)
                f_rank = int((self.task_count_f + 1) * pt_rank)
                if train and self.task_count_f > 0:
                    # key/attention
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    # prompt
                    # pool size
                    p1 = torch.cat((p1[:s].detach().clone(),p1[s:f]), dim=0)
                    # # rank
                    # p1 = torch.cat((p1[:,:s_rank].detach().clone(),p1[:,s_rank:f_rank]), dim=1)
                    # p2 = torch.cat((p2[:s_rank].detach().clone(),p2[s_rank:f_rank]), dim=0)
                    p2 = p2.detach().clone()

                else:
                    K = K[0:f]
                    A = A[0:f]
                    # pool size
                    p1 = p1[0:f]
                    # # rank
                    # p1 = p1[:,0:f_rank]
                    # p2 = p2[0:f_rank]

            # form matrices
            p = torch.matmul(p1,p2)

            # resize p
            p = torch.reshape(p,(f, e_l, self.key_d))
            p = torch.permute(p,(1,0,2))
                
            if self.ortho_mu < 0:
                ##########
                # cosine sim
                ##########
                # # (b x 1 x d) - [1 x k x d] = (b x k) -> key = k x d
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(x_querry, dim=1)
                aq_k_p = torch.einsum('bd,kd->bk', q, n_K)
                # aq_k = nn.functional.softmax(aq_k_p,dim=1)
                # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
                P_ = torch.einsum('bk,lkd->bld', aq_k, p)

            else:
                ##########
                # with attention and cosine sim
                ##########
                # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
                a_querry = torch.einsum('bd,kd->bkd', x_querry, nn.functional.softmax(A,dim=1))
                # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(a_querry, dim=2)
                aq_k = torch.einsum('bkd,kd->bk', q, n_K)
                # aq_k = nn.functional.softmax(aq_k_p,dim=1)
                # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
                P_ = torch.einsum('bk,lkd->bld', aq_k, p)

            # if not train and DEBUG_METRICS:
            #     print(aq_k[0:5])
            #     self.metrics['keys'][l][0:f] = aq_k.sum(dim=0).detach().cpu()
            #     if self.counter == 5:
            #         print('**********')
            #         print('l = ' + str(l))
            #         print(self.metrics['keys'][l])
            #         self.counter = 0
            #         if self.task_count_f == 1: print(apple)
            #     self.counter += 1
             
            # select prompts
            i = int(e_l/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            loss = 0

        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block
        

class ResNetZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, mode=1, prompt_flag=False, prompt_param=None):
        super(ResNetZoo, self).__init__()

        # get last layer
        self.last = nn.Linear(512, num_classes)
        self.prompt_flag = prompt_flag
        self.task_id = None

        # get feature encoder
        if mode == 0:
            if pt:
                zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, 
                                           num_heads=12, use_grad_checkpointing=False, ckpt_layer=0,
                                           drop_path_rate=0
                                          )   
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                    map_location="cpu", check_hash=True)
                state_dict = checkpoint["model"]     
                msg = zoo_model.load_state_dict(state_dict,strict=False)
                self.last = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[0], prompt_param[1])

        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, prompt_param[0], prompt_param[1])

        else:
            self.prompt = None
        
        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model
        

    def forward(self, x, pen=False, train=False):

        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.feat(x)
                q = q[:,0,:]
            out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
            out = out[:,0,:]
        else:
            out, _ = self.feat(x)
            out = out[:,0,:]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.last(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out

def vit_pt_imnet(out_dim, block_division = None, prompt_flag = 'None', prompt_param=None):
    return ResNetZoo(num_classes=out_dim, pt=True, mode=0, prompt_flag=prompt_flag, prompt_param=prompt_param)