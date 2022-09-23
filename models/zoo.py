import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
import numpy as np

def tensor_prompt(a, b, c=None):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    nn.init.uniform_(p)
    return p

def init_weights_adapter(m):
    if type(m) == nn.Linear:
        m.weight.data.fill_(0.0)

def two_layer_adapter(a):
    adapter = nn.Sequential(
                            nn.Linear(a, a),
                            nn.Tanh(),
                            nn.Linear(a, a),
                        )
    adapter.apply(init_weights_adapter)
    return adapter

class DualAdapt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param):
        super().__init__()
        self.emb_d = emb_d
        self.key_d = 768
        self._init_smart(emb_d, n_tasks, prompt_param)

        # init frequency table
        self.frequency_past = None
        self.frequency_current = {}
        for l in self.e_layers:
            self.frequency_current[l] = [0.001 for i in range(self.e_pool_size)]

        # g prompt init
        for g in self.g_layers:
            a = two_layer_adapter(emb_d)
            setattr(self, f'g_a_{g}',a)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, emb_d)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)

    def _init_smart(self, emb_d, n_tasks, prompt_param):
        
        self.top_k = 1
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = [0,1]
        self.e_layers = [3,4,5]

        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = prompt_param[1]
        self.e_pool_size = prompt_param[0]

    def process_frequency(self):
        self.frequency_past = {}
        for key, f_table in self.frequency_current.items():
            f_past = []
            for p in range(len(f_table)):               
                f_past.append(float(f_table[p])/sum(f_table))
            self.frequency_past[key] = f_past
            self.frequency_current[key] = [0.001 for i in range(self.e_pool_size)]

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            K = getattr(self,f'e_k_{l}') # 0 based indexing here
            B, C = x_querry.shape

            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            top_k = torch.topk(cos_sim, self.top_k, dim=1)
            ix = top_k.indices

            if train:
                loss = 1.0 - top_k.values.mean()  # the cosine similarity is always le 1
                p = getattr(self,f'e_p_{l}') # 0 based indexing here
                if self.task_id_bootstrap:
                    P_ = p[task_id].expand(len(x_querry),-1,-1)
                else:
                    k_idx = top_k.indices
                    P_ = p[k_idx][:,0]

            else:
                k_idx = top_k.indices
                p = getattr(self,f'e_p_{l}') # 0 based indexing here
                P_ = p[k_idx][:,0]
                f_to_add = np.bincount(k_idx.detach().cpu().numpy().flatten(),minlength=self.e_pool_size)
                self.frequency_current[l] += f_to_add
            
            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]
        
        # g prompts
        if l in self.g_layers:
            a = getattr(self,f'g_a_{l}') # 0 based indexing here
            ada_out = a(x_block)
            x_block = x_block + ada_out/1000.0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None
            loss = 0

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block

class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param):
        super().__init__()
        self.emb_d = emb_d
        self.key_d = 768
        self._init_smart(emb_d, n_tasks, prompt_param)

        # init frequency table
        self.frequency_past = None
        self.frequency_current = {}
        for l in self.e_layers:
            self.frequency_current[l] = [0.001 for i in range(self.e_pool_size)]

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}',p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, emb_d)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)

    def _init_smart(self, emb_d, n_tasks, prompt_param):
        
        self.top_k = 1
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = [0,1]
        self.e_layers = [3,4,5]

        # prompt pool size
        self.g_p_length = prompt_param[2]
        self.e_p_length = prompt_param[1]
        self.e_pool_size = prompt_param[0]

    def process_frequency(self):
        self.frequency_past = {}
        for key, f_table in self.frequency_current.items():
            f_past = []
            for p in range(len(f_table)):               
                f_past.append(float(f_table[p])/sum(f_table))
            self.frequency_past[key] = f_past
            self.frequency_current[key] = [0.001 for i in range(self.e_pool_size)]

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            K = getattr(self,f'e_k_{l}') # 0 based indexing here
            B, C = x_querry.shape

            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            top_k = torch.topk(cos_sim, self.top_k, dim=1)
            ix = top_k.indices

            if train:
                loss = 1.0 - top_k.values.mean()  # the cosine similarity is always le 1
                p = getattr(self,f'e_p_{l}') # 0 based indexing here
                if self.task_id_bootstrap:
                    P_ = p[task_id].expand(len(x_querry),-1,-1)
                else:
                    k_idx = top_k.indices
                    P_ = p[k_idx][:,0]

            else:
                k_idx = top_k.indices
                p = getattr(self,f'e_p_{l}') # 0 based indexing here
                P_ = p[k_idx][:,0]
                f_to_add = np.bincount(k_idx.detach().cpu().numpy().flatten(),minlength=self.e_pool_size)
                self.frequency_current[l] += f_to_add
            
            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]
        
        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length/2)
            p = getattr(self,f'g_p_{l}') # 0 based indexing here
            P_ = p.expand(len(x_querry),-1,-1)
            Gk = P_[:,:j,:]
            Gv = P_[:,j:,:]

        # combine prompts for prefix tuning
        if e_valid and g_valid:
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block

class L2P(DualPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param):
        super().__init__(emb_d, n_tasks, prompt_param)

    def _init_smart(self, emb_d, n_tasks, prompt_param):
        self.top_k = 5
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = []
        if prompt_param[2] > 0:
            self.e_layers = [0,1,3,4,5]
        else:
            self.e_layers = [0]

        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = prompt_param[1]
        self.e_pool_size = prompt_param[0]





class ResNetZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, mode=1, prompt_flag=False, prompt_param=None):
        super(ResNetZoo, self).__init__()

        # get last layer
        self.last = nn.Linear(512, num_classes)
        self.prompt_flag = prompt_flag
        self.train_flag = False
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

        elif self.prompt_flag == 'dualadapter':
            self.prompt = DualAdapt(768, prompt_param[0], prompt_param[1])

        else:
            self.prompt = None
        
        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model
        

    def forward(self, x, pen=False):

        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.feat(x)
                q = q[:,0,:]
            out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=self.train_flag, task_id=self.task_id)
            # out = out[:,0,:]
        else:
            out, _ = self.feat(x)
            # out = out.mean(dim=1)
            # out = out[:,0,:]
        out = out.view(out.size(0), -1)
        if pen:
            return out
        else:
            out = self.last(out)
            if self.prompt is not None and self.train_flag:
                return out, prompt_loss
            else:
                return out

def vit_pt_imnet(out_dim, block_division = None, prompt_flag = 'None', prompt_param=None):
    return ResNetZoo(num_classes=out_dim, pt=True, mode=0, prompt_flag=prompt_flag, prompt_param=prompt_param)