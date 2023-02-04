import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
import utils.clip_utils as utils
import clip.clip as clip
from .zeroshot import get_zeroshot_classifier
import numpy as np
import os
import time
def tensor_prompt(a, b, c=None):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    nn.init.uniform_(p)
    return p

class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count_f = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.expand_and_freeze = False
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # init frequency table
        for e in self.e_layers:
            setattr(self, f'freq_curr_{e}',torch.nn.Parameter(torch.zeros(self.e_pool_size,), requires_grad=False))
            setattr(self, f'freq_past_{e}',torch.nn.Parameter(torch.zeros(self.e_pool_size,), requires_grad=False))

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}',p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)

    def _init_smart(self, emb_d, prompt_param):
        
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
        self.task_count_f += 1
        if not self.task_id_bootstrap:
            for e in self.e_layers:
                f_ = getattr(self, f'freq_curr_{e}')
                f_ = f_ / torch.sum(f_)
                setattr(self, f'freq_past_{e}',torch.nn.Parameter(f_, requires_grad=False))


    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape ##SHAUN : x_querry has shape (batch_size,dim) e.g 16,768. It basically the CLS vec for each image 
            #print('Shapes : ',x_querry.shape)
            
            if self.expand_and_freeze:
                K = getattr(self,f'e_k_{l}')
                p = getattr(self,f'e_p_{l}')

                # freeze/control past tasks
                pt = self.e_pool_size / self.n_tasks
                s = int(self.task_count_f * pt)
                f = int((self.task_count_f + 1) * pt)
                
                if train:
                    if self.task_count_f > 0:
                        K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                        p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                    else:
                        K = K[s:f]
                        p = p[s:f]
                else:
                    K = K[0:f]
                    p = p[0:f]
                
            else:
                K = getattr(self,f'e_k_{l}') # 0 based indexing here
                p = getattr(self,f'e_p_{l}') # 0 based indexing here
            

            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            if train:

                # prompting
                if self.task_id_bootstrap:
                    loss = 1.0 - cos_sim[:,task_id].mean()  # the cosine similarity is always le 1
                    #print(p[task_id][:,0].shape)
                    #time.sleep(10)
                    P_ = p[task_id][:,0].expand(len(x_querry),-1,-1)
                else:
                    if self.task_count_f > 0:
                        f_ = getattr(self, f'freq_past_{l}')
                        f_tensor = f_.expand(B,-1)
                        # cos_sim_scaled = 1.0 - (f_tensor * (1.0 - cos_sim))
                        cos_sim_scaled = cos_sim
                    else:
                        cos_sim_scaled = cos_sim
                    top_k = torch.topk(cos_sim_scaled, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = 1.0 - cos_sim[:,k_idx].mean()  # the cosine similarity is always le 1
                    P_ = p[k_idx][:,0] ## SHAUN : This only chooses the top-1 prompt and not the top-k !!
                    # update frequency
                    f_ = getattr(self, f'freq_curr_{l}')
                    f_to_add = torch.bincount(k_idx.flatten().detach(),minlength=self.e_pool_size)
                    f_ += f_to_add
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                P_ = p[k_idx][:,0]
                
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
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)

    def _init_smart(self, emb_d, prompt_param):
        self.top_k = 5
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = []
        if prompt_param[2] > 0:
            self.e_layers = [0,1,3,4,5,6,7,8]
        else:
            self.e_layers = [0]

        if prompt_param[3] == 3:
            self.expand_and_freeze = True


        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = prompt_param[1]
        self.e_pool_size = prompt_param[0]

### SHAUN TODO : Instead of vit.py, add the architecture code in clip folder !

### SHAUN TODO : Add ImageEncoder class here 

class ImageEncoder(torch.nn.Module):
    def __init__(self, keep_lang=False):
        super().__init__()

        self.model, self.train_preprocess, self.val_preprocess = clip.load(
            "ViT-B/16", "cuda", jit=False)
        
        self.cache_dir = 'checkpoints'#args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images, prompt=None, q=None, train=False, feat_extraction=False):
        assert self.model is not None
        return self.model.encode_image(images,prompt=prompt,q=q,train=train,feat_extraction=feat_extraction)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)

class ImageClassifier(nn.Module):
    def __init__(self, image_encoder, last, prompt_flag, prompt_param, process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.last = last
        self.process_images = process_images
        self.prompt_flag = prompt_flag
        
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, prompt_param[0], prompt_param[1])
        else:
            self.prompt = None

    def forward(self, inputs,train=False):
        if self.prompt is not None:
            with torch.no_grad():
                q = self.image_encoder(inputs,feat_extraction=True)
            out, prompt_loss = self.image_encoder(inputs, prompt=self.prompt, q=q, train=train)
        else:
            out = self.image_encoder(inputs)
        outputs = self.last(out)
        if self.prompt is not None and train:
            return outputs, prompt_loss
        else:
            return outputs

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename) 

def clip_pt(out_dim,prompt_flag = None,prompt_param = None, template_style = 'openai_imagenet_template' ):
    
    #build and store ZS model from wise_ft stuff here. Return the ImageClassifier

    image_encoder = ImageEncoder(keep_lang=True)
    zeroshot_weights = get_zeroshot_classifier(image_encoder.model,template_style=template_style)
    last = ClassificationHead(normalize=True, weights=zeroshot_weights)
    delattr(image_encoder.model, 'transformer')
    return ImageClassifier(image_encoder, last, prompt_flag=prompt_flag, prompt_param=prompt_param)
