# This file is mainly from rabeehk/hyperformer
# https://github.com/rabeehk/hyperformer

"""Implementation of different utility functions for adapter layers."""

import torch
import torch.nn as nn
from transformers.activations import get_activation


class Activations(nn.Module):
    def __init__(self, activation_type):
        super().__init__()
        self.f = get_activation(activation_type)

    def forward(self, x):
        return self.f(x)


def init_linear_layer(linear_layer, std=1e-2):
    """Initializes the given linear module as explained in adapter paper."""
    nn.init.normal_(linear_layer.weight, std=std)
    if linear_layer.bias is not None:
        nn.init.zeros_(linear_layer.bias)


def zero_linear_layer(linear_layer):
    """Initializes the given linear module as explained in adapter paper."""
    nn.init.zeros_(linear_layer.weight)
    if linear_layer.bias is not None:
        nn.init.zeros_(linear_layer.bias)


def linear_layer(input_dim, output_dim, std=1e-2, use_bias=True, zeros_all=False):
    """Generates a linear module and initializes it."""
    linear = nn.Linear(input_dim, output_dim, bias=use_bias)
    if zeros_all:
        zero_linear_layer(linear)
    else:
        init_linear_layer(linear, std=std)
    return linear


class TaskHyperNet(nn.Module):
    """This module generates the task-embeddings from the initial feeded task embeddings."""

    def __init__(self, config):
        super(TaskHyperNet, self).__init__()
        self.task_hidden_dim = config.task_hidden_dim
        self.projected_task_embedding_dim = config.projected_task_embedding_dim
        self.task_embeding_generator = nn.Sequential(
            linear_layer(config.task_embedding_dim, self.task_hidden_dim),
            nn.ReLU(),
            linear_layer(self.task_hidden_dim, self.projected_task_embedding_dim))

    def forward(self, task_embedding):
        return self.task_embeding_generator(task_embedding)


class LayerNormHyperNet(nn.Module):
    """This module generates the weight and bias for the task conditioned layer norm."""

    def __init__(self, config):
        super(LayerNormHyperNet, self).__init__()
        self.task_embedding_dim = config.projected_task_embedding_dim \
            if config.train_task_embeddings else config.task_embedding_dim
        self.weight_generator = linear_layer(
            self.task_embedding_dim, config.input_dim)
        self.bias_generator = linear_layer(
            self.task_embedding_dim, config.input_dim)

    def forward(self, input):
        return self.weight_generator(input), self.bias_generator(input)


class TaskEmbeddingController(nn.Module):
    """Main module controlling task embeddings."""

    def __init__(self, config):
        super(TaskEmbeddingController, self).__init__()
        self.device = config.device
        self.task_embedding_dim = config.task_embedding_dim
        self.tasks = config.tasks
        self.task_to_task_embeddings = {task: task for task in self.tasks}
        if config.task_to_embeddings is not None:
            self.task_to_task_embeddings = config.task_to_embeddings
            self.tasks = self.task_to_task_embeddings.values()
        self.set_task_embeddings(self.tasks)
        self.train_task_embeddings = config.train_task_embeddings
        if self.train_task_embeddings:
            self.task_hyper_net = TaskHyperNet(config)

    def get_task(self, task):
        return self.task_to_task_embeddings[task]

    def set_task_embeddings(self, tasks):
        # self.task_to_embeddings = nn.ParameterDict(dict())
        self.task_to_embeddings = nn.ModuleDict(dict())
        # import pdb
        # pdb.set_trace()
        for task in tasks:
            task_embedding = torch.Tensor(torch.randn(
                self.task_embedding_dim)).to(self.device)
            task_embedding_module = nn.Embedding(
                1, self.task_embedding_dim).to(self.device)
            task_embedding_module.weight = nn.Parameter(task_embedding)
            self.task_to_embeddings[task] = task_embedding_module

    def forward(self, task):
        task_mapped = self.get_task(task)
        # import pdb
        # pdb.set_trace()
        task_embedding = self.task_to_embeddings[task_mapped].weight
        if self.train_task_embeddings:
            return self.task_hyper_net(task_embedding)

        return task_embedding


# class PHMRule_emb(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.device = config.device
#         self.phm_dim = config.hypercomplex_division
#         self.learn_phm = config.learn_phm
#         self.tasks = config.tasks
#         self.phm_c_init = config.phm_c_init
#         self.phm_init_range = config.phm_init_range
#         phmrule = torch.Tensor(torch.randn(self.phm_dim**3)).to(self.device)
#         phmrule_module = nn.Embedding(1, self.phm_dim**3).to(self.device)
#         phmrule_module.weight = nn.Parameter(phmrule)


# Embedding
class TaskPHMRULEController(nn.Module):
    """Main module controlling task embeddings."""

    def __init__(self, config):
        super(TaskPHMRULEController, self).__init__()
        self.device = config.device
        self.phm_dim = config.hypercomplex_division
        self.learn_phm = config.learn_phm
        self.tasks = config.tasks
        self.phm_c_init = config.phm_c_init
        self.phm_init_range = config.phm_init_range
        self.task_to_task_embeddings = {task: task for task in self.tasks}
        if config.task_to_embeddings is not None:
            self.task_to_task_embeddings = config.task_to_embeddings
            self.tasks = self.task_to_task_embeddings.values()
        self.set_task_phmrules(self.tasks)

    def get_task(self, task):
        return self.task_to_task_embeddings[task]

    def set_task_phmrules(self, tasks):
        # self.task_to_embeddings = nn.ParameterDict(dict())
        self.task_to_phmrules = nn.ModuleDict(dict())
        for task in tasks:

            phmrule = torch.Tensor(torch.randn(
                self.phm_dim**3)).to(self.device)

            phmrule_module = nn.Embedding(1, self.phm_dim**3).to(self.device)
            phmrule_module.weight = nn.Parameter(phmrule)

            # init
            if self.phm_c_init == "normal":
                phmrule_module.weight.data.normal_(
                    mean=0, std=self.phm_init_range)
            elif self.phm_c_init == "uniform":
                phmrule_module.weight.data.uniform_(-1, 1)
            else:
                raise NotImplementedError

            self.task_to_phmrules[task] = phmrule_module

    def get_phmrule(self, task):
        task_mapped = self.get_task(task)
        task_embedding = self.task_to_phmrules[task_mapped].weight
        task_phmrule = task_embedding.view(
            self.phm_dim, self.phm_dim, self.phm_dim)

        return task_phmrule

    def forward(self, task):
        task_mapped = self.get_task(task)
        task_embedding = self.task_to_phmrules[task_mapped].weight
        task_phmrule = task_embedding.view(
            self.phm_dim, self.phm_dim, self.phm_dim)

        return task_phmrule
