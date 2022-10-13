# This file is mainly from rabeehk/hyperformer
# https://github.com/rabeehk/hyperformer

"""Implements Adapter Controller, a module that keeps multiple
layers of Adapters, and controls which adapter layer to use."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation

from .adapter_configuration import AdapterConfig, MetaAdapterConfig
from .adapter_hypernetwork import Adapter, AdapterHyperNet
from .adapter_utils import LayerNormHyperNet
from .adapter_compacter import HyperComplexAdapter, LowRankAdapter
import os

# hyperformer + compacter


class AdapterController(nn.Module):
    """Implements Adapter controller module which controls the logics of
    putting adapter layers within transformer's layers."""

    def __init__(self, config, layer_id=None):
        super().__init__()
        self.config = config
        self.adapters = nn.ModuleDict(dict())
        # for output scaling
        self.adapter_output_scale = config.adapter_output_scale

        self.shared_adapter = config.shared_adapter
        if self.shared_adapter:
            self.tasks = ["shared"]
        else:
            self.tasks = config.tasks
        self.device = config.device
        self.layer_id = layer_id

        # adapterchain
        # only expand inputs if adapter_chain and not first layer of a block and use concate
        if layer_id:
            self.expand_inputs = self.config.adapterchain and layer_id[
                1] != 0 and self.config.adapter_in_fusion == "concat"
        else:
            self.expand_inputs = False

        # decide which kind of adapter (low-rank, hypercomplex?)
        self.low_rank_adapters = config.low_rank_adapters
        self.hypercomplex_adapters = config.hypercomplex_adapters
        self.cut_skip_pass = config.cut_skip_pass
        self.use_lora = config.use_lora

        self.task_to_adapter = {task: task for task in self.tasks}
        # If a dictionary from task to adapter is given, the task is over-written by the given adapters.
        if config.task_to_adapter is not None:
            self.task_to_adapter = config.task_to_adapter
            self.tasks = self.task_to_adapter.values()

        self.adapters = self.construct_adapters(self.tasks)

        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter

        # layer_norm in adapter branchs
        if self.add_layer_norm_before_adapter:
            input_dim = config.input_dim * 2 if self.expand_inputs else config.input_dim
            self.pre_layer_norm = nn.ModuleDict(dict())
            for task in self.tasks:
                self.pre_layer_norm[task] = nn.LayerNorm(input_dim)

        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.ModuleDict(dict())
            for task in self.tasks:
                self.post_layer_norm[task] = nn.LayerNorm(config.input_dim)

        # low-rank adapters
        # self.intrinsic_projections_path = os.path.join(
        #     config.output_dir, "intrinsic_projections")

        # compacter
        self.shared_phm_rule = config.shared_phm_rule

    def set_task_to_adapter_map(self, mapping):
        self.task_to_adapter = mapping

    def get_task(self, task):
        return self.task_to_adapter[task]

    def construct_adapters(self, tasks):
        """
        Constructs adapter layers and adds them to a dictionary for the given
        tasks.
        Args:
            tasks: A list of string containing the task names.
        """
        for task in tasks:
            if self.hypercomplex_adapters:
                self.adapters[task] = HyperComplexAdapter(
                    self.config, self.layer_id)
            elif self.low_rank_adapters:
                self.adapters[task] = LowRankAdapter(
                    self.config, self.layer_id)
            else:
                self.adapters[task] = Adapter(self.config, self.layer_id)
        return self.adapters

    def disable_adapters(self, tasks):
        """
        Given a list of tasks, it freezes their corresponding adapter layers'
        parameters.
        Args:
           tasks: List of tasks.
        """
        tasks = self.convert_to_list(tasks)
        for task in tasks:
            adapter = self.get_adapter(task)
            for param in adapter.parameters():
                param.requires_grad = False

    def convert_to_list(self, tasks):
        if isinstance(tasks, list):
            return tasks
        return [tasks]

    def enable_adapters(self, tasks):
        """
        Given a list of tasks, it unfreezes their corresponding adapter layers.
        Args:
            tasks: Given list of tasks.
        """
        tasks = self.convert_to_list(tasks)
        for task in tasks:
            adapter = self.get_adapter(task)
            for param in adapter.parameters():
                param.requires_grad = True

    def get_adapter(self, task):
        """Given a task returns its corresponding adapter layer.
        Args:
            task: Input task name.
        Returns:
            Adapter layer corresponding to the given task.
        """
        return self.adapters[task]

    def forward(self, task, inputs, task_embedding=None):
        """Retrieves the adapter layer corresponding to the given
        task. It freezes the adapter layers for all the other tasks
        and call the selected adapter layer.
        Args:
            task: the name of the current task.
            inputs: the inputs to feed in in the adapter layer.
        Returns:
            outputs of the adapter layer."""

        task = self.get_task(task)
        # Enables the adapter layer for the given task.
        self.enable_adapters(task)
        # Disable other adapters.
        other_tasks = [x for x in self.tasks if x != task]
        self.disable_adapters(other_tasks)
        adapter = self.get_adapter(task)

        if self.config.adapterchain and self.layer_id[1] != 0:
            h, last_adapter_out = inputs
            if self.config.adapter_in_fusion == "concat":
                adapter_in = torch.cat((h, last_adapter_out), dim=-1)
            elif self.config.adapter_in_fusion == "add":
                adapter_in = h + last_adapter_out
        else:
            h = inputs
            adapter_in = inputs

        z = self.pre_layer_norm[task](
            adapter_in) if self.add_layer_norm_before_adapter else h
        if self.hypercomplex_adapters:
            outputs = adapter(z, task_embedding)
        else:
            outputs = adapter(z)

        if self.add_layer_norm_after_adapter:
            outputs = self.post_layer_norm[task](outputs)

        if not self.cut_skip_pass:
            outputs = outputs + h

        # output scaling
        outputs = outputs * self.adapter_output_scale

        return outputs


# Adapter which does not have learnable parameters
# parameter/weights can given by the external weights
class MetaLayersAdapterController(nn.Module):
    """Implements Meta Adapter controller module, in which
    the adapter layers' weights are generated from a unique hyper-network."""

    def __init__(self, config):
        super().__init__()
        self.activation_type = config.non_linearity.lower()
        self.input_dim = config.input_dim
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        self.adapter_output_scale = config.adapter_output_scale

    def apply_layer_norm(self, inputs, layer_norm_weights):
        """Applies layer norm to the inputs."""
        return torch.nn.functional.layer_norm(inputs, (self.input_dim,),
                                              weight=layer_norm_weights.weight,
                                              bias=layer_norm_weights.bias)

    def call_adapter(self, inputs, adapter_weights):
        """Computes the output of the adapter layers."""
        # YC: input adapter weights and predict from inputs
        down = F.linear(inputs, weight=adapter_weights.down.weight,
                        bias=adapter_weights.down.bias)
        middle = get_activation(self.activation_type)(down)
        output = F.linear(middle, weight=adapter_weights.up.weight,
                          bias=adapter_weights.up.bias)
        return output

    def forward(self, inputs, adapter_weights):
        z = self.apply_layer_norm(
            inputs, adapter_weights.pre_norm) if self.add_layer_norm_before_adapter else inputs
        outputs = self.call_adapter(z, adapter_weights)
        if self.add_layer_norm_after_adapter:
            outputs = self.apply_layer_norm(outputs, adapter_weights.post_norm)
        outputs = outputs + inputs

        # output scaling
        outputs = outputs * self.adapter_output_scale

        return outputs


class AutoAdapterController(nn.Module):
    """Generic adapter controller class to instantiate different adapter
    controller classes."""

    @classmethod
    def get(cls, config, layer_id=None):
        if isinstance(config, MetaAdapterConfig):
            # deprecated ?
            return MetaAdapterController(config, layer_id)
        elif isinstance(config, AdapterConfig):
            return AdapterController(config, layer_id)
        raise ValueError("Unrecognized adapter config", config)


# deprecated
# class MetaAdapterController(nn.Module):
#     """Implements Meta Adapter controller module, in which
#     the adapter layers' weights are generated from a hyper-network.
#     In this case, task-embeddings are fixed, and the task
#     embeddings will be initialized to random."""

#     def __init__(self, config):
#         super().__init__()
#         self.device = config.device
#         self.adapters = nn.ModuleDict(dict())
#         self.config = config
#         self.input_dim = config.input_dim
#         self.down_sample_size = self.input_dim // config.reduction_factor
#         self.meta_up_sampler = AdapterHyperNet(
#             config, self.input_dim, self.down_sample_size)
#         self.meta_down_sampler = AdapterHyperNet(
#             config, self.down_sample_size, self.input_dim)
#         self.activation_type = config.non_linearity.lower()
#         self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
#         self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
#         self.conditional_layer_norm = config.conditional_layer_norm
#         if self.add_layer_norm_after_adapter:
#             if self.conditional_layer_norm:
#                 self.post_layernorm_hypernet = LayerNormHyperNet(config)
#             else:
#                 self.post_layer_norm = nn.LayerNorm(self.input_dim)
#         if self.add_layer_norm_before_adapter:
#             if self.conditional_layer_norm:
#                 self.pre_layernorm_hypernet = LayerNormHyperNet(config)
#             else:
#                 self.pre_layer_norm = nn.LayerNorm(self.input_dim)

#     def call_adapter(self, inputs, task_embedding):
#         weight_up, bias_up = self.meta_up_sampler(task_embedding)
#         weight_down, bias_down = self.meta_down_sampler(task_embedding)
#         down = F.linear(inputs, weight=weight_down, bias=bias_down)
#         middle = get_activation(self.activation_type)(down)
#         output = F.linear(middle, weight=weight_up, bias=bias_up)
#         return output

#     def apply_pre_layer_norm(self, inputs, task_embeddings):
#         """Applies pre layer norm to the inputs."""
#         if self.conditional_layer_norm:
#             weight, bias = self.pre_layernorm_hypernet(task_embeddings)
#             return torch.nn.functional.layer_norm(inputs, (self.input_dim,), weight=weight, bias=bias)
#         else:
#             return self.pre_layer_norm(inputs)

#     def apply_post_layer_norm(self, inputs, task_embeddings):
#         """Applies post layer norm to the inputs."""
#         if self.conditional_layer_norm:
#             weight, bias = self.post_layernorm_hypernet(task_embeddings)
#             return torch.nn.functional.layer_norm(inputs, (self.input_dim,), weight=weight, bias=bias)
#         else:
#             return self.post_layer_norm(inputs)

#     def forward(self, task_embedding, inputs):
#         """Retrieves the adapter layer corresponding to the given
#         task. It freezes the adapter layers for all the other tasks
#         and call the selected adapter layer.
#         Args:
#             task: the name of the current task.
#             inputs: the inputs to feed in in the adapter layer.
#         Returns:
#             outputs of the adapter layer."""
#         z = self.apply_pre_layer_norm(
#             inputs, task_embedding) if self.add_layer_norm_before_adapter else inputs
#         outputs = self.call_adapter(z, task_embedding)
#         if self.add_layer_norm_after_adapter:
#             outputs = self.apply_post_layer_norm(outputs, task_embedding)
#         outputs = outputs + inputs
#         return outputs
