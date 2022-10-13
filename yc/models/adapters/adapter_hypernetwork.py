# This file is mainly from rabeehk/hyperformer
# https://github.com/rabeehk/hyperformer

"""Implements an Adapter and Hyper-adapter Layers."""
from random import sample
import torch
import torch.nn as nn

from .adapter_outputs import (SamplerOutput, LayerNormOutput,
                              AdapterBlockOutput, AdapterOutput)
from .adapter_utils import Activations, linear_layer, LayerNormHyperNet, TaskHyperNet

from ..hypercomplex.kronecker import kronecker_product, kronecker_product_einsum_batched


class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config, layer_id=None):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.weight_init_range = config.weight_init_range
        self.use_lora = config.use_lora

        if self.use_lora:
            assert config.output_dim is not None
            self.output_dim = config.output_dim

            input_dim = self.input_dim
            output_dim = self.output_dim
            self.rank = config.low_rank_rank
            if self.rank < 0:
                self.rank = max(int(input_dim / -self.rank), 1)

            self.down_sampler = linear_layer(
                input_dim, self.rank,  std=self.weight_init_range, use_bias=False)
            self.up_sampler = linear_layer(
                self.rank, output_dim, use_bias=False, zeros_all=True)  # zero for B
            # and no activation function

        else:  # standard adapter
            self.down_sample_size = self.input_dim // config.reduction_factor
            self.activation = Activations(config.non_linearity.lower())

            self.expand_inputs = self.config.adapterchain and layer_id[
                1] != 0 and self.config.adapter_in_fusion == "concat"

            if self.expand_inputs:
                input_dim = self.input_dim * 2
            else:
                input_dim = self.input_dim

            self.down_sampler = linear_layer(
                input_dim, self.down_sample_size, std=self.weight_init_range)
            self.up_sampler = linear_layer(
                self.down_sample_size, self.input_dim, std=self.weight_init_range)

    def forward(self, x):
        z = self.down_sampler(x)
        if not self.use_lora:  # adapter use activation in hidden state
            z = self.activation(z)
        output = self.up_sampler(z)
        return output


class AdapterHyperNet(nn.Module):
    """This module generates the weights for the meta adapter layers."""

    def __init__(self, config, input_dim, output_dim):
        super(AdapterHyperNet, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_task_embeddings = config.train_task_embeddings
        self.task_embedding_dim = config.projected_task_embedding_dim if \
            config.train_task_embeddings else config.task_embedding_dim
        # Considers weight and bias parameters for generating adapter weights.
        self.weight_generator = nn.Sequential(
            linear_layer(self.task_embedding_dim, self.input_dim * self.output_dim))
        self.bias_generator = nn.Sequential(
            linear_layer(self.task_embedding_dim, self.input_dim))

    def forward(self, task_embedding):
        task_embedding = task_embedding.view(-1)
        weight = self.weight_generator(task_embedding).view(
            self.input_dim, self.output_dim)
        bias = self.bias_generator(task_embedding).view(-1)
        return weight, bias


class AdapterLayersHyperNet(nn.Module):
    """This module generates the weights for all the meta adapter layers
    given the task embeddings and layer id."""

    def __init__(self, config, input_dim, output_dim):
        super(AdapterLayersHyperNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.low_rank_adapters = config.low_rank_adapters
        if self.low_rank_adapters:
            rank = config.low_rank_rank
            self.rank = rank
            if rank < 0:
                self.rank = max(int(min(input_dim, output_dim) / -rank), 1)

        if self.low_rank_adapters:
            self.weight_generator = nn.Sequential(
                linear_layer(config.projected_task_embedding_dim, self.input_dim*self.rank + self.output_dim*self.rank))
            self.bias_generator = nn.Sequential(
                linear_layer(config.projected_task_embedding_dim, self.input_dim))
        else:
            self.weight_generator = nn.Sequential(
                linear_layer(config.projected_task_embedding_dim, self.input_dim * self.output_dim))
            self.bias_generator = nn.Sequential(
                linear_layer(config.projected_task_embedding_dim, self.input_dim))

    def forward(self, embeddings, batch_size=1):
        if self.low_rank_adapters:
            # YC: this might not be ideal in term of computation but better for implementation
            weight_low_rank = self.weight_generator(embeddings)
            if embeddings.ndim > 1:  # batch verison
                weight_in = weight_low_rank[:, :self.input_dim *
                                            self.rank].view(batch_size, self.input_dim, self.rank)
                weight_out = weight_low_rank[:, self.input_dim *
                                             self.rank:].view(batch_size, self.rank, self.output_dim)
            else:  # single image
                weight_in = weight_low_rank[:self.input_dim *
                                            self.rank].view(batch_size, self.input_dim, self.rank)
                weight_out = weight_low_rank[self.input_dim *
                                             self.rank:].view(batch_size, self.rank, self.output_dim)

            weight = torch.matmul(input=weight_in, other=weight_out)

            bias = self.bias_generator(embeddings).view(batch_size, -1)
        else:
            weight = self.weight_generator(embeddings).view(
                batch_size, self.input_dim, self.output_dim)
            bias = self.bias_generator(embeddings).view(batch_size, -1)

        sample_out_list = []
        for i in range(batch_size):
            sample_out_list.append(SamplerOutput(
                weight=weight[i], bias=bias[i]))

        if batch_size == 1:
            return sample_out_list[0]
        elif batch_size > 1:
            return sample_out_list


# hyperformer, hyperlowrank
class AdapterLayersHyperNetController(nn.Module):
    """This modules contains the hyper-nets for the feed forward
    and self-attention modules and it generates the adapter's weights and
    layer norm's weights for all the layers of transformers."""

    def __init__(self, config,):
        super(AdapterLayersHyperNetController, self).__init__()

        self.layer_norm_epsilon = 1e-6
        self.max_position_embeddings = 2
        self.device = config.device
        # self.device = "cuda"
        self.use_att_adapter = config.use_att_adapter
        self.task_embedding_dim = config.task_embedding_dim
        # self.layer_id_embeddings = nn.Embedding(self.num_layers,
        #                                         self.task_embedding_dim).to(self.device)
        # self.token_type_embeddings = nn.Embedding(self.max_position_embeddings,
        #                                          self.task_embedding_dim).to(self.device)
        # multiply 2 if layer_id + task_id
        config.task_embedding_dim = self.task_embedding_dim
        self.task_hypernet = TaskHyperNet(config)
        config.task_embedding_dim = self.task_embedding_dim
        self.unique_hyper_net_layer_norm = config.unique_hyper_net_layer_norm
        if self.unique_hyper_net_layer_norm:
            self.LayerNorm = nn.LayerNorm(
                config.projected_task_embedding_dim, eps=self.layer_norm_epsilon)
        self.input_dim = config.input_dim

        self.down_sample_size = self.input_dim // config.reduction_factor

        self.low_rank_adapters = config.low_rank_adapters

        # Defines the adapters hyper-nets.
        self.feed_forward_up_sampler_hyper_net = AdapterLayersHyperNet(config,
                                                                       self.input_dim, self.down_sample_size)
        self.feed_forward_down_sampler_hyper_net = AdapterLayersHyperNet(config,
                                                                         self.down_sample_size, self.input_dim)
        if self.use_att_adapter:
            self.self_attention_up_sampler_hyper_net = AdapterLayersHyperNet(config,
                                                                             self.input_dim, self.down_sample_size)
            self.self_attention_down_sampler_hyper_net = AdapterLayersHyperNet(config,
                                                                               self.down_sample_size, self.input_dim)

        # [LayerNorm HyperNets]
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        self.train_task_embeddings = config.train_task_embeddings
        config.train_task_embeddings = True
        if self.add_layer_norm_before_adapter:
            self.feed_forward_pre_layernorm_hypernet = LayerNormHyperNet(
                config)
            if self.use_att_adapter:
                self.self_attention_pre_layernorm_hypernet = LayerNormHyperNet(
                    config)
        if self.add_layer_norm_after_adapter:
            self.feed_forward_post_layernorm_hypernet = LayerNormHyperNet(
                config)
            if self.use_att_adapter:
                self.self_attention_post_layernorm_hypernet = LayerNormHyperNet(
                    config)
        config.train_task_embeddings = self.train_task_embeddings

    def get_embedding(self, task_embedding):
        """Concatenates the task embedding with the embedding for the layer id and
        returns the final joint embedding."""
        # layer_id_tensor = torch.tensor(
        #     [layer_id], dtype=torch.long, device=self.device)
        # layer_embedding = self.layer_id_embeddings(layer_id_tensor)
        # layer_embedding = layer_embedding.view(-1)
        # embeddings = torch.cat([task_embedding.view(
        #     1, -1), layer_embedding.view(1, -1)], axis=0)
        embeddings = task_embedding.view(1, -1)
        embeddings = self.task_hypernet(embeddings.view(-1))
        if self.unique_hyper_net_layer_norm:
            embeddings = self.LayerNorm(embeddings)
        return embeddings

    def forward(self, task_embedding):
        embeddings = self.get_embedding(task_embedding)

        # FFN layers
        feed_forward_down = self.feed_forward_down_sampler_hyper_net(
            embeddings)
        feed_forward_up = self.feed_forward_up_sampler_hyper_net(embeddings)
        feed_forward_output = AdapterOutput(
            up=feed_forward_up, down=feed_forward_down)
        if self.add_layer_norm_before_adapter:
            weight, bias = self.feed_forward_pre_layernorm_hypernet(embeddings)
            feed_forward_output.pre_norm = LayerNormOutput(
                weight=weight, bias=bias)
        if self.add_layer_norm_after_adapter:
            weight, bias = self.feed_forward_post_layernorm_hypernet(
                embeddings)
            feed_forward_output.post_norm = LayerNormOutput(
                weight=weight, bias=bias)

        # self-attention layers
        self_attention_output = None
        if self.use_att_adapter:
            self_attention_down = self.self_attention_down_sampler_hyper_net(
                embeddings)
            self_attention_up = self.self_attention_up_sampler_hyper_net(
                embeddings)
            self_attention_output = AdapterOutput(
                up=self_attention_up, down=self_attention_down)
            # Generates the weights and baises for pre and post layer norms.
            if self.add_layer_norm_before_adapter:
                weight, bias = self.self_attention_pre_layernorm_hypernet(
                    embeddings)
                self_attention_output.pre_norm = LayerNormOutput(
                    weight=weight, bias=bias)
            if self.add_layer_norm_after_adapter:
                weight, bias = self.self_attention_post_layernorm_hypernet(
                    embeddings)
                self_attention_output.post_norm = LayerNormOutput(
                    weight=weight, bias=bias)

        return AdapterBlockOutput(feed_forward=feed_forward_output,
                                  self_attention=self_attention_output)

# hyperformer++, hyperlowrank


class AdapterLayersOneHyperNetController(nn.Module):
    """This modules contains the hyper-nets for the feed forward
    and self-attention modules and it generates the adapter's weights and
    layer norm's weights for all the layers of transformers."""

    def __init__(self, config, num_layers=6):
        super(AdapterLayersOneHyperNetController, self).__init__()
        self.num_layers = num_layers
        self.layer_norm_epsilon = 1e-6
        self.max_position_embeddings = 2
        self.device = config.device
        self.use_att_adapter = config.use_att_adapter

        self.task_embedding_dim = config.task_embedding_dim

        # YC: original implementation: while this somewhat cannot run with dataparallel
        # self.layer_id_embeddings = nn.Embedding(self.num_layers,
        #                                         self.task_embedding_dim).to(self.device)

        # YC: newer verison, using moduledict to store layerembedding
        self.layer_id_embeddings = nn.ModuleDict(dict())
        for layer_id in range(num_layers):
            layer_embedding = torch.Tensor(torch.randn(
                self.task_embedding_dim)).to(self.device)

            layer_embedding_module = nn.Embedding(
                1, self.task_embedding_dim).to(self.device)

            layer_embedding_module.weight = nn.Parameter(layer_embedding)
            self.layer_id_embeddings[str(layer_id)] = layer_embedding_module

        # This is 2 types of adapters for feed-forward, and self-attention.
        # self.adapters_block_type = nn.Embedding(
        #     2, self.task_embedding_dim).to(self.device)
        if self.use_att_adapter:
            self.adapters_block_type = nn.ModuleDict(dict())
            for layer_id in range(2):
                adapters_block = torch.Tensor(torch.randn(
                    self.task_embedding_dim)).to(self.device)

                adapters_block_module = nn.Embedding(
                    1, self.task_embedding_dim).to(self.device)

                adapters_block_module.weight = nn.Parameter(adapters_block)
                self.adapters_block_type[str(layer_id)] = adapters_block_module
            config.task_embedding_dim = self.task_embedding_dim * 3
            # extra slot for pos id
        else:
            config.task_embedding_dim = self.task_embedding_dim * 2

        self.task_hypernet = TaskHyperNet(config)
        config.task_embedding_dim = self.task_embedding_dim
        self.unique_hyper_net_layer_norm = config.unique_hyper_net_layer_norm
        if self.unique_hyper_net_layer_norm:
            self.LayerNorm = nn.LayerNorm(
                config.projected_task_embedding_dim, eps=self.layer_norm_epsilon)
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor

        # Defines the adapters hyper-nets.
        self.up_sampler_hyper_net = AdapterLayersHyperNet(
            config, self.input_dim, self.down_sample_size)
        self.down_sampler_hyper_net = AdapterLayersHyperNet(
            config, self.down_sample_size, self.input_dim)

        # Defines the layer norms' hyper net.
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        self.train_task_embeddings = config.train_task_embeddings
        config.train_task_embeddings = True
        if self.add_layer_norm_before_adapter:
            self.pre_layernorm_hypernet = LayerNormHyperNet(config)
        if self.add_layer_norm_after_adapter:
            self.post_layernorm_hypernet = LayerNormHyperNet(config)
        config.train_task_embeddings = self.train_task_embeddings

    def get_embedding(self, task_embedding, layer_id, pos_id):
        """Concatenates the task embedding with the embedding for the layer id and
        returns the final joint embedding."""

        # YC: original implementation: while this somewhat cannot run with dataparallel
        # layer_id_tensor = torch.tensor(
        #     [layer_id], dtype=torch.long, device=self.device)
        # layer_embedding = self.layer_id_embeddings(layer_id_tensor)

        # YC: newer verison follows the task embedding

        layer_embedding = self.layer_id_embeddings[str(layer_id)].weight

        # type_id_tensor = torch.tensor(
        #     [block_type], dtype=torch.long, device=self.device)
        # type_embedding = self.adapters_block_type(type_id_tensor)
        layer_embedding = layer_embedding.view(-1)
        # type_embedding = type_embedding.view(-1)

        if self.use_att_adapter:
            type_embedding = self.adapters_block_type[str(pos_id)].weight
            embeddings = torch.cat([task_embedding.view(
                1, -1), layer_embedding.view(1, -1), type_embedding.view(1, -1)], axis=0)
        else:
            embeddings = torch.cat([task_embedding.view(
                1, -1), layer_embedding.view(1, -1)], axis=0)

        embeddings = self.task_hypernet(embeddings.view(-1))
        if self.unique_hyper_net_layer_norm:
            embeddings = self.LayerNorm(embeddings)
        return embeddings

    def forward(self, task_embedding, layer_id):
        # forward input: layer-id, task embedding
        # forward output: weights in adapters
        feed_forward_embeddings = self.get_embedding(
            task_embedding, layer_id, 0)
        # Generating of weights in adapters
        # Generates the adapters weights in feed-forward.
        feed_forward_down = self.down_sampler_hyper_net(
            feed_forward_embeddings)
        feed_forward_up = self.up_sampler_hyper_net(feed_forward_embeddings)
        feed_forward_output = AdapterOutput(
            up=feed_forward_up, down=feed_forward_down)

        if self.add_layer_norm_before_adapter:
            weight, bias = self.pre_layernorm_hypernet(feed_forward_embeddings)
            feed_forward_output.pre_norm = LayerNormOutput(
                weight=weight, bias=bias)

        if self.add_layer_norm_after_adapter:
            weight, bias = self.post_layernorm_hypernet(
                feed_forward_embeddings)
            feed_forward_output.post_norm = LayerNormOutput(
                weight=weight, bias=bias)

        self_attention_output = None
        # Generates the adapter weights in self-attention.
        if self.use_att_adapter:
            self_attention_embeddings = self.get_embedding(
                task_embedding, layer_id, 1)
            self_attention_down = self.down_sampler_hyper_net(
                self_attention_embeddings)
            self_attention_up = self.up_sampler_hyper_net(
                self_attention_embeddings)
            self_attention_output = AdapterOutput(
                up=self_attention_up, down=self_attention_down)

            # Generates the weights and baises for pre and post layer norms.
            if self.add_layer_norm_before_adapter:
                weight, bias = self.pre_layernorm_hypernet(
                    self_attention_embeddings)
                self_attention_output.pre_norm = LayerNormOutput(
                    weight=weight, bias=bias)

            if self.add_layer_norm_after_adapter:
                weight, bias = self.post_layernorm_hypernet(
                    self_attention_embeddings)
                self_attention_output.post_norm = LayerNormOutput(
                    weight=weight, bias=bias)

        return AdapterBlockOutput(feed_forward=feed_forward_output,
                                  self_attention=self_attention_output)


# hyperformer_kron
class AdapterLayersOneKroneckerHyperNetController(nn.Module):
    """This modules contains the hyper-nets for the feed forward
    and self-attention modules and it generates the adapter's weights and
    layer norm's weights for all the layers of transformers."""

    def __init__(self, config, depths=[]):
        super(AdapterLayersOneKroneckerHyperNetController, self).__init__()

        self.depths = depths
        self.num_layers = sum(depths)
        self.layer_norm_epsilon = 1e-6
        self.max_position_embeddings = 2
        self.device = config.device
        self.use_att_adapter = config.use_att_adapter  # not using now

        self.task_embedding_dim = config.task_embedding_dim

        # This is 2 types of adapters for feed-forward, and self-attention.
        if self.use_att_adapter:
            self.adapters_block_type = nn.ModuleDict(dict())
            for layer_id in range(2):
                adapters_block = torch.Tensor(torch.randn(1,
                                                          self.task_embedding_dim)).to(self.device)

                adapters_block_module = nn.Embedding(
                    1, self.task_embedding_dim).to(self.device)

                adapters_block_module.weight = nn.Parameter(adapters_block)
                self.adapters_block_type[str(layer_id)] = adapters_block_module
            config.task_embedding_dim = self.task_embedding_dim * 3
            # extra slot for pos id
        else:
            config.task_embedding_dim = self.task_embedding_dim * 2

        self.task_hypernet = TaskHyperNet(config)
        config.task_embedding_dim = self.task_embedding_dim
        self.unique_hyper_net_layer_norm = config.unique_hyper_net_layer_norm
        if self.unique_hyper_net_layer_norm:
            self.LayerNorm = nn.LayerNorm(
                config.projected_task_embedding_dim, eps=self.layer_norm_epsilon)
        self.input_dim = config.input_dim

        self.down_sample_size = self.input_dim // config.reduction_factor

        # Defines the adapters hyper-nets.
        self.up_sampler_hyper_net = AdapterLayersHyperNet(
            config, self.input_dim, self.down_sample_size)
        self.down_sampler_hyper_net = AdapterLayersHyperNet(
            config, self.down_sample_size, self.input_dim)

        # Defines the layer norms' hyper net.
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        self.train_task_embeddings = config.train_task_embeddings

        config.train_task_embeddings = True
        if self.add_layer_norm_before_adapter:
            self.pre_layernorm_hypernet = LayerNormHyperNet(config)

        if self.add_layer_norm_after_adapter:
            self.post_layernorm_hypernet = LayerNormHyperNet(config)
        config.train_task_embeddings = self.train_task_embeddings

        # addiitonal learnable parameters (layer-wise scale kernel, bias kernel, layernorm kernel)
        self.scale = [1, 2, 4, 8]
        self.per_layer_scale = []

        self.kernel_front = config.kernel_front

        # ffn
        self.layer_id_embeddings = nn.ModuleDict(dict())
        self.scaleup_kronkernel = nn.ModuleDict(dict())
        self.coef_bias = nn.ModuleDict(dict())
        if self.use_att_adapter:
            self.scaleup_kronkernel_att = nn.ModuleDict(dict())
            self.coef_bias_att = nn.ModuleDict(dict())

        if self.add_layer_norm_before_adapter:
            self.coef_lnb = nn.ModuleDict(dict())
            if self.use_att_adapter:
                self.coef_lnb_att = nn.ModuleDict(dict())

        if self.add_layer_norm_after_adapter:
            self.coef_lna = nn.ModuleDict(dict())
            if self.use_att_adapter:
                self.coef_lna_att = nn.ModuleDict(dict())

        layer_id = 0
        for block_id in range(len(self.depths)):
            for _ in range(self.depths[block_id]):
                # layer_embedding (since num of embedding change wrt the scale)
                layer_embedding = torch.Tensor(torch.randn(
                    self.scale[block_id], self.task_embedding_dim)).to(self.device)
                layer_embedding_module = nn.Embedding(
                    self.scale[block_id], self.task_embedding_dim).to(self.device)
                layer_embedding_module.weight = nn.Parameter(layer_embedding)
                self.layer_id_embeddings[str(
                    layer_id)] = layer_embedding_module

                # scale-up kronecker constant
                if self.scale[block_id] != 1:
                    # kernel
                    kernel = torch.Tensor(torch.randn(
                        self.scale[block_id]**3)).to(self.device)
                    kernel_module = nn.Embedding(
                        1, self.scale[block_id]**3).to(self.device)
                    kernel_module.weight = nn.Parameter(kernel)
                    self.scaleup_kronkernel[str(layer_id)] = kernel_module
                    if self.use_att_adapter:
                        kernel_att = torch.Tensor(torch.randn(
                            self.scale[block_id]**3)).to(self.device)
                        kernel_module_att = nn.Embedding(
                            1, self.scale[block_id]**3).to(self.device)
                        kernel_module_att.weight = nn.Parameter(kernel_att)
                        self.scaleup_kronkernel_att[str(
                            layer_id)] = kernel_module_att

                    # coef bias
                    coef_b = torch.Tensor(torch.randn(
                        self.scale[block_id]**2)).to(self.device)
                    coef_b_module = nn.Embedding(
                        1, self.scale[block_id]**2).to(self.device)
                    coef_b_module.weight = nn.Parameter(coef_b)
                    self.coef_bias[str(layer_id)] = coef_b_module
                    if self.use_att_adapter:
                        coef_b_att = torch.Tensor(torch.randn(
                            self.scale[block_id]**2)).to(self.device)
                        coef_b_module_att = nn.Embedding(
                            1, self.scale[block_id]**2).to(self.device)
                        coef_b_module_att.weight = nn.Parameter(coef_b_att)
                        self.coef_bias_att[str(layer_id)] = coef_b_module_att

                    if self.add_layer_norm_before_adapter:
                        coef_lnb = torch.Tensor(torch.randn(
                            self.scale[block_id]**2)).to(self.device)
                        coef_lnb_module = nn.Embedding(
                            1, self.scale[block_id]**2).to(self.device)
                        coef_lnb_module.weight = nn.Parameter(coef_lnb)
                        self.coef_lnb[str(layer_id)] = coef_lnb_module
                        if self.use_att_adapter:
                            coef_lnb_att = torch.Tensor(torch.randn(
                                self.scale[block_id]**2)).to(self.device)
                            coef_lnb_module_att = nn.Embedding(
                                1, self.scale[block_id]**2).to(self.device)
                            coef_lnb_module_att.weight = nn.Parameter(
                                coef_lnb_att)
                            self.coef_lnb_att[str(
                                layer_id)] = coef_lnb_module_att

                    if self.add_layer_norm_after_adapter:
                        coef_lna = torch.Tensor(torch.randn(
                            self.scale[block_id]**2)).to(self.device)
                        coef_lna_module = nn.Embedding(
                            1, self.scale[block_id]**2).to(self.device)
                        coef_lna_module.weight = nn.Parameter(coef_lna)
                        self.coef_lna[str(layer_id)] = coef_lna_module
                        if self.use_att_adapter:
                            coef_lna_att = torch.Tensor(torch.randn(
                                self.scale[block_id]**2)).to(self.device)
                            coef_lna_module_att = nn.Embedding(
                                1, self.scale[block_id]**2).to(self.device)
                            coef_lna_module_att.weight = nn.Parameter(
                                coef_lna_att)
                            self.coef_lna_att[str(
                                layer_id)] = coef_lna_module_att

                else:  # first block does not need scale up
                    self.scaleup_kronkernel[str(layer_id)] = None
                    self.coef_bias[str(layer_id)] = None
                    if self.use_att_adapter:
                        self.scaleup_kronkernel_att[str(layer_id)] = None
                        self.coef_bias_att[str(layer_id)] = None

                    if self.add_layer_norm_before_adapter:
                        self.coef_lnb[str(layer_id)] = None
                        if self.use_att_adapter:
                            self.coef_lnb_att[str(layer_id)] = None

                    if self.add_layer_norm_after_adapter:
                        self.coef_lna[str(layer_id)] = None
                        if self.use_att_adapter:
                            self.coef_lna_att[str(layer_id)] = None

                # coefficient to combine bias
                self.per_layer_scale.append(self.scale[block_id])
                layer_id += 1

    def get_embedding(self, task_embedding, layer_id, pos_id):
        """Concatenates the task embedding with the embedding for the layer id and
        returns the final joint embedding."""

        # YC: original implementation: while this somewhat cannot run with dataparallel
        # layer_id_tensor = torch.tensor(
        #     [layer_id], dtype=torch.long, device=self.device)
        # layer_embedding = self.layer_id_embeddings(layer_id_tensor)
        # YC: newer verison follows the task embedding
        layer_embedding = self.layer_id_embeddings[str(layer_id)].weight
        batch_size = layer_embedding.shape[0]
        task_embedding = task_embedding.view(1, -1).expand(batch_size, -1)
        # type_id_tensor = torch.tensor(
        #     [block_type], dtype=torch.long, device=self.device)
        # type_embedding = self.adapters_block_type(type_id_tensor)
        # layer_embedding = layer_embedding.view(-1)
        # type_embedding = type_embedding.view(-1)

        if self.use_att_adapter:
            type_embedding = self.adapters_block_type[str(pos_id)].weight
            type_embedding = type_embedding.view(1, -1).expand(batch_size, -1)
            embeddings = torch.cat(
                [task_embedding, layer_embedding, type_embedding], axis=1)
        else:
            embeddings = torch.cat([task_embedding, layer_embedding], axis=1)
        embeddings = self.task_hypernet(embeddings)
        if self.unique_hyper_net_layer_norm:
            embeddings = self.LayerNorm(embeddings)
        return embeddings

    def get_scaleup_kernel(self, layer_id, name=None, pos_name="ffn"):
        kdim = self.per_layer_scale[layer_id]

        if pos_name == "ffn":
            if name == "W":
                assert self.scaleup_kronkernel[str(layer_id)] is not None
                kronkernel = self.scaleup_kronkernel[str(layer_id)].weight
                kronkernel = kronkernel.view(kdim, kdim, kdim)
            elif name == "B":
                assert self.coef_bias[str(layer_id)] is not None
                kronkernel = self.coef_bias[str(layer_id)].weight
                kronkernel = kronkernel.view(kdim, kdim, 1)
            elif name == "preln":
                assert self.coef_lnb[str(layer_id)] is not None
                kronkernel = self.coef_lnb[str(layer_id)].weight
                kronkernel = kronkernel.view(kdim, kdim, 1)
            elif name == "postln":
                assert self.coef_lna[str(layer_id)] is not None
                kronkernel = self.coef_lna[str(layer_id)].weight
                kronkernel = kronkernel.view(kdim, kdim, 1)

            else:
                raise ValueError('Unknown kernel name')
        elif pos_name == "att":
            if name == "W":
                assert self.scaleup_kronkernel_att[str(layer_id)] is not None
                kronkernel = self.scaleup_kronkernel_att[str(layer_id)].weight
                kronkernel = kronkernel.view(kdim, kdim, kdim)
            elif name == "B":
                assert self.coef_bias_att[str(layer_id)] is not None
                kronkernel = self.coef_bias_att[str(layer_id)].weight
                kronkernel = kronkernel.view(kdim, kdim, 1)
            elif name == "preln":
                assert self.coef_lnb_att[str(layer_id)] is not None
                kronkernel = self.coef_lnb_att[str(layer_id)].weight
                kronkernel = kronkernel.view(kdim, kdim, 1)
            elif name == "postln":
                assert self.coef_lna_att[str(layer_id)] is not None
                kronkernel = self.coef_lna_att[str(layer_id)].weight
                kronkernel = kronkernel.view(kdim, kdim, 1)

            else:
                raise ValueError('Unknown kernel name')
        else:
            raise ValueError('Unknown position name')

        return kronkernel

    def integrate_adapter_weights(self, adapter_weights_list, layer_id, pos_name="ffn"):
        kernel_W = self.get_scaleup_kernel(layer_id, "W", pos_name)
        kernel_B = self.get_scaleup_kernel(layer_id, "B", pos_name)
        # coef_bias = self.coef_bias[str(layer_id)].weight.view(1, -1)

        weights_list = []
        bias_list = []
        for adaw in adapter_weights_list:
            weights_list.append(torch.unsqueeze(adaw.weight, 0))
            bias_list.append(torch.unsqueeze(adaw.bias, 0))
        weight_multi = torch.cat(weights_list, dim=0)
        bias_multi = torch.cat(bias_list, dim=0)
        bias_multi = torch.unsqueeze(bias_multi, -1)

        if self.kernel_front:
            weight = kronecker_product_einsum_batched(
                kernel_W, weight_multi).sum(0)
            bias = kronecker_product_einsum_batched(
                kernel_B, bias_multi).sum(0)

        else:
            weight = kronecker_product_einsum_batched(
                weight_multi, kernel_W).sum(0)
            bias = kronecker_product_einsum_batched(
                bias_multi, kernel_B).sum(0)
        bias = torch.squeeze(bias)

        return SamplerOutput(weight=weight, bias=bias)

    def integrate_layernorm_weights(self, weight, bias, layer_id, name=None, pos_name="ffn"):

        kernel = self.get_scaleup_kernel(layer_id, name, pos_name)
        # coef_bias = self.coef_bias[str(layer_id)].weight.view(1, -1)

        weight_multi = torch.unsqueeze(weight, -1)
        bias_multi = torch.unsqueeze(bias, -1)

        # bias = torch.matmul(coef_bias, bias_multi).view(-1)

        if self.kernel_front:
            weight = kronecker_product_einsum_batched(
                kernel, weight_multi).sum(0)
            bias = kronecker_product_einsum_batched(kernel, bias_multi).sum(0)

        else:
            weight = kronecker_product_einsum_batched(
                weight_multi, kernel).sum(0)
            bias = kronecker_product_einsum_batched(bias_multi, kernel).sum(0)
        weight = torch.squeeze(weight)
        bias = torch.squeeze(bias)

        return weight, bias

    def forward(self, task_embedding, layer_id):
        # forward input: layer-id, task embedding
        # forward output: weights in adapters

        feed_forward_embeddings = self.get_embedding(
            task_embedding, layer_id, 0)
        batch_size = feed_forward_embeddings.shape[0]

        # Generating of weights in adapters
        # Generates the adapters weights in feed-forward.
        feed_forward_down = self.down_sampler_hyper_net(
            feed_forward_embeddings, batch_size=batch_size)
        feed_forward_up = self.up_sampler_hyper_net(
            feed_forward_embeddings, batch_size=batch_size)

        # if scale > 1: use kronecker product
        if type(feed_forward_down) is list and type(feed_forward_up) is list:
            feed_forward_down = self.integrate_adapter_weights(
                feed_forward_down, layer_id, pos_name="ffn")
            feed_forward_up = self.integrate_adapter_weights(
                feed_forward_up, layer_id, pos_name="ffn")

        feed_forward_output = AdapterOutput(
            up=feed_forward_up, down=feed_forward_down)

        if self.add_layer_norm_before_adapter:
            weight, bias = self.pre_layernorm_hypernet(feed_forward_embeddings)
            if weight.ndim > 1:
                # if scale > 1: use kronecker product
                if weight.shape[0] > 1:
                    weight, bias = self.integrate_layernorm_weights(
                        weight, bias, layer_id, name="preln", pos_name="ffn")
                else:  # scale = 1
                    weight = torch.squeeze(weight)
                    bias = torch.squeeze(bias)

            feed_forward_output.pre_norm = LayerNormOutput(
                weight=weight, bias=bias)

        if self.add_layer_norm_after_adapter:
            weight, bias = self.post_layernorm_hypernet(
                feed_forward_embeddings)
            if weight.ndim > 1:
                # if scale > 1: use kronecker product
                if weight.shape[0] > 1:
                    weight, bias = self.integrate_layernorm_weights(
                        weight, bias, layer_id, name="postln", pos_name="ffn")
                else:  # scale = 1
                    weight = torch.squeeze(weight)
                    bias = torch.squeeze(bias)

            feed_forward_output.post_norm = LayerNormOutput(
                weight=weight, bias=bias)

        self_attention_output = None
        #  ================  [self-attention]  ===================
        if self.use_att_adapter:
            self_attention_embeddings = self.get_embedding(
                task_embedding, layer_id, 1)
            self_attention_down = self.down_sampler_hyper_net(
                self_attention_embeddings, batch_size=batch_size)
            self_attention_up = self.up_sampler_hyper_net(
                self_attention_embeddings, batch_size=batch_size)

            # if scale > 1: use kronecker product
            if type(self_attention_down) is list and type(self_attention_up) is list:
                self_attention_down = self.integrate_adapter_weights(
                    self_attention_down, layer_id, pos_name="att")
                self_attention_up = self.integrate_adapter_weights(
                    self_attention_up, layer_id, pos_name="att")

            self_attention_output = AdapterOutput(
                up=self_attention_up, down=self_attention_down)

            # Generates the weights and baises for pre and post layer norms.
            if self.add_layer_norm_before_adapter:
                weight, bias = self.pre_layernorm_hypernet(
                    self_attention_embeddings)
                if weight.ndim > 1:
                    # if scale > 1: use kronecker product
                    if weight.shape[0] > 1:
                        weight, bias = self.integrate_layernorm_weights(
                            weight, bias, layer_id, name="preln", pos_name="att")
                    else:  # scale = 1
                        weight = torch.squeeze(weight)
                        bias = torch.squeeze(bias)

                self_attention_output.pre_norm = LayerNormOutput(
                    weight=weight, bias=bias)

            if self.add_layer_norm_after_adapter:
                weight, bias = self.post_layernorm_hypernet(
                    self_attention_embeddings)
                if weight.ndim > 1:
                    # if scale > 1: use kronecker product
                    if weight.shape[0] > 1:
                        weight, bias = self.integrate_layernorm_weights(
                            weight, bias, layer_id, name="postln", pos_name="att")
                    else:  # scale = 1
                        weight = torch.squeeze(weight)
                        bias = torch.squeeze(bias)
                self_attention_output.post_norm = LayerNormOutput(
                    weight=weight, bias=bias)

        return AdapterBlockOutput(feed_forward=feed_forward_output,
                                  self_attention=self_attention_output)


# class AdapterLayersHyperNetController(nn.Module):
#     """This modules contains the hyper-nets for the feed forward
#     and self-attention modules and it generates the adapter's weights and
#     layer norm's weights for all the layers of transformers."""

#     def __init__(self, config, num_layers=6):
#         super(AdapterLayersHyperNetController, self).__init__()

#         import pdb
#         pdb.set_trace()
#         self.num_layers = num_layers
#         self.layer_norm_epsilon = 1e-6
#         self.max_position_embeddings = 2
#         # self.device = config.device
#         self.device = "cuda"

#         self.task_embedding_dim = config.task_embedding_dim
#         self.layer_id_embeddings = nn.Embedding(self.num_layers,
#                                                 self.task_embedding_dim).to(self.device)
#         # self.token_type_embeddings = nn.Embedding(self.max_position_embeddings,
#         #                                          self.task_embedding_dim).to(self.device)
#         config.task_embedding_dim = self.task_embedding_dim * 2
#         self.task_hypernet = TaskHyperNet(config)
#         config.task_embedding_dim = self.task_embedding_dim
#         self.unique_hyper_net_layer_norm = config.unique_hyper_net_layer_norm
#         if self.unique_hyper_net_layer_norm:
#             self.LayerNorm = nn.LayerNorm(
#                 config.projected_task_embedding_dim, eps=self.layer_norm_epsilon)
#         self.input_dim = config.input_dim
#         self.down_sample_size = self.input_dim // config.reduction_factor
#         # Defines the adapters hyper-nets.
#         self.feed_forward_up_sampler_hyper_net = AdapterLayersHyperNet(config,
#                                                                        self.input_dim, self.down_sample_size)
#         self.feed_forward_down_sampler_hyper_net = AdapterLayersHyperNet(config,
#                                                                          self.down_sample_size, self.input_dim)
#         self.self_attention_up_sampler_hyper_net = AdapterLayersHyperNet(config,
#                                                                          self.input_dim, self.down_sample_size)
#         self.self_attention_down_sampler_hyper_net = AdapterLayersHyperNet(config,
#                                                                            self.down_sample_size, self.input_dim)
#         # Defines the layer norms' hyper net.
#         self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
#         self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
#         self.train_task_embeddings = config.train_task_embeddings
#         config.train_task_embeddings = True
#         if self.add_layer_norm_before_adapter:
#             self.feed_forward_pre_layernorm_hypernet = LayerNormHyperNet(
#                 config)
#             self.self_attention_pre_layernorm_hypernet = LayerNormHyperNet(
#                 config)
#         if self.add_layer_norm_after_adapter:
#             self.feed_forward_post_layernorm_hypernet = LayerNormHyperNet(
#                 config)
#             self.self_attention_post_layernorm_hypernet = LayerNormHyperNet(
#                 config)
#         config.train_task_embeddings = self.train_task_embeddings

#     def get_embedding(self, task_embedding, layer_id):
#         """Concatenates the task embedding with the embedding for the layer id and
#         returns the final joint embedding."""
#         layer_id_tensor = torch.tensor(
#             [layer_id], dtype=torch.long, device=self.device)
#         layer_embedding = self.layer_id_embeddings(layer_id_tensor)
#         layer_embedding = layer_embedding.view(-1)
#         embeddings = torch.cat([task_embedding.view(
#             1, -1), layer_embedding.view(1, -1)], axis=0)
#         embeddings = self.task_hypernet(embeddings.view(-1))
#         if self.unique_hyper_net_layer_norm:
#             embeddings = self.LayerNorm(embeddings)
#         return embeddings

#     def forward(self, task_embedding, layer_id):
#         embeddings = self.get_embedding(task_embedding, layer_id)
#         # Generates the adapters weights in feed-forward and self-attention modules.
#         feed_forward_down = self.feed_forward_down_sampler_hyper_net(
#             embeddings)
#         feed_forward_up = self.feed_forward_up_sampler_hyper_net(embeddings)
#         self_attention_down = self.self_attention_down_sampler_hyper_net(
#             embeddings)
#         self_attention_up = self.self_attention_up_sampler_hyper_net(
#             embeddings)
#         feed_forward_output = AdapterOutput(
#             up=feed_forward_up, down=feed_forward_down)
#         self_attention_output = AdapterOutput(
#             up=self_attention_up, down=self_attention_down)
#         # Generates the weights and baises for pre and post layer norms.
#         if self.add_layer_norm_before_adapter:
#             weight, bias = self.feed_forward_pre_layernorm_hypernet(embeddings)
#             feed_forward_output.pre_norm = LayerNormOutput(
#                 weight=weight, bias=bias)
#             weight, bias = self.self_attention_pre_layernorm_hypernet(
#                 embeddings)
#             self_attention_output.pre_norm = LayerNormOutput(
#                 weight=weight, bias=bias)
#         if self.add_layer_norm_after_adapter:
#             weight, bias = self.feed_forward_post_layernorm_hypernet(
#                 embeddings)
#             feed_forward_output.post_norm = LayerNormOutput(
#                 weight=weight, bias=bias)
#             weight, bias = self.self_attention_post_layernorm_hypernet(
#                 embeddings)
#             self_attention_output.post_norm = LayerNormOutput(
#                 weight=weight, bias=bias)
#         return AdapterBlockOutput(feed_forward=feed_forward_output,
#                                   self_attention=self_attention_output)
