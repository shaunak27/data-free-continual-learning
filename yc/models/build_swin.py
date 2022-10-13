# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from email.policy import strict
from .swin_transformer import SwinTransformer
import torch


def swin_model(para):
    # TODO: check input RGB or BGR?
    # TODO: swin transformer feature map size
    # TODO:
    # RuntimeError: Error(s) in loading state_dict for SwinTransformer:
    #         Missing key(s) in state_dict: "norm0.weight", "norm0.bias", "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "norm3.weight", "norm3.bias".
    #         Unexpected key(s) in state_dict: "norm.weight", "norm.bias", "head.weight", "head.bias", "layers.0.blocks.1.attn_mask", "layers.1.blocks.1.attn_mask", "layers.2.blocks.1.attn_mask", "layers.2.blocks.3.attn_mask", "layers.2.blocks.5.attn_mask".    model_type = para['backbone']
    model_type = para['backbone']

    if model_type == 'swin_tiny':
        model = SwinTransformer(para=para,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                drop_path_rate=0.3,
                                ape=False,
                                patch_norm=True,
                                frozen_stages=-1,
                                use_checkpoint=False)
        if para['head'] == 'allmlp':
            backbone_channels = [96, 192, 384, 768]
        else:  # deeplab
            backbone_channels = 768
        weight_path = "pretrain/swin_tiny_patch4_window7_224.pth"

    elif model_type == 'swin_base1k':
        model = SwinTransformer(para=para,
                                embed_dim=128,
                                depths=[2, 2, 18, 2],
                                num_heads=[4, 8, 16, 32],
                                window_size=7,
                                drop_path_rate=0.5,
                                ape=False,
                                patch_norm=True,
                                frozen_stages=-1,
                                use_checkpoint=False)
        if para['head'] == 'allmlp':
            backbone_channels = [128, 256, 512, 1024]
        else:  # deeplab
            backbone_channels = 1024
        weight_path = "pretrain/swin_base_patch4_window7_224.pth"

    elif model_type == 'swin_base22k':
        model = SwinTransformer(para=para,
                                embed_dim=128,
                                depths=[2, 2, 18, 2],
                                num_heads=[4, 8, 16, 32],
                                window_size=7,
                                drop_path_rate=0.2,
                                ape=False,
                                patch_norm=True,
                                frozen_stages=-1,
                                use_checkpoint=False)
        if para['head'] == 'allmlp':
            backbone_channels = [128, 256, 512, 1024]
        else:  # deeplab
            backbone_channels = 1024
        weight_path = "pretrain/swin_base_patch4_window7_224_22k.pth"

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    pretrained = para['backbone_kwargs']['pretrained']
    if pretrained:
        checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = checkpoint['model']

        model.load_state_dict(state_dict, strict=False)
        print("Swin Petrained weight loaded")
    return model, backbone_channels
