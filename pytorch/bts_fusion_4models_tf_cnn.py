# Copyright (C) 2020 Guanglei Yang
#
# This file is a part of PGA
# add btsnet with attention .
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import torch
import torch.nn as nn
import torch.nn.functional as NF
import math
from base_models.AttentionGraphCondKernel import AttentionGraphCondKernel
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from base_models.bts_resnet import BtsModel_Res
from base_models.bts_densenet import BtsModel_Dense
from base_models.bts_resnext import BtsModel_Resnext
from base_models.bts_trans import BtsModel_Trans
from base_models.bts_FCT import BtsModel_FCT
from base_models.bts_Fvolo import BtsModel_FVolo
from util import correlate
import numpy as np


class SpatialTransformer(nn.Module):

    def __init__(self, size, mode='bilinear'):
        """
        Instiantiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids) # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the source image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2*(new_locs[:, i, ...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return NF.grid_sample(src, new_locs, mode=self.mode, padding_mode="border")


class GRU(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size):
        super(GRU, self).__init__()

        # filters used for gates
        gru_input_channel = input_channel + output_channel
        self.output_channel = output_channel

        self.gate_conv = nn.Conv2d(gru_input_channel, output_channel * 2, kernel_size, padding=1)
        self.reset_gate_norm = nn.GroupNorm(1, output_channel, 1e-5, True)
        self.update_gate_norm = nn.GroupNorm(1, output_channel, 1e-5, True)

        # filters used for outputs
        self.output_conv = nn.Conv2d(gru_input_channel, output_channel, kernel_size, padding=1)
        self.output_norm = nn.GroupNorm(1, output_channel, 1e-5, True)

        self.activation = nn.Tanh()

    def gates(self, x, h):
        # x = N x C x H x W
        # h = N x C x H x W

        # c = N x C*2 x H x W
        c = torch.cat((x, h), dim=1)
        f = self.gate_conv(c)

        # r = reset gate, u = update gate
        # both are N x O x H x W
        C = f.shape[1]
        r, u = torch.split(f, C // 2, 1)

        rn = self.reset_gate_norm(r)
        un = self.update_gate_norm(u)
        rns = NF.sigmoid(rn)
        uns = NF.sigmoid(un)
        return rns, uns

    def output(self, x, h, r, u):
        f = torch.cat((x, r * h), dim=1)
        o = self.output_conv(f)
        on = self.output_norm(o)
        return on

    def forward(self, x, h=None):
        N, C, H, W = x.shape
        HC = self.output_channel
        if(h is None):
            h = torch.zeros((N, HC, H, W), dtype=torch.float, device=x.device)
        r, u = self.gates(x, h)
        o = self.output(x, h, r, u)
        y = self.activation(o)
        return u * h + (1 - u) * y


class BtsModel_Fusion(nn.Module):
    def __init__(self, params):
        super(BtsModel_Fusion, self).__init__()
        self.params = params
        # base model

        self.resnet = BtsModel_Res(self.params)
        self.resnet = torch.nn.DataParallel(self.resnet)
        checkpoint = torch.load("/mnt/data/liran/TransDepth-main/pytorch/stage1_nyu/resnet-134500-best_rms_0.40122",
                                map_location='cuda:{}'.format(self.params.gpu))
        self.resnet.load_state_dict(checkpoint['model'])
        for parameter in self.resnet.parameters():
            parameter.requires_grad = False

        self.densenet = BtsModel_Dense(self.params)
        self.densenet = torch.nn.DataParallel(self.densenet)
        checkpoint = torch.load("/mnt/data/liran/TransDepth-main/pytorch/stage1_nyu/densenet-140000-best_rms_0.39198",
                                map_location='cuda:{}'.format(self.params.gpu))
        self.densenet.load_state_dict(checkpoint['model'])
        for parameter in self.densenet.parameters():
            parameter.requires_grad = False

        self.resnext = BtsModel_Resnext(self.params)
        self.resnext = torch.nn.DataParallel(self.resnext)
        checkpoint = torch.load("/mnt/data/liran/TransDepth-main/pytorch/stage1_nyu/resnext-118000-best_rms_0.40210" ,
                                map_location='cuda:{}'.format(self.params.gpu))
        self.resnext.load_state_dict(checkpoint['model'])
        for parameter in self.resnext.parameters():
            parameter.requires_grad = False

        self.transformer = BtsModel_Trans(self.params)
        self.transformer = torch.nn.DataParallel(self.transformer)
        checkpoint = torch.load("/mnt/data/liran/TransDepth-main/pytorch/stage1_nyu/R_50+Vit-16-229000-best_rms_0.37511",
                                map_location='cuda:{}'.format(self.params.gpu))
        self.transformer.load_state_dict(checkpoint['model'])
        for parameter in self.transformer.parameters():
            parameter.requires_grad = False

        self.conformer = BtsModel_FCT(self.params)
        self.conformer = torch.nn.DataParallel(self.conformer)
        checkpoint = torch.load("/mnt/data/liran/TransDepth-main/pytorch/stage1_nyu/conformer-147000-best_rms_0.38184",
                                map_location='cuda:{}'.format(self.params.gpu))
        self.conformer.load_state_dict(checkpoint['model'])
        for parameter in self.conformer.parameters():
            parameter.requires_grad = False

        self.volo = BtsModel_FVolo(self.params)
        self.volo = torch.nn.DataParallel(self.volo)
        checkpoint = torch.load("/mnt/data/liran/TransDepth-main/pytorch/stage1_nyu/volo-104000-best_rms_0.37576",
                                map_location='cuda:{}'.format(self.params.gpu))
        self.volo.load_state_dict(checkpoint['model'])
        for parameter in self.volo.parameters():
            parameter.requires_grad = False

        # model to fuse
        gru_input_size = 32
        gru1_output_size = 16
        gru2_output_size = 8
        gru3_output_size = 4
        self.gru1 = GRU(gru_input_size, gru1_output_size, 3)
        self.gru2 = GRU(gru1_output_size, gru2_output_size, 3)
        self.gru3 = GRU(gru2_output_size, gru3_output_size, 3)

        self.get_depth = torch.nn.Sequential(nn.Conv2d(16, 1, 3, 1, 1, bias=False),
                                             nn.Sigmoid())
        self.conv1 = torch.nn.Sequential(nn.Conv2d(32, out_channels=16, kernel_size=1, stride=1, padding=0),
                                         nn.ELU(inplace=True))
        self.conv2 = torch.nn.Sequential(nn.Conv2d(32, out_channels=8, kernel_size=1, stride=1, padding=0),
                                         nn.ELU(inplace=True))
        self.conv3 = torch.nn.Sequential(nn.Conv2d(32, out_channels=4, kernel_size=1, stride=1, padding=0),
                                         nn.ELU(inplace=True))

    def forward(self, x):
        _, _, h, w = x.shape

        with torch.no_grad():
            feature_resnet = self.resnet(x)
            feature_densenet = self.densenet(x)
            feature_resnext = self.resnext(x)
            feature_transformer = self.transformer(x)
            feature_conformer = self.conformer(x)
            feature_volo = self.volo(x)

        features_to_fusion = [ feature_densenet, feature_conformer,feature_volo,feature_transformer]#精度由低到高

        features_fused = []
        fused_1 = self.conv1(feature_resnet)
        fused_2 = self.conv2(feature_resnet)
        fused_3 = self.conv3(feature_resnet)
        for feature in features_to_fusion:
            fused_1 = self.gru1(feature, fused_1)
            fused_2 = self.gru2(fused_1, fused_2)
            fused_3 = self.gru3(fused_2, fused_3)

            features_fused.append(fused_3)

        features_fused = torch.cat(features_fused, 1)
        final_depth = self.params.max_depth * self.get_depth(features_fused)

        return final_depth
