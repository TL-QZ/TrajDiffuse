import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb
from torch.autograd import Variable

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
)
from diffusion_models.layer import *


class ValueFunction(nn.Module):
    def __init__(self, pos_emb_dim, init_conv_channels, init_kernel_size, classifier_channels, classifier_kernel_size, classifier_linear_size, **kwargs) -> None:
        super().__init__()

        # coordinate embedding
        # bs x obs_len x 2 --> bs x obs_len x emb_dim
        self.pos_embed_layer = build_pos_emb(pos_emb_dim)

        # Initial CNN layer
        self.init_conv_layer = build_init_conv(init_conv_channels, init_kernel_size)
        
        # FiLM Residual block x 2
        self.film_res_block1 = FiLM_Res_block(init_conv_channels)
        self.film_res_block2 = FiLM_Res_block(init_conv_channels, False)

        # Classifier layer
        self.classifier_layer = build_classifier(classifier_channels, classifier_kernel_size, classifier_linear_size)

    def forward(self, coord, batched_map):
        

        # process map
        map_feature = self.init_conv_layer(batched_map)


        # process coordinate (add two coordinate embedding channels)
        FiLM_feat = self.pos_embed_layer(coord)
        FiLM_feat = FiLM_feat.unsqueeze(-1).unsqueeze(-1)
        gamma_dim = int(FiLM_feat.shape[1]/2)

        # FiLM residual block

        gamma, beta = FiLM_feat[:, 0:gamma_dim], FiLM_feat[:, gamma_dim:]
        FiLM_feat = self.film_res_block1(map_feature, gamma, beta)
        FiLM_feat = self.film_res_block2(FiLM_feat, gamma, beta)

        # Classifier layer
        out = self.classifier_layer(FiLM_feat)

        return out

class FiLM_Res_block(nn.Module):
    def __init__(self, init_conv_channels, add_coord_channel=True) -> None:
        super().__init__()
        self.add_coord_channel = add_coord_channel
        channel_num = init_conv_channels[-1] + 2
        self.conv1 = nn.Conv2d(channel_num, channel_num, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(channel_num)
        self.relu = nn.ReLU()
        self.film = FiLM()

    def forward(self, x, gammas, betas):
        if self.add_coord_channel:
            coord_channels = torch.stack([coord_map(x.shape[2:4])]*x.shape[0]).to(x.device)
            x = torch.cat([x, coord_channels], dim=1)
        x = self.conv1(x)
        out = self.relu(x)
        x = self.conv2(out)
        x = self.batch_norm(x)
        x = self.film(x, gammas, betas)
        x = self.relu(x)
        out = out + x
        return out



class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """
    def forward(self, x, gammas, betas):
        # gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        # betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas * x) + betas


def coord_map(shape, start=-1, end=1):
    """
    Gives, a 2d shape tuple, returns two mxn coordinate maps,
    Ranging min-max in the x and y directions, respectively.
    """
    m, n = shape
    x_coord_row = torch.linspace(start, end, steps=n)
    y_coord_row = torch.linspace(start, end, steps=m)
    x_coords = x_coord_row.unsqueeze(0).expand(torch.Size((m, n))).unsqueeze(0)
    y_coords = y_coord_row.unsqueeze(1).expand(torch.Size((m, n))).unsqueeze(0)
    return Variable(torch.cat([x_coords, y_coords], 0))