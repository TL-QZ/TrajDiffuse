import torch
import torch.nn as nn

def build_pos_emb(pos_emb_dim):

    pos_emb_layers = []
    for i in range(1, len(pos_emb_dim)):
        pos_emb_layers.append(
            nn.Linear(pos_emb_dim[i - 1], pos_emb_dim[i])
        )
        pos_emb_layers.append(nn.Mish())
    
    return nn.Sequential(*pos_emb_layers)


def build_init_conv(channel_list, kernel_sizes):

    conv_layers = []
    for i in range(1, len(channel_list)):
        conv_layers.append(
            nn.Conv2d(channel_list[i - 1], channel_list[i], kernel_sizes[i - 1], stride=1, padding=kernel_sizes[i - 1] // 2)
        )
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
    conv_layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
    
    return nn.Sequential(*conv_layers)

def build_classifier(channel_list, kernel_sizes, linear_size):

        conv_layers = []
        for i in range(1, len(channel_list)):
            conv_layers.append(
                nn.Conv2d(channel_list[i - 1], channel_list[i], kernel_sizes[i - 1], stride=1, padding=kernel_sizes[i - 1] // 2)
            )
            conv_layers.append(nn.ReLU())
        conv_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        conv_layers.append(nn.Flatten())
        for i in range(1, len(linear_size)):
            conv_layers.append(
                nn.Linear(linear_size[i - 1], linear_size[i])
            )
        return nn.Sequential(*conv_layers)