import torch
import torch.nn as nn
import torch.nn.functional as F
from models.baseline import Attention, MaxPoolLayer

class timegap_predictor(nn.Module):
    def __init__(self, d_model):
        super(timegap_predictor, self).__init__()
        self.W_lambda = nn.Linear(d_model, d_model)
        self.W_delta_t = nn.Linear(d_model, 1)
        self.tanh = nn.Tanh()
        self.sofplus = nn.Softplus()
        self.dropout = nn.Dropout(0.1)

    def forward(self, h_curr):
        lambda_curr = self.dropout(1-self.tanh(self.W_lambda(h_curr)))
        Delta_t = self.sofplus(self.W_delta_t(lambda_curr))

        return Delta_t

class ResNetBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super(ResNetBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x, copy = None):

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if copy is not None:
            out += copy
        else:
            out += x
        out = self.relu(out)
        return out

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride = 1):
        super(DownsampleBlock, self).__init__()
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        out = self.downsample(x)
        return out

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride = 1):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        out = self.upsample(x)
        return out


class skipConnectionBlock(nn.Module):
    def __init__(self,d_model, num_heads = 1, dropout = 0.1):
        super(skipConnectionBlock, self).__init__()
        self.attention = Attention(d_model, num_heads, dropout)

    def forward(self, H, eta, attn_mask = None):
        eta_prime, _ = self.attention(eta.transpose(1, 2),  H.transpose(1, 2),  H.transpose(1, 2), attn_mask = attn_mask)
        return eta_prime.transpose(1, 2)

class H_MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(H_MLP, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2)
        return x

class unetSkip(nn.Module):
    def __init__(self, channelList, num_resBlocks, num_heads = 4, dropout = 0.1):
        super(unetSkip, self).__init__()

        reversed_channelList = channelList[::-1]
        self.resblocks_down = nn.ModuleList()
        self.downsampling_blocks = nn.ModuleList()
        self.h_mlps = nn.ModuleList()
        self.skip_blocks = nn.ModuleList()

        for i in range(len(channelList)-1):
            res_blocks = nn.Sequential(*[ResNetBlock1d(channelList[i], channelList[i]) for _ in range(num_resBlocks)])
            downsample = DownsampleBlock(channelList[i], channelList[i+1])
            self.resblocks_down.append(res_blocks)
            self.downsampling_blocks.append(downsample)
            self.h_mlps.append(H_MLP(channelList[i], channelList[i + 1]))
            skip_block = skipConnectionBlock(channelList[i], num_heads, dropout)
            self.skip_blocks.append(skip_block)

        self.concat_and_conv_blocks = nn.ModuleList()
        self.resblocks_up = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()
        for i in range(len(reversed_channelList)-1):
            concat_and_conv = ResNetBlock1d(reversed_channelList[i+1] * 2, reversed_channelList[i+1])
            self.concat_and_conv_blocks.append(concat_and_conv)
            res_blocks = nn.Sequential(*[ResNetBlock1d(reversed_channelList[i+1], reversed_channelList[i+1]) for _ in range(num_resBlocks-1)])
            upsample = UpsampleBlock(reversed_channelList[i], reversed_channelList[i+1])
            self.resblocks_up.append(res_blocks)
            self.upsampling_blocks.append(upsample)

        self.bottleneck = nn.Sequential(*[ResNetBlock1d(channelList[-1], channelList[-1]) for _ in range(num_resBlocks)])
        self.bottleneck_skip = skipConnectionBlock(channelList[-1], num_heads, dropout)
    def forward(self, eta, H, attn_mask = None):

        # downsampled_eta = []
        skip_connections = []
        H_dense = [H]

        for i in range(len(self.resblocks_down)):
            eta_res = self.resblocks_down[i](eta)
            skip = self.skip_blocks[i](H_dense[-1], eta_res, attn_mask)
            skip_connections.append(skip)

            eta = self.downsampling_blocks[i](eta_res)

            H_next = self.h_mlps[i](H_dense[-1])
            H_dense.append(H_next)

        eta = self.bottleneck(eta)
        eta = self.bottleneck_skip(H_dense[-1], eta, attn_mask)

        for j in range(len(self.resblocks_up)):
            eta_copy = self.upsampling_blocks[j](eta)
            eta = torch.cat([eta_copy, skip_connections.pop()], dim = 1)
            eta = self.concat_and_conv_blocks[j](eta, eta_copy)
            eta = self.resblocks_up[j](eta)

        return eta
