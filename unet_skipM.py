import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxPoolLayer(nn.Module):
    """
    A layer that performs max pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths=None):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if mask_or_lengths is not None:
            if len(mask_or_lengths.size()) == 1:
                mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(
                    1))
            else:
                mask = mask_or_lengths
            inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), float('-inf'))
        max_pooled = inputs.max(1)[0]
        return max_pooled

class Attention(nn.Module):
    def __init__(self, in_feature, num_head, dropout):
        super(Attention, self).__init__()
        self.in_feature = in_feature
        self.num_head = num_head
        self.size_per_head = in_feature // num_head
        self.out_dim = num_head * self.size_per_head
        assert self.size_per_head * num_head == in_feature
        self.q_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.k_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.v_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.fc = nn.Linear(in_feature, in_feature, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_feature)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = key.size(0)
        res = query
        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)

        query = query.view(batch_size, self.num_head, -1, self.size_per_head)
        key = key.view(batch_size, self.num_head, -1, self.size_per_head)
        value = value.view(batch_size, self.num_head, -1, self.size_per_head)

        scale = np.sqrt(self.size_per_head)
        energy = torch.matmul(query, key.permute(0, 1, 3, 2)) / scale
        if attn_mask is not None:
            batch_size, q_len, k_len = attn_mask.size()
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, self.num_head, q_len, k_len)
            energy = energy.masked_fill(attn_mask == 0, -np.inf)

        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(attention, value)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.in_feature)
        attention = attention.sum(dim=1).squeeze().permute(0, 2, 1) / self.num_head
        x = self.fc(x)
        x = self.dropout(x)
        x += res
        x = self.layer_norm(x)
        return x, attention

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
