import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.baseline import Attention, PositionalEncoding
from models.unet_skipM import unetSkip
from utils.diffUtil import get_beta_schedule

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


class PositionalEncoder(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model

    def forward(self, code_timegaps):
        # Expand time gaps to [batch_size, seq_len, num_codes, d_model]
        code_timegaps = code_timegaps.unsqueeze(-1).expand(-1, -1, -1, self.d_model)

        div_term = torch.exp(torch.arange(0, self.d_model).float() * (-math.log(10000.0) / self.d_model))

        # Making sure div_term can be broadcasted over code_timegaps
        div_term = div_term.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        pos = code_timegaps * div_term.to(code_timegaps.device)

        # Use torch.where to avoid in-place operations
        even_indices = torch.arange(0, self.d_model, 2, device=code_timegaps.device)
        odd_indices = torch.arange(1, self.d_model, 2, device=code_timegaps.device)

        sin = torch.sin(pos[:, :, :, even_indices])
        cos = torch.cos(pos[:, :, :, odd_indices])

        # Intermix sin and cos values
        pos = torch.stack((sin, cos), dim=4).flatten(start_dim=3)

        return pos

class VisitLevelPositionalEncoder(nn.Module):
    def __init__(self, d_model):
        super(VisitLevelPositionalEncoder, self).__init__()
        self.d_model = d_model

    def forward(self, code_timegaps):
        # Expand time gaps to [batch_size, seq_len, d_model]
        code_timegaps = code_timegaps.unsqueeze(-1).expand(-1, -1, self.d_model)

        div_term = torch.exp(torch.arange(0, self.d_model).float() * (-math.log(10000.0) / self.d_model))

        # Making sure div_term can be broadcasted over code_timegaps
        div_term = div_term.unsqueeze(0).unsqueeze(0)
        pos = code_timegaps * div_term.to(code_timegaps.device)

        # Use torch.where to avoid in-place operations
        even_indices = torch.arange(0, self.d_model, 2, device=code_timegaps.device)
        odd_indices = torch.arange(1, self.d_model, 2, device=code_timegaps.device)

        sin = torch.sin(pos[:, :, even_indices])
        cos = torch.cos(pos[:, :, odd_indices])

        # Intermix sin and cos values
        pos = torch.stack((sin, cos), dim=3).flatten(start_dim=2)

        return pos

class LSTM_with_time(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.pooler = MaxPoolLayer()
        self.rnns = nn.LSTM(d_model, d_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)
        self.positional_encoder = PositionalEncoder(d_model)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, d_model)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.ddto2 = nn.Linear(2 * d_model, 2)
        self.dtod1 = nn.Linear(d_model, d_model)
        self.dtod2 = nn.Linear(d_model, d_model)
        self.ddtod = nn.Linear(2 * d_model, d_model)

    def modeling_time_vs_code(self, input_seqs, masks, lengths, seq_time_step, code_masks, code_timegaps, visit_timegaps):
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        c = self.embedding(input_seqs)

        code_gap_emb = self.positional_encoder(code_timegaps)
        delta_t = self.fc2(self.relu(self.fc1(code_gap_emb)))

        concat_embeddings = torch.cat([c, delta_t], dim=-1)
        p = F.gumbel_softmax(torch.log_softmax(self.ddto2(concat_embeddings), -1), hard=True)

        decision_to_modify = p[..., 0].unsqueeze(-1)

        attn =self.softmax(torch.stack((self.dtod1(delta_t), self.dtod2(c)),dim = -1))

        attn_c = attn[..., 0]
        attn_delta_t = attn[..., 1]

        eta = c * (1-decision_to_modify) + decision_to_modify * (c*attn_c + delta_t*attn_delta_t)

        return eta

    def forward(self, input_seqs, masks, lengths, seq_time_step, code_masks, code_timegaps, visit_timegaps):
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.modeling_time_vs_code(input_seqs, masks, lengths, seq_time_step, code_masks, code_timegaps, visit_timegaps).sum(dim=-2)
        x = self.emb_dropout(x)
        rnn_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnns(rnn_input)
        x, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=seq_len)
        x = self.pooler(x, lengths)
        x = self.output_mlp(x)
        return x


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

        return Delta_t, lambda_curr

class MedDiffGa(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, demo_len, device, num_heads = 4, num_prompts=5, channel_list = [256,512,1024], num_resnet_blocks = 2):
        super().__init__()
        self.name = 'MedDiffGa'
        self.device = device
        self.demo_len = demo_len
        self.d_model = d_model
        self.demoMLP = nn.Sequential(nn.Linear(self.demo_len, 64), nn.ReLU(), nn.Linear(64, d_model))
        self.diag_embedding = nn.Embedding(vocab_size[0]+1, d_model, padding_idx=-1)
        self.drug_embedding = nn.Embedding(vocab_size[1]+1, d_model, padding_idx=-1)
        self.lab_embedding = nn.Embedding(vocab_size[2]+1, d_model, padding_idx=-1)
        self.proc_embedding = nn.Embedding(vocab_size[3]+1, d_model, padding_idx=-1)

        self.diag_ff = nn.Linear(d_model, d_model)
        self.drug_ff = nn.Linear(d_model, d_model)
        self.lab_ff = nn.Linear(d_model, d_model)
        self.proc_ff = nn.Linear(d_model, d_model)
        self.modaltiy_att = nn.Linear(4 * d_model, 4)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)

        self.diag_output_mlp = nn.Sequential(nn.Linear(d_model, vocab_size[0]))
        self.drug_output_mlp = nn.Sequential(nn.Linear(d_model, vocab_size[1]))
        self.lab_output_mlp = nn.Sequential(nn.Linear(d_model, vocab_size[2]))
        self.proc_output_mlp = nn.Sequential(nn.Linear(d_model, vocab_size[3]))
        self.pooler = MaxPoolLayer()
        self.rnns = nn.LSTM(d_model, d_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)
        self.positional_encoder = PositionalEncoder(d_model)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, d_model)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.ddto2 = nn.Linear(2 * d_model, 2)
        self.dtod1 = nn.Linear(d_model, d_model)
        self.dtod2 = nn.Linear(d_model, d_model)
        self.ddtod = nn.Linear(2 * d_model, d_model)
        self.timegap_predictor = timegap_predictor(d_model)
        self.unet = unetSkip(channel_list, num_resnet_blocks, num_heads, dropout)
        betas = get_beta_schedule(beta_schedule="linear",
                                  beta_start=0.0001,
                                  beta_end=0.02,
                                  num_diffusion_timesteps=1000)
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.diffusion_num_timesteps = betas.shape[0]

        
        self.prompt_embeddings = nn.Parameter(torch.randn(num_prompts, d_model))
        self.num_prompts = num_prompts

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.modality_transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def encoder_time_vs_code(self, input_seqs, code_timegaps, embedding_layer):
        if embedding_layer == 'diag':
            c = self.diag_embedding(input_seqs)
        elif embedding_layer == 'drug':
            c = self.drug_embedding(input_seqs)
        elif embedding_layer == 'lab':
            c = self.lab_embedding(input_seqs)
        elif embedding_layer == 'proc':
            c = self.proc_embedding(input_seqs)
        else:
            raise ValueError('Wrong embedding layer')
        code_gap_emb = self.positional_encoder(code_timegaps)
        delta_t = self.fc2(self.relu(self.fc1(code_gap_emb)))
        concat_embeddings = torch.cat([c, delta_t], dim=-1)
        p = F.gumbel_softmax(torch.log_softmax(self.ddto2(concat_embeddings), -1), hard=True)
        decision_to_modify = p[..., 0].unsqueeze(-1)
        attn = self.softmax(torch.stack((self.dtod1(delta_t), self.dtod2(c)), dim=-1))
        attn_c = attn[..., 0]
        attn_delta_t = attn[..., 1]
        eta = c * (1 - decision_to_modify) + decision_to_modify * (c * attn_c + delta_t * attn_delta_t)
        return self.emb_dropout(eta)

    def forward(self, diag_seq, drug_seq, lab_seq, proc_seq, time_step, visit_timegap, diag_timegaps, drug_timegaps, lab_timegaps, proc_timegaps,\
                        diag_mask, drug_mask, lab_mask, proc_mask, diag_length, drug_length, lab_length, proc_length, demo):

        batch_size, seq_len, num_cui_per_visit = diag_seq.size()
        B, S = batch_size, seq_len
        diag_eta = self.encoder_time_vs_code(diag_seq, diag_timegaps, 'diag')
        drug_eta = self.encoder_time_vs_code(drug_seq, drug_timegaps, 'drug')
        lab_eta = self.encoder_time_vs_code(lab_seq, lab_timegaps, 'lab')
        proc_eta = self.encoder_time_vs_code(proc_seq, proc_timegaps, 'proc')


        diag_s = diag_eta.sum(dim=-2)
        drug_s = drug_eta.sum(dim=-2)
        lab_s = lab_eta.sum(dim=-2)
        proc_s = proc_eta.sum(dim=-2)

        diag_z = self.relu(self.diag_ff(diag_s))
        drug_z = self.relu(self.drug_ff(drug_s))
        lab_z = self.relu(self.lab_ff(lab_s))
        proc_z = self.relu(self.proc_ff(proc_s))

        modality_tensor = torch.stack((diag_z, drug_z, lab_z, proc_z), dim=2)  # (B, S, 4, D)


        modality_tensor = modality_tensor.view(batch_size * seq_len, 4, self.d_model)  # (B*S, 4, D)

        prompt_embeddings_expanded = self.prompt_embeddings.expand(B * S, -1, -1)

        transformer_input = torch.cat((prompt_embeddings_expanded, modality_tensor), dim=1)

        transformer_output = self.modality_transformer(transformer_input)

        individual_z = transformer_output[:, self.num_prompts:, :]  # (B*S, 4, D)
        individual_z = transformer_output[:, self.num_prompts:, :].max(dim=1).values.view(batch_size, seq_len, self.d_model)  # (B, S, D)


        shared_z = transformer_output[:, :self.num_prompts, :]  # (B*S, num_prompts, D)
        shared_z = shared_z.view(batch_size * seq_len, self.num_prompts, self.d_model)  # (B, S, num_prompts, D)



        rnn_input = pack_padded_sequence(individual_z, diag_length.cpu(), batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnns(rnn_input)
        h_packed, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=seq_len)
        h = h_packed  # (B, S, D)


        #ablation study 2:
        Delta_ts, _ = self.timegap_predictor(h)
        Delta_ts = Delta_ts.squeeze(-1)


        eta = torch.cat((diag_eta, drug_eta, lab_eta, proc_eta), dim=-2)
        eta = eta.view(batch_size * seq_len, num_cui_per_visit * 4, -1)

        h = h.view(batch_size * seq_len, -1)
        individual_z = individual_z.view(batch_size * seq_len, -1)
        De = self.demoMLP(demo)
        De = De.unsqueeze(1).expand(-1, seq_len, -1).reshape(batch_size * seq_len, -1)

        Delta_ts_emb = self.positional_encoder(Delta_ts.unsqueeze(-1))
        Delta_ts_emb = self.fc2(self.relu(self.fc1(Delta_ts_emb))).squeeze(-2).view(batch_size * seq_len, -1)


        H = torch.stack((h,De, Delta_ts_emb), dim=-2)
        H = torch.concat((H, shared_z), dim=-2)



        diag_mask = diag_mask.view(batch_size * seq_len, num_cui_per_visit).unsqueeze(-1)
        drug_mask = drug_mask.view(batch_size * seq_len, num_cui_per_visit).unsqueeze(-1)
        lab_mask = lab_mask.view(batch_size * seq_len, num_cui_per_visit).unsqueeze(-1)
        proc_mask = proc_mask.view(batch_size * seq_len, num_cui_per_visit).unsqueeze(-1)
        code_masks = torch.cat((diag_mask, drug_mask, lab_mask, proc_mask), dim=-2)

        diffusion_time_t = torch.randint(
            low=0, high=1000, size=[eta.shape[0], ]).to(
            self.device)
        alpha = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t).view(-1, 1, 1)
        z = torch.randn_like(eta)
        added_z = z * (1.0 - alpha).sqrt()
        eta_hat = eta * alpha.sqrt() + added_z
        eta_hat = eta_hat * code_masks


        eta_next = self.unet(eta_hat.transpose(1, 2), H.transpose(1, 2), None).transpose(1, 2).view(batch_size, seq_len, num_cui_per_visit * 4, -1)


        eta_next_shifted = torch.cat(
            [torch.zeros(batch_size, 1, num_cui_per_visit * 4, eta_next.size(-1), device=eta_next.device),
             eta_next[:, :-1]], dim=1)

        # Calculate learned_z based on the shifted eta_next
        learned_z = eta_next_shifted - eta_hat.view(batch_size, seq_len, num_cui_per_visit * 4, -1)
        added_z = added_z.view(batch_size, seq_len, num_cui_per_visit * 4, -1)

        # v_next = eta_next.sum(dim=-2)
        diag_s_next = eta_next[:, :, :num_cui_per_visit, :].sum(dim=-2)
        drug_s_next = eta_next[:, :, num_cui_per_visit:2*num_cui_per_visit, :].sum(dim=-2)
        lab_s_next = eta_next[:, :, 2*num_cui_per_visit:3*num_cui_per_visit, :].sum(dim=-2)
        proc_s_next = eta_next[:, :, 3*num_cui_per_visit:, :].sum(dim=-2)

        diag_logits = self.diag_output_mlp(diag_s_next)
        drug_logits = self.drug_output_mlp(drug_s_next)
        lab_logits = self.lab_output_mlp(lab_s_next)
        proc_logits = self.proc_output_mlp(proc_s_next)


        return diag_logits, drug_logits, lab_logits, proc_logits, Delta_ts, added_z, learned_z



