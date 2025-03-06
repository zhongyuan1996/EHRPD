import argparse
import os
import time
import csv
import ast
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from models.dataset import pancreas_Gendataset, gen_collate_fn
from torch.utils.data import Dataset, DataLoader
from models.toy import *
from utils.utils import check_path, export_config, bool_flag
from utils.icd_rel import *
from models.evaluators import Evaluator
import warnings
import random
from collections import Counter
torch.autograd.set_detect_anomaly(True)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


def main(seed, name, data, max_len, max_num, save_dir, mode, focal_alpha, focal_gamma, short_ICD, subset, num_prompt):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=seed, type=int, help='seed')
    parser.add_argument('-bs', '--batch_size', default=24, type=int)
    parser.add_argument('-me', '--max_epochs_before_stop', default=10, type=int)
    parser.add_argument('--d_model', default=256, type=int, help='dimension of hidden layers')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate of hidden layers')
    parser.add_argument('--dropout_emb', default=0.1, type=float, help='dropout rate of embedding layers')
    parser.add_argument('--num_layers', default=1, type=int, help='number of transformer layers of EHR encoder')
    parser.add_argument('--num_heads', default=4, type=int, help='number of attention heads')
    parser.add_argument('--max_len', default=max_len, type=int, help='max visits of EHR')
    parser.add_argument('--max_num_codes', default=max_num, type=int, help='max number of ICD codes in each visit')
    parser.add_argument('--max_num_blks', default=100, type=int, help='max number of blocks in each visit')
    parser.add_argument('--blk_emb_path', default='./data/processed/block_embedding.npy',
                        help='embedding path of blocks')
    parser.add_argument('--target_disease', default=data, choices=['Heartfailure', 'COPD', 'Kidney', 'Dementia', 'Amnesia', 'mimic'])
    parser.add_argument('--target_att_heads', default=4, type=int, help='target disease attention heads number')
    parser.add_argument('--mem_size', default=15, type=int, help='memory size')
    parser.add_argument('--mem_update_size', default=15, type=int, help='memory update size')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--target_rate', default=0.3, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--warmup_steps', default=200, type=int)
    parser.add_argument('--n_epochs', default=30, type=int)
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--mode', default=mode)
    parser.add_argument('--model', default=name)
    parser.add_argument('--save_dir', default=save_dir)
    parser.add_argument('--lambda_timegap', default=0.000011, type=float)
    parser.add_argument('--lambda_diff', default=0.5, type=float)
    parser.add_argument('--lambda_ce', default=10000, type=float)
    parser.add_argument('--focal_alpha', default=focal_alpha, type=float)
    parser.add_argument('--focal_gamma', default=focal_gamma, type=float)
    parser.add_argument('--short_ICD', default=short_ICD, type=bool_flag, nargs='?', const=True, help='use short ICD codes')
    parser.add_argument('--toy', default=subset, type=bool_flag, nargs='?', const=True, help='use toy dataset')
    parser.add_argument('--subtask', default='')
    parser.add_argument('--num_prompt', default=num_prompt, type=int)

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'gen':
        gen(args)
    else:
        raise ValueError('Invalid mode')

def Attribute_Disclosure(model, real_seq, synthetic_seq, modality, seq_length, code_disclosure_percentage, pad_id, k=5):
    batchsize, seqlen, codelen = real_seq.shape
    # Select 1% of patients for attribute disclosure
    num_disclosed_patients = max(int(batchsize * 0.01), 1)
    disclosed_patients_idx = np.random.choice(batchsize, num_disclosed_patients, replace=False)

    total_success = 0
    total_masked_features = 0

    if modality == 'diag':
        synthetic_emb = model.diag_embedding(synthetic_seq).sum(dim=-2)
    elif modality == 'drug':
        synthetic_emb = model.drug_embedding(synthetic_seq).sum(dim=-2)
    elif modality == 'lab':
        synthetic_emb = model.lab_embedding(synthetic_seq).sum(dim=-2)
    elif modality == 'proc':
        synthetic_emb = model.proc_embedding(synthetic_seq).sum(dim=-2)
    flat_synthetic_emb = synthetic_emb.view(-1, synthetic_emb.shape[-1])

    for idx in disclosed_patients_idx:
        # Aggregate all viable codes from the compromised patient
        all_non_padding_codes = []
        for visit_idx in range(seq_length[idx].item()):
            visit_codes = real_seq[idx, visit_idx, :]
            non_padding_mask = visit_codes != pad_id
            all_non_padding_codes.extend(visit_codes[non_padding_mask].cpu().numpy())

        # Select a percentage of these codes as known to the attacker
        num_known_codes = max(1,int(len(all_non_padding_codes) * code_disclosure_percentage))
        known_codes = np.random.choice(all_non_padding_codes, num_known_codes, replace=False)

        known_codes_mask = torch.full(real_seq[idx, :, :].shape, True)  # Initialize mask to True (known)
        known_visit_indices = []
        all_masked_codes_idx = []
        # Mask the codes unknown to the attacker in each visit
        for visit_idx in range(seq_length[idx].item()):
            visit_codes = real_seq[idx, visit_idx, :]
            non_padding_mask = visit_codes != pad_id
            visit_unknown_codes_mask = np.isin(visit_codes.cpu().numpy(), known_codes,
                                               invert=True) & non_padding_mask.cpu().numpy()

            # Update the known codes mask for the current visit
            known_codes_mask[visit_idx] = torch.from_numpy(~visit_unknown_codes_mask)

            # Store the indices of masked codes for this visit
            masked_codes_idx = np.where(visit_unknown_codes_mask)[0]
            all_masked_codes_idx.append(masked_codes_idx)

            if np.any(~visit_unknown_codes_mask):
                known_visit_indices.append(visit_idx)

        # Apply the inverse of mask to get only known codes
        known_codes = torch.where(known_codes_mask, real_seq[idx].cpu(), torch.tensor(pad_id))
        # Get the embedding of the real known visits
        if modality == 'diag':
            device = next(model.parameters()).device
            known_codes = known_codes.to(device)
            real_emb = model.diag_embedding(known_codes).sum(dim=-2)
        elif modality == 'drug':
            device = next(model.parameters()).device
            known_codes = known_codes.to(device)
            real_emb = model.drug_embedding(known_codes).sum(dim=-2)
        elif modality == 'lab':
            device = next(model.parameters()).device
            known_codes = known_codes.to(device)
            real_emb = model.lab_embedding(known_codes).sum(dim=-2)
        elif modality == 'proc':
            device = next(model.parameters()).device
            known_codes = known_codes.to(device)
            real_emb = model.proc_embedding(known_codes).sum(dim=-2)

        for visit_idx in known_visit_indices:
            masked_codes_idx = all_masked_codes_idx[visit_idx]
            # Calculate similarity for the known visit
            flat_real_emb = real_emb[visit_idx].view(-1, real_emb.shape[-1])
            similarity = F.cosine_similarity(flat_real_emb, flat_synthetic_emb, dim=-1)
            top_k_indices = torch.topk(similarity, k, largest=True).indices

            reconstructed_visit = infer_and_reconstruct_visit(top_k_indices, synthetic_seq, masked_codes_idx,
                                                              known_codes[visit_idx])

            # Calculate sensitivity for this visit
            original_visit = real_seq[idx, visit_idx, :]
            sensitivity = calculate_visit_sensitivity(reconstructed_visit, original_visit, masked_codes_idx)
            total_success += sensitivity
            total_masked_features += 1

    mean_sensitivity = total_success / total_masked_features if total_masked_features > 0 else 0
    return mean_sensitivity

def infer_and_reconstruct_visit(top_k_indices, synthetic_seq, masked_codes_idx, known_codes):
    # Get those visits from the synthetic sequence by the top k indices
    selected_visits = synthetic_seq.view(-1, synthetic_seq.shape[-1])[top_k_indices]
    # Convert selected_visits to python list and count the number of each code
    selected_visits_list = selected_visits.view(-1).cpu().numpy().tolist()
    code_count = Counter(selected_visits_list)

    for idx in masked_codes_idx:
        if code_count:
            # Get the most common code
            most_common_code, _ = code_count.most_common(1)[0]
            known_codes[idx] = most_common_code
            # Remove this code from the counter
            del code_count[most_common_code]
        else:
            # If code_count is exhausted, append a placeholder value
            known_codes[idx] = -1

    return known_codes

def handle_no_available_code():
    # Define how to handle the situation where all potential codes are already used.
    return -1  # Example: returning a placeholder code like -1


def calculate_visit_sensitivity(reconstructed_visit, original_visit, masked_codes_idx):
    # Initialize counts
    correct_inferences = 0
    total_masked_features = len(masked_codes_idx)

    # Compare only the masked (inferred) positions
    for idx in masked_codes_idx:
        if reconstructed_visit[idx] == original_visit[idx]:
            correct_inferences += 1

    # Calculate sensitivity
    sensitivity = correct_inferences / total_masked_features if total_masked_features > 0 else 0
    return sensitivity


def Presence_Disclosure(model, real_seq, synthetic_seq, modality, seq_length, percentage = 0.1):
    batchsize, seqlen, codelen = real_seq.shape
    #randomly choose 5% of the patients to be disclosed
    num_disclosed = max(int(batchsize * percentage),1)
    disclosed_idx = np.random.choice(batchsize, num_disclosed, replace=False)
    #first use the embedding layer to get the embedding of the real and synthetic codes
    if modality == 'diag':
        real_emb = model.diag_embedding(real_seq).sum(dim=-2)
        synthetic_emb = model.diag_embedding(synthetic_seq).sum(dim=-2)
    elif modality == 'drug':
        real_emb = model.drug_embedding(real_seq).sum(dim=-2)
        synthetic_emb = model.drug_embedding(synthetic_seq).sum(dim=-2)
    elif modality == 'lab':
        real_emb = model.lab_embedding(real_seq).sum(dim=-2)
        synthetic_emb = model.lab_embedding(synthetic_seq).sum(dim=-2)
    elif modality == 'proc':
        real_emb = model.proc_embedding(real_seq).sum(dim=-2)
        synthetic_emb = model.proc_embedding(synthetic_seq).sum(dim=-2)

    flat_real_emb = real_emb.view(-1, real_emb.shape[-1])
    flat_synthetic_emb = synthetic_emb.view(-1, synthetic_emb.shape[-1])
    similarity = F.cosine_similarity(flat_real_emb.unsqueeze(1), flat_synthetic_emb.unsqueeze(0), dim=-1)

    success_count = 0
    total_count = 0

    for idx in disclosed_idx:
        valid_length = seq_length[idx].item()  # Get the non-padding length for the current sequence
        for visit_idx in range(valid_length):  # Only iterate over non-padded visits
            flat_idx = idx * seqlen + visit_idx
            matched_idx = torch.argmax(similarity[flat_idx]).item()
            if matched_idx // seqlen == idx:
                success_count += 1
            total_count += 1  # Increment total_count within the valid_length loop

    sensitivity = success_count / total_count if total_count > 0 else 0
    return sensitivity



def train(args):
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    files = os.listdir(str(args.save_dir))
    csv_filename = str(args.model) + '_' + str(args.target_disease) + '_' + str(args.seed) + '_' + str(args.focal_alpha) + '_' + str(args.focal_gamma) + '.csv'
    if csv_filename in files:
        print("conducted_experiments")
    else:
        config_path = os.path.join(args.save_dir, 'config.json')
        model_path = os.path.join(args.save_dir, str(args.model) + '_' + str(args.target_disease) + '_' + str(args.seed) + '_' + str(args.focal_alpha) + '_' + str(args.focal_gamma) + '.pt')
        log_path = os.path.join(args.save_dir, 'log.csv')
        # export_config(args, config_path)
        check_path(model_path)
        # with open(log_path, 'w') as fout:
        #     fout.write('step,train_auc,dev_auc,test_auc\n')

        if args.target_disease == 'pancreas':
            code2id = pd.read_csv('./data/pancreas/code_to_int_mapping.csv', header=None)
            demo_len = 15
            pad_id = len(code2id)
            data_path = './data/pancreas/'
        elif args.target_disease == 'mimic' and args.short_ICD:
            diag2id = pd.read_csv('data/mimic/diagnosis_to_int_mapping_3dig.csv', header=None)
            drug2id = pd.read_csv('data/mimic/drug_to_int_mapping_3dig.csv', header=None)
            lab2id = pd.read_csv('data/mimic/lab_to_int_mapping_3dig.csv', header=None)
            proc2id = pd.read_csv('data/mimic/proc_to_int_mapping_3dig.csv', header=None)
            demo_len = 76
            diag_pad_id = len(diag2id)
            drug_pad_id = len(drug2id)
            lab_pad_id = len(lab2id)
            proc_pad_id = len(proc2id)
            drug_nan_id = 146
            lab_nan_id = 206
            proc_nan_id = 24
            pad_id_list = [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id]
            nan_id_list = [diag_pad_id, drug_nan_id, lab_nan_id, proc_nan_id]
            data_path = './data/mimic/'
        elif args.target_disease == 'mimic' and not args.short_ICD:
            diag2id = pd.read_csv('./data/mimic/diagnosis_to_int_mapping_5dig.csv', header=None)
            drug2id = pd.read_csv('./data/mimic/drug_to_int_mapping_5dig.csv', header=None)
            lab2id = pd.read_csv('./data/mimic/lab_to_int_mapping_5dig.csv', header=None)
            proc2id = pd.read_csv('./data/mimic/proc_to_int_mapping_5dig.csv', header=None)
            demo_len = 76
            diag_pad_id = len(diag2id)
            drug_pad_id = len(drug2id)
            lab_pad_id = len(lab2id)
            proc_pad_id = len(proc2id)
            drug_nan_id = 234
            lab_nan_id = 206
            proc_nan_id = 28
            pad_id_list = [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id]
            nan_id_list = [diag_pad_id, drug_nan_id, lab_nan_id, proc_nan_id]
            data_path = './data/mimic/'
        elif args.target_disease == 'breast':
            diag2id = pd.read_csv('./data/breast/ae_to_int_mapping.csv', header=None)
            drug2id = pd.read_csv('./data/breast/drug_to_int_mapping.csv', header=None)
            lab2id = pd.read_csv('./data/breast/lab_to_int_mapping.csv', header=None)
            proc2id = pd.read_csv('./data/breast/proc_to_int_mapping.csv', header=None)
            id2diag = dict(zip(diag2id[1], diag2id[0]))
            id2drug = dict(zip(drug2id[1], drug2id[0]))
            id2lab = dict(zip(lab2id[1], lab2id[0]))
            id2proc = dict(zip(proc2id[1], proc2id[0]))
            demo_len = 10
            diag_pad_id = len(diag2id)
            drug_pad_id = len(drug2id)
            lab_pad_id = len(lab2id)
            proc_pad_id = len(proc2id)
            diag_nan_id = 51
            drug_nan_id = 101
            lab_nan_id = 10
            proc_nan_id = 4
            pad_id_list = [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id]
            nan_id_list = [diag_nan_id, drug_nan_id, lab_nan_id, proc_nan_id]
            data_path = './data/breast/'
        elif args.target_disease == 'mimic4':
            diag2id = pd.read_csv('./data/mimic4/diagnosis_to_int_mapping_mimic4.csv', header=None)
            drug2id = pd.read_csv('./data/mimic4/drug_to_int_mapping_mimic4.csv', header=None)
            lab2id = pd.read_csv('./data/mimic4/lab_to_int_mapping_mimic4.csv', header=None)
            proc2id = pd.read_csv('./data/mimic4/proc_to_int_mapping_mimic4.csv', header=None)
            demo_len = 2
            diag_pad_id = len(diag2id)
            drug_pad_id = len(drug2id)
            lab_pad_id = len(lab2id)
            proc_pad_id = len(proc2id)
            diag_nan_id = 855
            drug_nan_id = 38
            lab_nan_id = 147
            proc_nan_id = 2
            pad_id_list = [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id]
            nan_id_list = [diag_nan_id, drug_nan_id, lab_nan_id, proc_nan_id]
            data_path = './data/mimic4/'
        elif args.target_disease == 'eicu':
            diag2id = pd.read_csv('./data/eicu/diagnosis_to_int_mapping_3dig.csv', header=None)
            drug2id = pd.read_csv('./data/eicu/drug_to_int_mapping_3dig.csv', header=None)
            lab2id = pd.read_csv('./data/eicu/lab_to_int_mapping_3dig.csv', header=None)
            proc2id = pd.read_csv('./data/eicu/proc_to_int_mapping_3dig.csv', header=None)
            demo_len = 2
            diag_pad_id = len(diag2id)
            drug_pad_id = len(drug2id)
            lab_pad_id = len(lab2id)
            proc_pad_id = len(proc2id)
            diag_nan_id = 0
            drug_nan_id = 148
            lab_nan_id = 158
            proc_nan_id = 1207
            pad_id_list = [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id]
            nan_id_list = [diag_nan_id, drug_nan_id, lab_nan_id, proc_nan_id]
            data_path = './data/eicu/'
        else:
            raise ValueError('Invalid disease')
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

        if args.model in ('MedDiffGa'):

            if args.short_ICD:

                train_dataset = pancreas_Gendataset(data_path + 'train_3dig' + str(args.target_disease) + '.csv',
                                          args.max_len, args.max_num_codes, pad_id_list, nan_id_list, device)
                dev_dataset = pancreas_Gendataset(data_path + 'val_3dig' + str(args.target_disease) + '.csv',
                                        args.max_len,args.max_num_codes, pad_id_list, nan_id_list, device)
                test_dataset = pancreas_Gendataset(data_path + 'test_3dig' + str(args.target_disease) + '.csv',  args.max_len,
                                         args.max_num_codes, pad_id_list, nan_id_list, device)

            else:
                train_dataset = pancreas_Gendataset(data_path + 'train_5dig' + str(args.target_disease) + '.csv',
                                          args.max_len, args.max_num_codes, pad_id_list, nan_id_list, device)
                dev_dataset = pancreas_Gendataset(data_path + 'val_5dig' + str(args.target_disease) + '.csv',
                                        args.max_len,args.max_num_codes, pad_id_list, nan_id_list, device)
                test_dataset = pancreas_Gendataset(data_path + 'test_5dig' + str(args.target_disease) + '.csv',  args.max_len,
                                         args.max_num_codes, pad_id_list, nan_id_list, device)
            train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=gen_collate_fn)
            dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=gen_collate_fn)
            test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=gen_collate_fn)
        if args.model == 'MedDiffGa':
                model = MedDiffGa(pad_id_list, args.d_model, args.dropout, args.dropout_emb, args.num_layers, demo_len, device,num_prompts=args.num_prompt,channel_list=[args.d_model,int(args.d_model*2),int(args.d_model*4)])
        else:
                raise ValueError('Invalid model')
        model.to(device)

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.learning_rate},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.learning_rate}
        ]
        optim = Adam(grouped_parameters)
        CE = nn.BCEWithLogitsLoss(reduction='mean')
        MSE = nn.MSELoss(reduction='mean')

        print('parameters:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print('\t{:45}\ttrainable\t{}'.format(name, param.size()))
            else:
                print('\t{:45}\tfixed\t{}'.format(name, param.size()))
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('\ttotal:', num_params)
        print()
        print('-' * 71)
        global_step, best_dev_epoch, total_loss = 0, 0, 0.0
        CE_loss, Time_loss, Diff_loss = 0.0, 0.0, 0.0
        best_diag_lpl, best_drug_lpl, best_lab_lpl, best_proc_lpl = 1e10, 1e10, 1e10, 1e10
        best_diag_mpl, best_drug_mpl, best_lab_mpl, best_proc_mpl = 1e10, 1e10, 1e10, 1e10
        best_choosing_statistic = 1e10
        eva = Evaluator(model, device)
        focal = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, reduction='mean')
        model.train()
        for epoch_id in range(args.n_epochs):
            print('epoch: {:5} '.format(epoch_id))
            model.train()
            start_time = time.time()

            train_dataloader = tqdm(train_dataloader)
            for i, data in enumerate(train_dataloader):
                optim.zero_grad()

                if args.model in ('MedDiffGa'):

                    diag_seq, drug_seq, lab_seq, proc_seq, time_step, visit_timegaps, diag_timegaps, drug_timegaps, lab_timegaps, proc_timegaps,\
                        diag_mask, drug_mask, lab_mask, proc_mask, diag_length, drug_length, lab_length, proc_length, demo, _ = data
                    diag_logits, drug_logits, lab_logits, proc_logits, Delta_ts, added_z, learned_z = model(diag_seq, drug_seq, lab_seq, proc_seq, time_step, visit_timegaps, diag_timegaps, drug_timegaps, lab_timegaps, proc_timegaps,\
                        diag_mask, drug_mask, lab_mask, proc_mask, diag_length, drug_length, lab_length, proc_length, demo)
                else:
                    raise ValueError('Invalid model')
                multihot_diag = torch.zeros_like(diag_logits, dtype=torch.float32)
                multihot_drug = torch.zeros_like(drug_logits, dtype=torch.float32)
                multihot_lab = torch.zeros_like(lab_logits, dtype=torch.float32)
                multihot_proc = torch.zeros_like(proc_logits, dtype=torch.float32)


                valid_diag_mask = diag_mask
                valid_drug_mask = drug_mask
                valid_lab_mask = lab_mask
                valid_proc_mask = proc_mask

                valid_diag_batch_indices, valid_diag_seq_indices, valid_diag_code_indices = torch.where(valid_diag_mask)
                valid_drug_batch_indices, valid_drug_seq_indices, valid_drug_code_indices = torch.where(valid_drug_mask)
                valid_lab_batch_indices, valid_lab_seq_indices, valid_lab_code_indices = torch.where(valid_lab_mask)
                valid_proc_batch_indices, valid_proc_seq_indices, valid_proc_code_indices = torch.where(valid_proc_mask)

                multihot_diag[valid_diag_batch_indices, valid_diag_seq_indices, valid_diag_code_indices] = 1.0
                multihot_drug[valid_drug_batch_indices, valid_drug_seq_indices, valid_drug_code_indices] = 1.0
                multihot_lab[valid_lab_batch_indices, valid_lab_seq_indices, valid_lab_code_indices] = 1.0
                multihot_proc[valid_proc_batch_indices, valid_proc_seq_indices, valid_proc_code_indices] = 1.0

                sequence_range = torch.arange(args.max_len, device=diag_seq.device).expand(diag_seq.size(0), args.max_len)
                expanded_lengths = diag_length.unsqueeze(-1).expand_as(sequence_range)
                length_mask = (sequence_range < expanded_lengths).float()

                a = (focal(diag_logits, multihot_diag) + focal(drug_logits, multihot_drug) + focal(lab_logits, multihot_lab) + focal(proc_logits, multihot_proc)) * args.lambda_ce
                b = (MSE(Delta_ts * length_mask, visit_timegaps * length_mask) + 1e-10) * args.lambda_timegap
                c = MSE(added_z, learned_z) * args.lambda_diff
                loss = a + b + c
                loss.backward()
                total_loss += (loss.item() / visit_timegaps.size(0)) * args.batch_size
                CE_loss += (a.item() / visit_timegaps.size(0)) * args.batch_size
                Time_loss += (b.item() / visit_timegaps.size(0)) * args.batch_size
                Diff_loss += (c.item() / visit_timegaps.size(0)) * args.batch_size

                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()
                if (global_step + 1) % args.log_interval == 0:
                    total_loss /= args.log_interval
                    CE_loss /= args.log_interval
                    Time_loss /= args.log_interval
                    Diff_loss /= args.log_interval
                    ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                    print('| step {:5} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step,
                                                                                   total_loss,
                                                                                   ms_per_batch))
                    print('| CE loss {:7.4f} | Time loss {:7.4f} | Diff loss {:7.4f} |'.format(CE_loss, Time_loss, Diff_loss))
                    total_loss, CE_loss, Time_loss, Diff_loss = 0.0, 0.0, 0.0, 0.0
                    start_time = time.time()
                global_step += 1
            drug_nan_id = 146
            lab_nan_id = 206
            proc_nan_id = 24
            model.eval()
            if args.model in ('MedDiffGa'):
                val_res = eva.eval(dev_dataloader, [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id], [diag_pad_id, drug_nan_id, lab_nan_id, proc_nan_id], ['diag', 'drug', 'lab', 'proc'], ['lpl', 'mpl'])
                test_res = eva.eval(test_dataloader, [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id], [diag_pad_id, drug_nan_id, lab_nan_id, proc_nan_id], ['diag', 'drug', 'lab', 'proc'], ['lpl', 'mpl'])
                val_diag_lpl, val_drug_lpl, val_lab_lpl, val_proc_lpl = val_res['lpl_diag'], val_res['lpl_drug'], val_res['lpl_lab'], val_res['lpl_proc']
                test_diag_lpl, test_drug_lpl, test_lab_lpl, test_proc_lpl = test_res['lpl_diag'], test_res['lpl_drug'], test_res['lpl_lab'], test_res['lpl_proc']

                val_diag_mpl, val_drug_mpl, val_lab_mpl, val_proc_mpl = val_res['mpl_diag'], val_res['mpl_drug'], val_res['mpl_lab'], val_res['mpl_proc']
                test_diag_mpl, test_drug_mpl, test_lab_mpl, test_proc_mpl = test_res['mpl_diag'], test_res['mpl_drug'], test_res['mpl_lab'], test_res['mpl_proc']

                choosing_statistic = np.median([val_diag_lpl, val_drug_lpl, val_lab_lpl, val_proc_lpl, val_diag_mpl, val_drug_mpl, val_lab_mpl, val_proc_mpl])

            else:
                raise ValueError('Invalid model')
            print('-' * 71)
            # print('train lpl {:7.4f} | dev lpl {:7.4f} | test lpl {:7.4f}'.format(train_lpl, val_lpl, test_lpl))
            train_diag_lpl, tran_drug_lpl, train_lab_lpl, train_proc_lpl = 0, 0, 0, 0
            train_diag_mpl, tran_drug_mpl, train_lab_mpl, train_proc_mpl = 0, 0, 0, 0
            print('Epoch: {:5}'.format(epoch_id))
            print('Diagnosis:')
            print('train lpl {:7.4f} | dev lpl {:7.4f} | test lpl {:7.4f}'.format(train_diag_lpl, val_diag_lpl, test_diag_lpl))
            print('train mpl {:7.4f} | dev mpl {:7.4f} | test mpl {:7.4f}'.format(train_diag_mpl, val_diag_mpl, test_diag_mpl))
            print('Drug:')
            print('train lpl {:7.4f} | dev lpl {:7.4f} | test lpl {:7.4f}'.format(tran_drug_lpl, val_drug_lpl, test_drug_lpl))
            print('train mpl {:7.4f} | dev mpl {:7.4f} | test mpl {:7.4f}'.format(tran_drug_mpl, val_drug_mpl, test_drug_mpl))
            print('Lab:')
            print('train lpl {:7.4f} | dev lpl {:7.4f} | test lpl {:7.4f}'.format(train_lab_lpl, val_lab_lpl, test_lab_lpl))
            print('train mpl {:7.4f} | dev mpl {:7.4f} | test mpl {:7.4f}'.format(train_lab_mpl, val_lab_mpl, test_lab_mpl))
            print('Procedure:')
            print('train lpl {:7.4f} | dev lpl {:7.4f} | test lpl {:7.4f}'.format(train_proc_lpl, val_proc_lpl, test_proc_lpl))
            print('train mpl {:7.4f} | dev mpl {:7.4f} | test mpl {:7.4f}'.format(train_proc_mpl, val_proc_mpl, test_proc_mpl))
            print('-' * 71)

            if choosing_statistic <= best_choosing_statistic:
                best_diag_lpl, best_drug_lpl, best_lab_lpl, best_proc_lpl = test_diag_lpl, test_drug_lpl, test_lab_lpl, test_proc_lpl
                best_diag_mpl, best_drug_mpl, best_lab_mpl, best_proc_mpl = test_diag_mpl, test_drug_mpl, test_lab_mpl, test_proc_mpl
                best_dev_epoch = epoch_id
                best_choosing_statistic = choosing_statistic
                torch.save([model, args], model_path)
                # torch.save([model, args], model_path)
                # with open(log_path, 'a') as fout:
                #     fout.write('{},{},{},{}\n'.format(global_step, tr_pr_auc, d_pr_auc, t_pr_auc))
                print(f'model saved to {model_path}')
            if epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
                break

        print()
        print('training ends at {} epoch'.format(epoch_id))
        print('Final statistics:')
        print('lpl: diag {:7.4f} | drug {:7.4f} | lab {:7.4f} | proc {:7.4f}'.format(best_diag_lpl, best_drug_lpl, best_lab_lpl, best_proc_lpl))
        print('mpl: diag {:7.4f} | drug {:7.4f} | lab {:7.4f} | proc {:7.4f}'.format(best_diag_mpl, best_drug_mpl, best_lab_mpl, best_proc_mpl))
        with open(args.save_dir + csv_filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([best_diag_lpl, best_drug_lpl, best_lab_lpl, best_proc_lpl, best_diag_mpl, best_drug_mpl, best_lab_mpl, best_proc_mpl])
        print()


if __name__ == '__main__':
    import sys
    import os

    # Modes and configurations
    modes = ['train']
    short_ICD = True
    subset = False
    seeds = [10]
    

    focal_alphas = [0.5]
    focal_gammas = [5]
    
    # Model names and directories
    save_path = './saved_EHRPD2025_'
    names = ['MedDiffGa']
    save_dirs = [save_path + name + '/' for name in names]
    
    # Data configurations
    datas = ['mimic']

    gpu_index = 1
    

    # Set the CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)

    max_len = 20
    max_num = 10


    for mode in modes:
        for seed in seeds:
            for focal_alpha in focal_alphas:
                for focal_gamma in focal_gammas:
                    for name, dir in zip(names, save_dirs):
                        for data in datas:
                            for p in num_prompts:
                                main(
                                    seed=seed,
                                    name=name,
                                    data=data,
                                    max_len=max_len,
                                    max_num=max_num,
                                    save_dir=dir+ 'max_len_'+str(max_len)+'_max_num_'+str(max_num)+'/'+'focal_alpha_'+str(focal_alpha)+'_focal_gamma_'+str(focal_gamma)+'/',
                                    mode=mode,
                                    focal_alpha=focal_alpha,
                                    focal_gamma=focal_gamma,
                                    short_ICD=short_ICD,
                                    subset=subset,
                                    num_prompt=p
                                )

