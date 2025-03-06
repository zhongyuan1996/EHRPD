import warnings
import torch
from tqdm import tqdm
import numpy as np
import math
class Evaluator:
    def __init__(self, model, device=None):
        self.model = model
        self.device = 'cpu' if device is None else device

    def lpl(self, logits, targets, lengths):
        ppl_list = []

        batch_size, num_visits, _ = logits.shape

        for i in range(batch_size):
            for visit in range(lengths[i]):
                # Extract logits and targets for the current visit of the current batch item
                logits_visit = logits[i, visit, :]
                target_visit = targets[i, visit, :]  # Extracting target codes for the specific visit

                prob = logits_visit.softmax(dim=-1)[target_visit == 1]
                nll = -torch.log(prob + 1e-10)
                ppl = nll.exp()

                if torch.isnan(ppl).any():
                    warnings.warn('NaN perplexity detected during lpl calculation')
                ppl_list.append(ppl)

        if ppl_list:
            median_ppl = torch.median(torch.cat(ppl_list))
            return median_ppl.item()
        else:
            print('No valid perplexity values found')
            return 0.0

    def mpl(self, logits_dict, targets_dict, lengths_dict):
        nll_values = {modality: [] for modality in logits_dict.keys()}
        mpl_values = {}

        for modality, logits in logits_dict.items():
            targets = targets_dict[modality]
            lengths = lengths_dict[modality]

            # Process each batch
            for i in range(logits.size(0)):
                for visit in range(lengths[i]):
                    # Extract logits and targets for the current visit of the current batch item
                    logits_visit = logits[i, visit, :]
                    target_visit = targets[i, visit, :]

                    # Calculate probabilities
                    prob = logits_visit.softmax(dim=-1)[target_visit == 1]

                    # Calculate NLL
                    nll = -torch.log(prob + 1e-10)
                    nll_values[modality].extend(nll.tolist())

        # Calculate mpl for each modality
        for modality, nll_list in nll_values.items():
            if nll_list:
                avg_nll = sum(nll_list) / len(nll_list)
                mpl_values[modality] = math.exp(avg_nll)
            else:
                mpl_values[modality] = float('inf')  # Or another placeholder value indicating no valid NLL was found

        return mpl_values

    def eval(self, dataloader, pad_id_list, nan_id_list, modalities, metric_list):

        if 'lpl' in metric_list:
            lpl_results = {modality: [] for modality in modalities}
        if 'mpl' in metric_list:
            mpl_results = {modality: [] for modality in modalities}

        for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating"):
            diag_seq, drug_seq, lab_seq, proc_seq, time_step, visit_timegaps, diag_timegaps, drug_timegaps, lab_timegaps, proc_timegaps, \
                diag_mask, drug_mask, lab_mask, proc_mask, diag_length, drug_length, lab_length, proc_length, demo, _ = data

            with torch.no_grad():
                if self.model.name == 'MedDiffGa':
                    diag_logits, drug_logits, lab_logits, proc_logits, Delta_ts, added_z, learned_z = self.model(diag_seq, drug_seq, lab_seq, proc_seq,time_step,visit_timegaps,diag_timegaps,drug_timegaps,lab_timegaps,proc_timegaps,
                    diag_mask,drug_mask,lab_mask,proc_mask,diag_length,drug_length,lab_length,proc_length,demo)
                elif self.model.name == 'GAN' or self.model.name == 'WGAN-GP':
                    _, _, _, _, diag_logits, drug_logits, lab_logits, proc_logits, _, _ = self.model(
                        diag_seq, drug_seq, lab_seq, proc_seq, diag_length)
                elif self.model.name == 'MLP':
                    diag_logits, drug_logits, lab_logits, proc_logits, _, _, _, _, _, _ = self.model(
                        diag_seq, drug_seq, lab_seq, proc_seq, diag_length)
                elif self.model.name == 'VAE':
                    diag_logits, drug_logits, lab_logits, proc_logits, _, _, _ = self.model(
                        diag_seq, drug_seq, lab_seq, proc_seq, diag_length, time_step)
                elif self.model.name == 'Diff':
                    _, _, _, _, diag_logits, drug_logits, lab_logits, proc_logits, _, _ ,_ ,_ = self.model(
                        diag_seq, drug_seq, lab_seq, proc_seq, diag_length)
                else:
                    print('Invalid model name')

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

                if 'lpl' in metric_list:
                    for modality in modalities:
                        if modality == 'diag':
                            lpl_results[modality].append(self.lpl(diag_logits, multihot_diag, diag_length))
                        elif modality == 'drug':
                            lpl_results[modality].append(self.lpl(drug_logits, multihot_drug, drug_length))
                        elif modality == 'lab':
                            lpl_results[modality].append(self.lpl(lab_logits, multihot_lab, lab_length))
                        elif modality == 'proc':
                            lpl_results[modality].append(self.lpl(proc_logits, multihot_proc, proc_length))
                        else:
                            raise ValueError('Invalid modality')

                if 'mpl' in metric_list:
                    logits_dict = {'diag': diag_logits, 'drug': drug_logits, 'lab': lab_logits, 'proc': proc_logits}
                    targets_dict = {'diag': multihot_diag, 'drug': multihot_drug, 'lab': multihot_lab, 'proc': multihot_proc}
                    lengths_dict = {'diag': diag_length, 'drug': drug_length, 'lab': lab_length, 'proc': proc_length}
                    temp_res = self.mpl(logits_dict, targets_dict, lengths_dict)
                    for modality in modalities:
                        mpl_results[modality].append(temp_res[modality])

        res = {}
        if 'lpl' in metric_list:
            for modality in modalities:
                res[f'lpl_{modality}'] = np.median(lpl_results[modality]) if lpl_results[modality] else warnings.warn(
                    f'No valid perplexity values found for {modality}')
        if 'mpl' in metric_list:
            for modality in modalities:
                res[f'mpl_{modality}'] = np.median(mpl_results[modality]) if mpl_results[modality] else warnings.warn(
                    f'No valid perplexity values found for {modality}')

        return res