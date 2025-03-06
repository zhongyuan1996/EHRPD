import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import ast
import pandas as pd


def padMatrix(input_data, max_num_pervisit, maxlen, pad_id):
    pad_seq = [pad_id] * max_num_pervisit
    output = []
    lengths = []
    for seq in input_data:
        record_ids = []
        for visit in seq:
            visit_ids = visit[0: max_num_pervisit]
            for i in range(0, (max_num_pervisit - len(visit_ids))):
                visit_ids.append(pad_id)
            record_ids.append(visit_ids)
        record_ids = record_ids[-maxlen:]
        lengths.append(len(record_ids))
        for j in range(0, (maxlen - len(record_ids))):
            record_ids.append(pad_seq)
        output.append(record_ids)
    masks = []
    for l in lengths:
        mask = np.tril(np.ones((maxlen, maxlen)))
        # mask[:l, :l] = np.tril(np.ones((l, l)))
        masks.append(mask)
    return output, masks, lengths


def padMatrix2(input_data, max_num_pervisit, maxlen, pad_id):
    pad_seq = [pad_id] * max_num_pervisit
    output = []
    lengths = []
    for seq in input_data:
        record_ids = []
        for visit in seq:
            visit_ids = [visit] if visit != pad_id else []  # Changed this line
            for i in range(0, (max_num_pervisit - len(visit_ids))):
                visit_ids.append(pad_id)
            record_ids.append(visit_ids)
        record_ids = record_ids[-maxlen:]
        lengths.append(len(record_ids))
        for j in range(0, (maxlen - len(record_ids))):
            record_ids.append(pad_seq)
        output.append(record_ids)
    masks = []
    for l in lengths:
        mask = np.tril(np.ones((maxlen, maxlen)))
        # mask[:l, :l] = np.tril(np.ones((l, l)))
        masks.append(mask)
    return output, masks, lengths


def padTime(time_step, maxlen, pad_id):
    for k in range(len(time_step)):
        time_step[k] = time_step[k][-maxlen:]
        while len(time_step[k]) < maxlen:
            time_step[k].append(pad_id)
    return time_step
def padTime3(time_step, maxlen, pad_id):

    return time_step
def codeMask(input_data, max_num_pervisit, maxlen):
    batch_mask = np.zeros((len(input_data), maxlen, max_num_pervisit), dtype=np.float32) + 1
    output = []
    for seq in input_data:
        record_ids = []
        for visit in seq:
            visit_ids = visit[0: max_num_pervisit]
            record_ids.append(visit_ids)
        record_ids = record_ids[-maxlen:]
        output.append(record_ids)

    for bid, seq in enumerate(output):
        for pid, subseq in enumerate(seq):
            for tid, code in enumerate(subseq):
                batch_mask[bid, pid, tid] = 0
    return batch_mask

def codeMask_inverse(input_data, max_num_pervisit, maxlen, nan_pad_id):
    batch_mask = np.zeros((len(input_data), maxlen, max_num_pervisit), dtype=np.float32)
    output = []
    for seq in input_data:
        record_ids = []
        for visit in seq:
            visit_ids = visit[0: max_num_pervisit]
            record_ids.append(visit_ids)
        record_ids = record_ids[-maxlen:]
        output.append(record_ids)

    for bid, seq in enumerate(output):
        for pid, subseq in enumerate(seq):
            for tid, code in enumerate(subseq):
                if code != nan_pad_id:
                    batch_mask[bid, pid, tid] = 1
    return batch_mask

class MyDataset(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, ehr_pad_id,
                 device):
        ehr, labels, time_step = pickle.load(open(dir_ehr, 'rb'))
        self.labels = [[0,1] if label == 1 else [1,0] for label in labels]

        self.ehr, _, self.lengths = padMatrix(ehr, max_numcode_pervisit, max_len, ehr_pad_id)
        self.time_step = padTime(time_step, max_len, 100000)
        self.code_mask = codeMask(ehr, max_numcode_pervisit, max_len)
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return torch.tensor(self.ehr[idx], dtype=torch.long).to(self.device),\
               torch.tensor(self.time_step[idx], dtype=torch.long).to(self.device),\
               torch.tensor(self.labels[idx], dtype=torch.float).to(self.device),\
                  torch.tensor(self.code_mask[idx], dtype=torch.float).to(self.device),\
                  torch.tensor(self.lengths[idx], dtype=torch.long).to(self.device)


class MyDataset_mapping(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, ehr_pad_id,
                 device):
        ehr, labels, time_step = pickle.load(open(dir_ehr, 'rb'))

        reshaped_ehr = []
        reshaped_time_step = []
        expanded_labels = []
        for b, batch in enumerate(ehr):
            for v, visit in enumerate(batch):
                for c, code in enumerate(visit):
                    reshaped_ehr.append([code])
                    reshaped_time_step.append([time_step[b][v]])
                    expanded_labels.append(labels[b])

        self.labels = [[0, 1] if label == 1 else [1, 0] for label in expanded_labels]
        self.ehr, _, _ = padMatrix2(reshaped_ehr, max_numcode_pervisit, max_len, ehr_pad_id)
        self.time_step = padTime(reshaped_time_step, max_len, 100000)
        self.code_mask = codeMask2(reshaped_ehr, max_numcode_pervisit, max_len)
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return torch.tensor(self.ehr[idx], dtype=torch.long).to(self.device), \
            torch.tensor(self.time_step[idx], dtype=torch.long).to(self.device), \
            torch.tensor(self.labels[idx], dtype=torch.float).to(self.device), \
            torch.tensor(self.code_mask[idx], dtype=torch.float).to(self.device)

class MyDataset2(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, max_numblk_pervisit, ehr_pad_id,
                 device):
        ehr, self.labels, time_step = pickle.load(open(dir_ehr, 'rb'))
        self.ehr, self.mask_ehr, self.lengths = padMatrix(ehr, max_numcode_pervisit, max_len, ehr_pad_id)
        # time_step = time_step_to_deltatime(time_step)
        self.time_step = padTime(time_step, max_len, 10000)
        self.code_mask = codeMask(ehr, max_numcode_pervisit, max_len)
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _ = None
        return torch.tensor(self.labels[idx], dtype=torch.long).to(self.device), torch.LongTensor(self.ehr[idx]).to(
            self.device), \
               torch.LongTensor(self.mask_ehr[idx]).to(self.device), _, \
               _, torch.tensor(self.lengths[idx], dtype=torch.long).to(self.device), \
               torch.Tensor(self.time_step[idx]).to(self.device), torch.FloatTensor(self.code_mask[idx]).to(self.device)

class MyDataset3(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, ehr_pad_id,
                 device):
        ehr, labels, time_step = pickle.load(open(dir_ehr, 'rb'))
        self.labels = [[0,1] if label == 1 else [1,0] for label in labels]

        self.ehr, _, self.lengths = padMatrix(ehr, max_numcode_pervisit, max_len, ehr_pad_id)
        self.time_step = padTime(time_step, max_len, 100000)
        self.code_mask = codeMask(ehr, max_numcode_pervisit, max_len)
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return torch.tensor(self.ehr[idx], dtype=torch.long).to(self.device),\
               torch.tensor(self.time_step[idx], dtype=torch.long).to(self.device),\
               torch.tensor(self.labels[idx], dtype=torch.float).to(self.device),\
                  torch.tensor(self.code_mask[idx], dtype=torch.float).to(self.device),\
                  torch.tensor(self.lengths[idx], dtype=torch.long).to(self.device)
class MyDataset4(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, ehr_pad_id, w2v, flag, device):
        self.labels = None
        if flag == True:
            data = pd.read_csv(dir_ehr)
            ehr = data['code_int'].apply(lambda x: ast.literal_eval(x)).tolist()
            time_step = data['time_gaps'].apply(lambda x: ast.literal_eval(x)).tolist()
        else:

            ehr, labels, time_step = pickle.load(open(dir_ehr, 'rb'))
            self.labels = [[0,1] if label == 1 else [1,0] for label in labels]

        self.ehr, _, _ = padMatrix(ehr, max_numcode_pervisit, max_len, ehr_pad_id)
        self.time_step = padTime(time_step, max_len, 100000)
        self.device = device

    def __len__(self):
        return len(self.ehr)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        if self.labels:

            return torch.tensor(self.ehr[idx], dtype=torch.long).to(self.device),\
               torch.tensor(self.time_step[idx], dtype=torch.long).to(self.device),\
               torch.tensor(self.labels[idx], dtype=torch.long).to(self.device)
        else:
            return torch.tensor(self.ehr[idx], dtype=torch.long).to(self.device),\
                torch.tensor(self.time_step[idx], dtype=torch.long).to(self.device)


class MyDataset_timegap(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, ehr_pad_id, device):
        ehr, labels, visit_timegaps, time_step, code_timegaps = pickle.load(open(dir_ehr, 'rb'))
        self.labels = [[0, 1] if label == 1 else [1, 0] for label in labels]

        self.ehr, _, self.lengths = padMatrix(ehr, max_numcode_pervisit, max_len, ehr_pad_id)

        self.time_step = padTime(time_step, max_len, 10000)
        self.visit_timegap = padTime(visit_timegaps, max_len, 10000)

        self.code_timegaps, _, _ = padMatrix(code_timegaps, max_numcode_pervisit, max_len, 100000)

        self.code_mask = codeMask(ehr, max_numcode_pervisit, max_len)

        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return torch.tensor(self.ehr[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.time_step[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.code_timegaps[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.labels[idx], dtype=torch.float).to(self.device),\
            torch.tensor(self.code_mask[idx], dtype=torch.float).to(self.device),\
            torch.tensor(self.lengths[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.visit_timegap[idx], dtype=torch.long).to(self.device)

def collate_fn_timegap(batch):
    ehr, time_step, code_timegaps, label, code_mask, length, visit_timegap = [], [], [], [], [], [], []
    for data in batch:
        ehr.append(data[0])
        time_step.append(data[1])
        code_timegaps.append(data[2])
        label.append(data[3])
        code_mask.append(data[4])
        length.append(data[5])
        visit_timegap.append(data[6])

    return torch.stack(ehr, 0), torch.stack(time_step, 0), torch.stack(code_timegaps, 0), torch.stack(label, 0), torch.stack(code_mask, 0), torch.stack(length, 0), torch.stack(visit_timegap, 0)



def collate_fn(batch):
    ehr, time_step, label, code_mask, length = [], [], [], [], []
    # label, ehr, mask, txt, mask_txt, length, time_step, code_mask = [], [], [], [], [], [], [], []
    for data in batch:
        ehr.append(data[0])
        time_step.append(data[1])
        label.append(data[2])
        code_mask.append(data[3])
        length.append(data[4])

    return torch.stack(ehr, 0), torch.stack(time_step, 0), torch.stack(label, 0), torch.stack(code_mask, 0), torch.stack(length, 0)


class w2vecDataset(Dataset):
    def __init__(self, dir_ehr,device):
        self.ehr, _, _ = pickle.load(open(dir_ehr, 'rb'))
        self.device = device
    def __len__(self):
        return len(self.ehr)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.tensor(self.ehr[idx], dtype=torch.long).to(self.device)

def collate_fn_w2vec(batch):
    ehr = []
    for data in batch:
        ehr.append(data)
    return torch.stack(ehr, 0)


class ehrGANDataset(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, ehr_pad_id, w2v, flag, device):
        self.labels = None
        if flag == True:
            data = pd.read_csv(dir_ehr)
            ehr = data['code_int'].apply(lambda x: ast.literal_eval(x)).tolist()
            time_step = data['time_gaps'].apply(lambda x: ast.literal_eval(x)).tolist()
        else:

            ehr, labels, time_step = pickle.load(open(dir_ehr, 'rb'))
            self.labels = [[0,1] if label == 1 else [1,0] for label in labels]

        # aggregate ehr and time_step into 90-day windows
        ehr, time_step = self.segment_time_windows(ehr, time_step)
        self.ehr, _, _ = padMatrix(ehr, max_numcode_pervisit, max_len, ehr_pad_id)
        self.time_step = padTime(time_step, max_len, 100000)
        self.device = device
        self.w2v = w2v
        self.pad_id = ehr_pad_id

    def ehr_to_embedding(self, ehr):
        embedding_sequence = []
        mask = []
        for visit in ehr:  # Loop over each visit in the EHR data
            visit_embedding = []
            visit_mask = []
            for code in visit:  # Loop over each code in the visit
                if code == self.pad_id:  # If the code is a padding token, return a zero vector.
                    visit_embedding.append(np.zeros(self.w2v.vector_size))
                    visit_mask.append(np.zeros(self.w2v.vector_size))  # Add a zero vector to the mask if it is a padding token.
                elif code in self.w2v.wv:  # If the code is in the Word2Vec model, add its vector.
                    visit_embedding.append(self.w2v.wv[code])
                    visit_mask.append(np.ones(self.w2v.vector_size))  # Add a one vector to the mask if it is a real code.
            embedding_sequence.append(visit_embedding)
            mask.append(visit_mask)
        return np.array(embedding_sequence), np.array(mask)

    def segment_time_windows(self, ehr, time_step):
        ehr_segmented, time_step_segmented = [], []
        for patient_idx in range(len(ehr)):
            patient_ehr = ehr[patient_idx]
            patient_time_step = time_step[patient_idx]
            patient_ehr_segmented, patient_time_step_segmented = [], []
            aggregated_codes, end_time = [], patient_time_step[0]
            for visit_idx in range(len(patient_ehr)):
                visit = patient_ehr[visit_idx]
                if end_time - patient_time_step[visit_idx] < 90:
                    aggregated_codes.extend(visit)
                else:
                    # Aggregate and append codes of the current window
                    patient_ehr_segmented.append(list(set(aggregated_codes)))
                    patient_time_step_segmented.append(end_time)
                    # Start a new window
                    aggregated_codes = visit
                    end_time = patient_time_step[visit_idx]
            # Don't forget to append the last window
            patient_ehr_segmented.append(list(set(aggregated_codes)))
            patient_time_step_segmented.append(end_time)

            # Append the segmented ehr and time_step of each patient
            ehr_segmented.append(patient_ehr_segmented)
            time_step_segmented.append(patient_time_step_segmented)

        return ehr_segmented, time_step_segmented

    def __len__(self):
        return len(self.ehr)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        temp_ehr, temp_code_mask = self.ehr_to_embedding(self.ehr[idx])

        ehr_embedding = torch.tensor(temp_ehr, dtype=torch.float).to(self.device)
        aggregated_code_ehr = torch.tensor(self.ehr[idx], dtype=torch.long).to(self.device)
        time_step = torch.tensor(self.time_step[idx], dtype=torch.long).to(self.device)
        code_mask = torch.tensor(temp_code_mask, dtype=torch.float).to(self.device)
        if self.labels:
            label = torch.tensor(self.labels[idx], dtype=torch.float).to(self.device)
            return ehr_embedding, time_step, label, code_mask, aggregated_code_ehr
        else:
            return ehr_embedding, time_step, code_mask, aggregated_code_ehr


class ehrGANDatasetWOAggregate(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, ehr_pad_id, w2v, device):
        ehr, labels, time_step = pickle.load(open(dir_ehr, 'rb'))
        self.labels = [[0,1] if label == 1 else [1,0] for label in labels]
        self.ehr, _, _ = padMatrix(ehr, max_numcode_pervisit, max_len, ehr_pad_id)
        self.time_step = padTime(time_step, max_len, 100000)
        self.device = device
        self.w2v = w2v
        self.pad_id = ehr_pad_id

    def ehr_to_embedding(self, ehr):
        embedding_sequence = []
        mask = []
        for visit in ehr:  # Loop over each visit in the EHR data
            visit_embedding = []
            visit_mask = []
            for code in visit:  # Loop over each code in the visit
                if code == self.pad_id:  # If the code is a padding token, return a zero vector.
                    visit_embedding.append(np.zeros(self.w2v.vector_size))
                    visit_mask.append(np.zeros(self.w2v.vector_size))  # Add a zero vector to the mask if it is a padding token.
                elif code in self.w2v.wv:  # If the code is in the Word2Vec model, add its vector.
                    visit_embedding.append(self.w2v.wv[code])
                    visit_mask.append(np.ones(self.w2v.vector_size))  # Add a one vector to the mask if it is a real code.
            embedding_sequence.append(visit_embedding)
            mask.append(visit_mask)
        return np.array(embedding_sequence), np.array(mask)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        temp_ehr, temp_code_mask = self.ehr_to_embedding(self.ehr[idx])

        ehr = torch.tensor(temp_ehr, dtype=torch.float).to(self.device)
        time_step = torch.tensor(self.time_step[idx], dtype=torch.long).to(self.device)
        label = torch.tensor(self.labels[idx], dtype=torch.float).to(self.device)
        code_mask = torch.tensor(temp_code_mask, dtype=torch.float).to(self.device)

        return ehr, time_step, label, code_mask


class pancreas_Gendataset(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, pad_id_list, nan_id_list, device, task = None):

        data = pd.read_csv(dir_ehr)
        if task == 'mortality':
            self.labels = data['Mortality_LABEL'].tolist()
        elif task == 'arf':
            self.labels = data['ARF_LABEL'].tolist()
        elif task == 'shock':
            self.labels = data['Shock_LABEL'].tolist()
        else:
            self.labels = data['MORTALITY'].tolist()
        # self.multihot_ehr = data['code_multihot'].apply(lambda x: ast.literal_eval(x)).tolist()
        self.demo = data['Demographic'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else print(x)).tolist()
        diag = data['DIAGNOSES_int'].apply(lambda x: ast.literal_eval(x)).tolist()
        drug = data['DRG_CODE_int'].apply(lambda x: ast.literal_eval(x)).tolist()
        lab = data['LAB_ITEM_int'].apply(lambda x: ast.literal_eval(x)).tolist()
        proc = data['PROC_ITEM_int'].apply(lambda x: ast.literal_eval(x)).tolist()

        # ehr = data['code_int'].apply(lambda x: ast.literal_eval(x)).tolist()
        time_steps = data['time_gaps'].apply(lambda x: ast.literal_eval(x)).tolist()
        visit_consecutive_timegaps = data['consecutive_time_gaps'].apply(lambda x: ast.literal_eval(x)).tolist()
        # code_timegaps = data['code_time_gaps'].apply(lambda x: ast.literal_eval(x)).tolist()
        diag_timegaps = data['code_time_gaps'].apply(lambda x: ast.literal_eval(x)).tolist()
        drug_timegaps = data['drug_time_gaps'].apply(lambda x: ast.literal_eval(x)).tolist()
        lab_timegaps = data['lab_time_gaps'].apply(lambda x: ast.literal_eval(x)).tolist()
        proc_timegaps = data['proc_time_gaps'].apply(lambda x: ast.literal_eval(x)).tolist()

        self.diag, _, self.diag_lens = padMatrix(diag, max_numcode_pervisit, max_len, pad_id_list[0])
        self.drug, _, self.drug_lens = padMatrix(drug, max_numcode_pervisit, max_len, pad_id_list[1])
        self.lab, _, self.lab_lens = padMatrix(lab, max_numcode_pervisit, max_len, pad_id_list[2])
        self.proc, _, self.proc_lens = padMatrix(proc, max_numcode_pervisit, max_len, pad_id_list[3])

        self.time_step = padTime(time_steps, max_len, 100000)
        self.visit_timegap = padTime(visit_consecutive_timegaps, max_len, 100000)


        self.diag_timegaps, _, _ = padMatrix(diag_timegaps, max_numcode_pervisit, max_len, 100000)
        self.drug_timegaps, _, _ = padMatrix(drug_timegaps, max_numcode_pervisit, max_len, 100000)
        self.lab_timegaps, _, _ = padMatrix(lab_timegaps, max_numcode_pervisit, max_len, 100000)
        self.proc_timegaps, _, _ = padMatrix(proc_timegaps, max_numcode_pervisit, max_len, 100000)

        # self.code_mask = codeMask_inverse(ehr, max_numcode_pervisit, max_len)
        self.diag_mask = codeMask_inverse(diag, max_numcode_pervisit, max_len, nan_id_list[0])
        self.drug_mask = codeMask_inverse(drug, max_numcode_pervisit, max_len, nan_id_list[1])
        self.lab_mask = codeMask_inverse(lab, max_numcode_pervisit, max_len, nan_id_list[2])
        self.proc_mask = codeMask_inverse(proc, max_numcode_pervisit, max_len, nan_id_list[3])

        self.device = device

    def __len__(self):
        return len(self.diag)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        assert self.diag_lens[idx] == self.drug_lens[idx] == self.lab_lens[idx] == self.proc_lens[idx]

        return torch.tensor(self.diag[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.drug[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.lab[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.proc[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.time_step[idx], dtype=torch.long).to(self.device), \
            torch.tensor(self.visit_timegap[idx], dtype=torch.float).to(self.device), \
            torch.tensor(self.diag_timegaps[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.drug_timegaps[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.lab_timegaps[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.proc_timegaps[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.diag_mask[idx], dtype=torch.float).to(self.device),\
            torch.tensor(self.drug_mask[idx], dtype=torch.float).to(self.device),\
            torch.tensor(self.lab_mask[idx], dtype=torch.float).to(self.device),\
            torch.tensor(self.proc_mask[idx], dtype=torch.float).to(self.device),\
            torch.tensor(self.diag_lens[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.drug_lens[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.lab_lens[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.proc_lens[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.demo[idx], dtype=torch.float).to(self.device),\
            torch.tensor(self.labels[idx], dtype=torch.long).to(self.device)

def gen_collate_fn(batch):
    diag, drug, lab, proc, time_step, visit_timegap, diag_timegaps, drug_timegaps, lab_timegaps, proc_timegaps, diag_mask, drug_mask, lab_mask, proc_mask, diag_lens, drug_lens, lab_lens, proc_lens, demo, label = \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for data in batch:
        diag.append(data[0])
        drug.append(data[1])
        lab.append(data[2])
        proc.append(data[3])
        time_step.append(data[4])
        visit_timegap.append(data[5])
        diag_timegaps.append(data[6])
        drug_timegaps.append(data[7])
        lab_timegaps.append(data[8])
        proc_timegaps.append(data[9])
        diag_mask.append(data[10])
        drug_mask.append(data[11])
        lab_mask.append(data[12])
        proc_mask.append(data[13])
        diag_lens.append(data[14])
        drug_lens.append(data[15])
        lab_lens.append(data[16])
        proc_lens.append(data[17])
        demo.append(data[18])
        label.append(data[19])

    return torch.stack(diag,0), \
        torch.stack(drug,0), \
        torch.stack(lab,0), \
        torch.stack(proc,0), \
        torch.stack(time_step,0), \
        torch.stack(visit_timegap,0), \
        torch.stack(diag_timegaps,0), \
        torch.stack(drug_timegaps,0), \
        torch.stack(lab_timegaps,0), \
        torch.stack(proc_timegaps,0), \
        torch.stack(diag_mask,0), \
        torch.stack(drug_mask,0), \
        torch.stack(lab_mask,0), \
        torch.stack(proc_mask,0), \
        torch.stack(diag_lens,0), \
        torch.stack(drug_lens,0), \
        torch.stack(lab_lens,0), \
        torch.stack(proc_lens,0),\
        torch.stack(demo,0),\
        torch.stack(label,0)

class pancreas_Gendataset_unimodal(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, pad_id, nan_id, modality, device):

        data = pd.read_csv(dir_ehr)
        self.label = data['MORTALITY'].tolist()
        if modality == 'diag':
            ehr_codes = data['DIAGNOSES_int'].apply(lambda x: ast.literal_eval(x)).tolist()
            ehr_codes_timegaps = data['code_time_gaps'].apply(lambda x: ast.literal_eval(x)).tolist()
        elif modality == 'drug':
            ehr_codes = data['DRG_CODE_int'].apply(lambda x: ast.literal_eval(x)).tolist()
            ehr_codes_timegaps = data['drug_time_gaps'].apply(lambda x: ast.literal_eval(x)).tolist()
        elif modality == 'lab':
            ehr_codes = data['LAB_ITEM_int'].apply(lambda x: ast.literal_eval(x)).tolist()
            ehr_codes_timegaps = data['lab_time_gaps'].apply(lambda x: ast.literal_eval(x)).tolist()
        elif modality == 'proc':
            ehr_codes = data['PROC_ITEM_int'].apply(lambda x: ast.literal_eval(x)).tolist()
            ehr_codes_timegaps = data['proc_time_gaps'].apply(lambda x: ast.literal_eval(x)).tolist()
        else:
            raise ValueError('modality not supported')

        time_steps = data['time_gaps'].apply(lambda x: ast.literal_eval(x)).tolist()
        visit_consecutive_timegaps = data['consecutive_time_gaps'].apply(lambda x: ast.literal_eval(x)).tolist()

        self.ehr_codes, self.mask_ehr, self.lengths = padMatrix(ehr_codes, max_numcode_pervisit, max_len, pad_id)

        self.time_step = padTime(time_steps, max_len, 100000)
        self.visit_timegap = padTime(visit_consecutive_timegaps, max_len, 100000)
        self.ehr_codes_timegaps, _, _ = padMatrix(ehr_codes_timegaps, max_numcode_pervisit, max_len, 100000)
        self.codemask = codeMask_inverse(ehr_codes, max_numcode_pervisit, max_len, nan_id)

        self.device = device

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return torch.tensor(self.label[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.ehr_codes[idx], dtype=torch.long).to(self.device), \
            torch.tensor(self.mask_ehr[idx], dtype=torch.float).to(self.device), \
            torch.tensor(self.time_step[idx], dtype=torch.long).to(self.device), \
            torch.tensor(self.visit_timegap[idx], dtype=torch.float).to(self.device), \
            torch.tensor(self.ehr_codes_timegaps[idx], dtype=torch.long).to(self.device), \
            torch.tensor(self.codemask[idx], dtype=torch.float).to(self.device), \
            torch.tensor(self.lengths[idx], dtype=torch.long).to(self.device)


def gen_collate_fn_unimodal(batch):
    label, ehr_codes, mask_ehr, time_step, visit_timegap, ehr_codes_timegaps, codemask, lengths = \
        [], [], [], [], [], [], [], []
    for data in batch:
        label.append(data[0])
        ehr_codes.append(data[1])
        mask_ehr.append(data[2])
        time_step.append(data[3])
        visit_timegap.append(data[4])
        ehr_codes_timegaps.append(data[5])
        codemask.append(data[6])
        lengths.append(data[7])

    return torch.stack(label,0), \
        torch.stack(ehr_codes,0), \
        torch.stack(mask_ehr,0), \
        torch.stack(time_step,0), \
        torch.stack(visit_timegap,0), \
        torch.stack(ehr_codes_timegaps,0), \
        torch.stack(codemask,0), \
        torch.stack(lengths,0)

class pancreas_Gendataset_unimodal_for_baselines(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, pad_id, nan_id, modality, device):

        data = pd.read_csv(dir_ehr)
        self.label = data['MORTALITY'].tolist()

        if modality == 'diag':
            ehr_codes = data['DIAGNOSES_int'].apply(lambda x: ast.literal_eval(x)).tolist()
        elif modality == 'drug':
            ehr_codes = data['DRG_CODE_int'].apply(lambda x: ast.literal_eval(x)).tolist()
        elif modality == 'lab':
            ehr_codes = data['LAB_ITEM_int'].apply(lambda x: ast.literal_eval(x)).tolist()
        elif modality == 'proc':
            ehr_codes = data['PROC_ITEM_int'].apply(lambda x: ast.literal_eval(x)).tolist()
        else:
            raise ValueError('modality not supported')

        time_steps = data['time_gaps'].apply(lambda x: ast.literal_eval(x)).tolist()

        self.ehr_codes, self.mask_ehr, self.lengths = padMatrix(ehr_codes, max_numcode_pervisit, max_len, pad_id)
        self.time_step = padTime(time_steps, max_len, 100000)
        self.codemask = codeMask_inverse(ehr_codes, max_numcode_pervisit, max_len, nan_id)

        self.device = device

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return torch.tensor(self.label[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.ehr_codes[idx], dtype=torch.long).to(self.device), \
            torch.tensor(self.mask_ehr[idx], dtype=torch.float).to(self.device), \
            torch.tensor(self.time_step[idx], dtype=torch.long).to(self.device), \
            torch.tensor(self.codemask[idx], dtype=torch.float).to(self.device), \
            torch.tensor(self.lengths[idx], dtype=torch.long).to(self.device)


def gen_collate_fn_unimodal_for_baselines(batch):
    label, ehr_codes, mask_ehr, time_step, codemask, lengths = [], [], [], [], [], []
    for data in batch:
        label.append(data[0])
        ehr_codes.append(data[1])
        mask_ehr.append(data[2])
        time_step.append(data[3])
        codemask.append(data[4])
        lengths.append(data[5])

    return torch.stack(label,0), \
        torch.stack(ehr_codes,0), \
        torch.stack(mask_ehr,0), \
        torch.stack(time_step,0), \
        torch.stack(codemask,0), \
        torch.stack(lengths,0)

class pancreas_Gendataset_multimodal(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, pad_id_list, nan_id_list, task, device):

        data = pd.read_csv(dir_ehr)
        if task == 'mortality':
            self.label = data['Mortality_LABEL'].tolist()
        elif task == 'arf':
            self.label = data['ARF_LABEL'].tolist()
        elif task == 'shock':
            self.label = data['Shock_LABEL'].tolist()
        else:
            self.label = data['MORTALITY'].tolist()

        self.demo = data['Demographic'].apply(lambda x: ast.literal_eval(x)).tolist()
        diag = data['DIAGNOSES_int'].apply(lambda x: ast.literal_eval(x)).tolist()
        drug = data['DRG_CODE_int'].apply(lambda x: ast.literal_eval(x)).tolist()
        lab = data['LAB_ITEM_int'].apply(lambda x: ast.literal_eval(x)).tolist()
        proc = data['PROC_ITEM_int'].apply(lambda x: ast.literal_eval(x)).tolist()

        self.diag, _, self.diag_lens = padMatrix(diag, max_numcode_pervisit, max_len, pad_id_list[0])
        self.drug, _, self.drug_lens = padMatrix(drug, max_numcode_pervisit, max_len, pad_id_list[1])
        self.lab, _, self.lab_lens = padMatrix(lab, max_numcode_pervisit, max_len, pad_id_list[2])
        self.proc, _, self.proc_lens = padMatrix(proc, max_numcode_pervisit, max_len, pad_id_list[3])

        self.diag_mask = codeMask_inverse(diag, max_numcode_pervisit, max_len, nan_id_list[0])
        self.drug_mask = codeMask_inverse(drug, max_numcode_pervisit, max_len, nan_id_list[1])
        self.lab_mask = codeMask_inverse(lab, max_numcode_pervisit, max_len, nan_id_list[2])
        self.proc_mask = codeMask_inverse(proc, max_numcode_pervisit, max_len, nan_id_list[3])


        self.device = device

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return torch.tensor(self.label[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.diag[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.drug[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.lab[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.proc[idx], dtype=torch.long).to(self.device),\
        torch.tensor(self.demo[idx], dtype=torch.float).to(self.device)

def gen_collate_fn_multimodal(batch):
    label, diag, drug, lab, proc, demo = [], [], [], [], [], []
    for data in batch:
        label.append(data[0])
        diag.append(data[1])
        drug.append(data[2])
        lab.append(data[3])
        proc.append(data[4])
        demo.append(data[5])
    return torch.stack(label,0), torch.stack(diag,0), torch.stack(drug,0), torch.stack(lab,0), torch.stack(proc,0), torch.stack(demo,0)

class pancreas_Gendataset_multimodal_for_baselines(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, pad_id_list, nan_id_list, task, device):

        data = pd.read_csv(dir_ehr)
        self.label = data['MORTALITY'].tolist()

        diag = data['DIAGNOSES_int'].apply(lambda x: ast.literal_eval(x)).tolist()
        drug = data['DRG_CODE_int'].apply(lambda x: ast.literal_eval(x)).tolist()
        lab = data['LAB_ITEM_int'].apply(lambda x: ast.literal_eval(x)).tolist()
        proc = data['PROC_ITEM_int'].apply(lambda x: ast.literal_eval(x)).tolist()

        self.demo = data['demo'].apply(lambda x: ast.literal_eval(x)).tolist()
        # time_steps = data['time_gaps'].apply(lambda x: ast.literal_eval(x)).tolist()

        self.diag, _, self.diag_lens = padMatrix(diag, max_numcode_pervisit, max_len, pad_id_list[0])
        self.drug, _, self.drug_lens = padMatrix(drug, max_numcode_pervisit, max_len, pad_id_list[1])
        self.lab, _, self.lab_lens = padMatrix(lab, max_numcode_pervisit, max_len, pad_id_list[2])
        self.proc, _, self.proc_lens = padMatrix(proc, max_numcode_pervisit, max_len, pad_id_list[3])

        self.diag_mask = codeMask_inverse(diag, max_numcode_pervisit, max_len, nan_id_list[0])
        self.drug_mask = codeMask_inverse(drug, max_numcode_pervisit, max_len, nan_id_list[1])
        self.lab_mask = codeMask_inverse(lab, max_numcode_pervisit, max_len, nan_id_list[2])
        self.proc_mask = codeMask_inverse(proc, max_numcode_pervisit, max_len, nan_id_list[3])

        self.device = device

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return torch.tensor(self.label[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.diag[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.drug[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.lab[idx], dtype=torch.long).to(self.device),\
            torch.tensor(self.proc[idx], dtype=torch.long).to(self.device),\
        torch.tensor(self.demo[idx], dtype=torch.float).to(self.device)

