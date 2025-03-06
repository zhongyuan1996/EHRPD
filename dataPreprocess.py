import sys
import pickle as pickle
from datetime import datetime

import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
import mimic_data_gen

def convert_to_3digit_icd9(dxStr):
    if not isinstance(dxStr, str):
        dxStr = str(dxStr)
    if dxStr.startswith('E'):
        if len(dxStr) > 4:
            return dxStr[:4]
        else:
            return dxStr
    else:
        if len(dxStr) > 3:
            return dxStr[:3]
        else:
            return dxStr
def convert_proc_to_shorter_icd9(dxStr, length = 3):
    if not isinstance(dxStr, str):
        dxStr = str(dxStr)
    if len(dxStr) > length:
        return dxStr[:length]
    else:
        return dxStr

def chop_formulary_code(row):

    if 'D5W' in row['FORMULARY_DRUG_CD'] or 'D5W' in row['DRUG']:
        return 'D5W'
    if 'NS' in row['FORMULARY_DRUG_CD']:
        return 'NS'
    if 'BAG' in row['FORMULARY_DRUG_CD']:
        return 'BAG'
    if 'AA5D' in row['FORMULARY_DRUG_CD']:
        return 'AA5D'
    if '5FU' in row['FORMULARY_DRUG_CD']:
        return '5FU'
    #find the first numeric character in the string and return the string before it
    chopped_code = row['FORMULARY_DRUG_CD'][:next((i for i, c in enumerate(row['FORMULARY_DRUG_CD']) if c.isdigit()), len(row['FORMULARY_DRUG_CD']))]
    return str(chopped_code)

def main(short_ICD = True, ICD9 = True, proc_digits = 5, drug_agg = True):
    # print(raw_drug['PROD_STRENGTH'].nunique()) # 100
    #loading raw 4 modalities files, their 4 icd name files, and the raw patietn file
    raw_admission = pd.read_csv('ADMISSIONS.csv')
    raw_patient = pd.read_csv('PATIENTS.csv')
    # raw_diag2icd = pd.read_csv('D_ICD_DIAGNOSES.csv')
    raw_diag = pd.read_csv('DIAGNOSES_ICD.csv')
    # raw_drug = pd.read_csv('DRGCODES.csv')
    raw_drug = pd.read_csv('PRESCRIPTIONS.csv')
    raw_drug['FORMULARY_DRUG_CD'] = raw_drug['FORMULARY_DRUG_CD'].astype(str)
    # raw_lab2icd = pd.read_csv('D_LABITEMS.csv')
    raw_lab = pd.read_csv('LABEVENTS.csv')
    # raw_proc2icd = pd.read_csv('D_ICD_PROCEDURES.csv')
    raw_proc = pd.read_csv('PROCEDURES_ICD.csv')

    #create df with patient SUBJECT_ID and EXPIRE_FLAG
    df = pd.DataFrame({
        'SUBJECT_ID': raw_patient['SUBJECT_ID'],
        'MORTALITY': raw_patient['DOD_HOSP'].notnull().astype(int)
    })
    assert df['SUBJECT_ID'].nunique() == len(df)
    #join df with admission table to get HADM_ID
    # df = df.merge(raw_admission[['SUBJECT_ID', 'HADM_ID']], on='SUBJECT_ID', how='left')
    # assert df['HADM_ID'].nunique() == len(df)

    #Lab item is left as it is
    #Preprocessing raw_diag to 3 digit ICD9 code
    if short_ICD and ICD9:
        raw_diag['diag_CODE'] = raw_diag['ICD9_CODE'].apply(convert_to_3digit_icd9)
        print('Total number of unique ICD9 diagnosis codes: ')
        print(raw_diag['diag_CODE'].nunique())
    if not short_ICD and ICD9:
        raw_diag['diag_CODE'] = raw_diag['ICD9_CODE']
        print('Total number of unique ICD9 diagnosis codes: ')
        print(raw_diag['diag_CODE'].nunique())
    #Preprocessing raw_proc to 3 or 2 digits
    if proc_digits != 5:
        raw_proc['proc_CODE'] = raw_proc['ICD9_CODE'].apply(convert_proc_to_shorter_icd9, proc_digits)
        print('Total number of unique ICD9 procedure codes: ')
        print(raw_proc['proc_CODE'].nunique())
    else:
        raw_proc['proc_CODE'] = raw_proc['ICD9_CODE']
        print('Total number of unique ICD9 procedure codes: ')
        print(raw_proc['proc_CODE'].nunique())
    #Preprocessing drugs in raw_drug to shrink the size of the table
    if drug_agg:
        raw_drug['DRUG'] = raw_drug.apply(chop_formulary_code, axis=1)
        print('Total number of unique drugs: ')
        print(raw_drug['DRUG'].nunique())
    else:
        raw_drug['DRUG'] = raw_drug['FORMULARY_DRUG_CD']
        print('Total number of unique drugs: ')
        print(raw_drug['DRUG'].nunique())

    #For all 4 modalities, join df with the corresponding modality table to get the corresponding codes

    ############################################################################################################
    #diagnosis

    #add the ADMITTIME and DISCHTIME to the raw_diag
    raw_diag = pd.merge(raw_diag, raw_admission[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME']], on=['SUBJECT_ID', 'HADM_ID'], how='left')

    # Function to aggregate diagnosis information into lists
    def aggregate_diagnoses(group):
        diagnoses = list(group['diag_CODE'])  # Replace with 'ICD10_CODE' if using ICD-10
        #convert codes to string
        diagnoses = [str(code) for code in diagnoses]
        admit_time = group['ADMITTIME'].iloc[0]
        disch_time = group['DISCHTIME'].iloc[0]
        return pd.Series([diagnoses, admit_time, disch_time], index=['DIAGNOSES', 'ADMIT_TIME', 'DISCH_TIME'])

    # Apply the aggregation function and sort the data
    aggregated_diag = raw_diag.groupby(['SUBJECT_ID', 'HADM_ID']).apply(aggregate_diagnoses).reset_index()
    aggregated_diag = aggregated_diag.sort_values(by=['SUBJECT_ID', 'ADMIT_TIME'])

    # Group by 'SUBJECT_ID' and aggregate data into lists
    grouped_diag = aggregated_diag.groupby('SUBJECT_ID').agg({'HADM_ID': lambda x: list(x),
                                                              'DIAGNOSES': lambda x: list(x),
                                                              'ADMIT_TIME': lambda x: list(x),
                                                              'DISCH_TIME': lambda x: list(x)}).reset_index()
    print('Grouping diagnoses by SUBJECT_ID')
    print(grouped_diag['SUBJECT_ID'].nunique())

    ############################################################################################################
    #drug
    raw_drug = pd.merge(raw_drug, raw_admission[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME']], on=['SUBJECT_ID', 'HADM_ID'], how='left')

    def aggregate_drugs(group):
        drugs = list(group['DRUG'].drop_duplicates())
        admit_time = group['ADMITTIME'].iloc[0]
        disch_time = group['DISCHTIME'].iloc[0]
        return pd.Series([drugs, admit_time, disch_time], index=['DRG_CODE', 'ADMIT_TIME', 'DISCH_TIME'])

    aggregated_drug = raw_drug.groupby(['SUBJECT_ID', 'HADM_ID']).apply(aggregate_drugs).reset_index()
    aggregated_drug = aggregated_drug.sort_values(by=['SUBJECT_ID', 'ADMIT_TIME'])

    grouped_drug = aggregated_drug.groupby('SUBJECT_ID').agg({'HADM_ID': lambda x: list(x),
                                                              'DRG_CODE': lambda x: list(x),
                                                              'ADMIT_TIME': lambda x: list(x),
                                                              'DISCH_TIME': lambda x: list(x)}).reset_index()
    print('Grouping drugs by SUBJECT_ID')
    print(grouped_drug['SUBJECT_ID'].nunique())
    ############################################################################################################
    #lab
    raw_lab = raw_lab.dropna(subset=['HADM_ID']).dropna(subset=['ITEMID'])
    raw_lab = pd.merge(raw_lab, raw_admission[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME']], on=['SUBJECT_ID', 'HADM_ID'], how='left')
    raw_lab['HADM_ID'] = raw_lab['HADM_ID'].astype(int)

    def aggregate_labs(group):
        labs = list(group['ITEMID'].drop_duplicates())
        admit_time = group['ADMITTIME'].iloc[0]
        disch_time = group['DISCHTIME'].iloc[0]
        return pd.Series([labs, admit_time, disch_time], index=['LAB_ITEM', 'ADMIT_TIME', 'DISCH_TIME'])

    aggregated_lab = raw_lab.groupby(['SUBJECT_ID', 'HADM_ID']).apply(aggregate_labs).reset_index()
    aggregated_lab = aggregated_lab.sort_values(by=['SUBJECT_ID', 'ADMIT_TIME'])

    grouped_lab = aggregated_lab.groupby('SUBJECT_ID').agg({'HADM_ID': lambda x: list(x),
                                                            'LAB_ITEM': lambda x: list(x),
                                                            'ADMIT_TIME': lambda x: list(x),
                                                            'DISCH_TIME': lambda x: list(x)}).reset_index()
    print('Grouping labs by SUBJECT_ID')
    print(grouped_lab['SUBJECT_ID'].nunique())
    ############################################################################################################
    #procedure
    raw_proc = pd.merge(raw_proc, raw_admission[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME']], on=['SUBJECT_ID', 'HADM_ID'], how='left')

    def aggregate_procedures(group):
        procedures = list(group['proc_CODE'].drop_duplicates())
        admit_time = group['ADMITTIME'].iloc[0]
        disch_time = group['DISCHTIME'].iloc[0]
        return pd.Series([procedures, admit_time, disch_time], index=['PROC_ITEM', 'ADMIT_TIME', 'DISCH_TIME'])

    aggregated_proc = raw_proc.groupby(['SUBJECT_ID', 'HADM_ID']).apply(aggregate_procedures).reset_index()
    aggregated_proc = aggregated_proc.sort_values(by=['SUBJECT_ID', 'ADMIT_TIME'])

    grouped_proc = aggregated_proc.groupby('SUBJECT_ID').agg({'HADM_ID': lambda x: list(x),
                                                              'PROC_ITEM': lambda x: list(x),
                                                              'ADMIT_TIME': lambda x: list(x),
                                                              'DISCH_TIME': lambda x: list(x)}).reset_index()
    print('Grouping procedures by SUBJECT_ID')
    print(grouped_proc['SUBJECT_ID'].nunique())
    ############################################################################################################
    all_hadm_ids = pd.concat([grouped_diag[['SUBJECT_ID', 'HADM_ID', 'ADMIT_TIME', 'DISCH_TIME']].explode(['HADM_ID', 'ADMIT_TIME', 'DISCH_TIME']),
                              grouped_drug[['SUBJECT_ID', 'HADM_ID', 'ADMIT_TIME', 'DISCH_TIME']].explode(['HADM_ID', 'ADMIT_TIME', 'DISCH_TIME']),
                              grouped_lab[['SUBJECT_ID', 'HADM_ID', 'ADMIT_TIME', 'DISCH_TIME']].explode(['HADM_ID', 'ADMIT_TIME', 'DISCH_TIME']),
                              grouped_proc[['SUBJECT_ID', 'HADM_ID', 'ADMIT_TIME', 'DISCH_TIME']].explode(['HADM_ID', 'ADMIT_TIME', 'DISCH_TIME'])],
                             ignore_index=True)

    all_hadm_ids = all_hadm_ids.drop_duplicates(subset=['SUBJECT_ID', 'HADM_ID'])

    unique_hadm_ids = all_hadm_ids.groupby('SUBJECT_ID').agg({
        'HADM_ID': list,
        'ADMIT_TIME': list,
        'DISCH_TIME': list
    }).reset_index()

    # Function to add placeholders
    def add_placeholders(df, unique_hadm_ids, col_name):
        # Merge df with unique_hadm_ids
        df = df.merge(unique_hadm_ids, on='SUBJECT_ID', how='right', suffixes=('', '_unique'))

        def align_data(row):
            # Ensure 'HADM_ID' and 'col_name' are lists, even if NaN
            hadm_ids = [row['HADM_ID']] if isinstance(row['HADM_ID'], float) else row['HADM_ID']
            col_data = [row[col_name]] if isinstance(row[col_name], float) else row[col_name]

            # Mapping of existing HADM_IDs to their data
            existing_data = {hid: data for hid, data in zip(hadm_ids, col_data) if pd.notna(hid)}

            # New lists for aligned data
            aligned_hadm_ids = row['HADM_ID_unique']
            aligned_data = [existing_data.get(hid, [None]) for hid in aligned_hadm_ids]  # [None] as placeholder
            aligned_admit_times = row['ADMIT_TIME_unique']
            aligned_disch_times = row['DISCH_TIME_unique']

            return pd.Series([aligned_hadm_ids, aligned_data, aligned_admit_times, aligned_disch_times])

        # Apply the function to align data
        df[['HADM_ID', col_name, 'ADMIT_TIME', 'DISCH_TIME']] = df.apply(align_data, axis=1)

        # Drop the '_unique' columns
        df.drop(columns=['HADM_ID_unique', 'ADMIT_TIME_unique', 'DISCH_TIME_unique'], inplace=True)

        return df
    #
    # common_subject_ids = set(grouped_diag['SUBJECT_ID']) or set(grouped_drug['SUBJECT_ID']) or \
    #                      set(grouped_lab['SUBJECT_ID']) or set(grouped_proc['SUBJECT_ID'])


    ##############change the place holder addition
    ###############

    # Filter each DataFrame to include only common SUBJECT_IDs
    # df = df[df['SUBJECT_ID'].isin(common_subject_ids)]
    # grouped_diag = grouped_diag[grouped_diag['SUBJECT_ID'].isin(common_subject_ids)]
    # grouped_drug = grouped_drug[grouped_drug['SUBJECT_ID'].isin(common_subject_ids)]
    # grouped_lab = grouped_lab[grouped_lab['SUBJECT_ID'].isin(common_subject_ids)]
    # grouped_proc = grouped_proc[grouped_proc['SUBJECT_ID'].isin(common_subject_ids)]

    # Step 2: Add placeholders for missing HADM_IDs, codes, and times in each modality
    print('Adding placeholders for diagnoses')
    grouped_diag = add_placeholders(grouped_diag, unique_hadm_ids, 'DIAGNOSES')
    print('Adding placeholders for drugs')
    grouped_drug = add_placeholders(grouped_drug, unique_hadm_ids, 'DRG_CODE')
    print('Adding placeholders for labs')
    grouped_lab = add_placeholders(grouped_lab, unique_hadm_ids, 'LAB_ITEM')
    print('Adding placeholders for procedures')
    grouped_proc = add_placeholders(grouped_proc, unique_hadm_ids, 'PROC_ITEM')

    grouped_diag = grouped_diag.rename(
        columns={'HADM_ID': 'HADM_ID_diag', 'ADMIT_TIME': 'ADMIT_TIME_diag', 'DISCH_TIME': 'DISCH_TIME_diag'})
    grouped_drug = grouped_drug.rename(
        columns={'HADM_ID': 'HADM_ID_drug', 'ADMIT_TIME': 'ADMIT_TIME_drug', 'DISCH_TIME': 'DISCH_TIME_drug'})
    grouped_lab = grouped_lab.rename(
        columns={'HADM_ID': 'HADM_ID_lab', 'ADMIT_TIME': 'ADMIT_TIME_lab', 'DISCH_TIME': 'DISCH_TIME_lab'})
    grouped_proc = grouped_proc.rename(
        columns={'HADM_ID': 'HADM_ID_proc', 'ADMIT_TIME': 'ADMIT_TIME_proc', 'DISCH_TIME': 'DISCH_TIME_proc'})

    df = df.merge(grouped_diag, on='SUBJECT_ID', how='left')
    df = df.merge(grouped_drug, on='SUBJECT_ID', how='left')
    df = df.merge(grouped_lab, on='SUBJECT_ID', how='left')
    df = df.merge(grouped_proc, on='SUBJECT_ID', how='left')

    # Assuming the values should be the same, keep the first set and drop the others
    df = df.drop(columns=['HADM_ID_drug', 'ADMIT_TIME_drug', 'DISCH_TIME_drug',
                          'HADM_ID_lab', 'ADMIT_TIME_lab', 'DISCH_TIME_lab',
                          'HADM_ID_proc', 'ADMIT_TIME_proc', 'DISCH_TIME_proc'])
    df.rename(columns={'HADM_ID_diag': 'HADM_ID','ADMIT_TIME_diag': 'ADMIT_TIME', 'DISCH_TIME_diag': 'DISCH_TIME'}, inplace=True)

    # Iterate through the dataframe and check the consistency of list lengths for each patient
    for index, row in df.iterrows():
        hadm_id_length = len(row['HADM_ID'])
        diagnoses_length = len(row['DIAGNOSES'])
        drug_code_length = len(row['DRG_CODE'])
        lab_item_length = len(row['LAB_ITEM'])
        proc_item_length = len(row['PROC_ITEM'])

        if not (hadm_id_length == diagnoses_length == drug_code_length == lab_item_length == proc_item_length):
            print(f"Inconsistency found for SUBJECT_ID {row['SUBJECT_ID']}")
            print(
                f"HADM_ID length: {hadm_id_length}, DIAGNOSES length: {diagnoses_length}, DRG_CODE length: {drug_code_length}, LAB_ITEM length: {lab_item_length}, PROC_ITEM length: {proc_item_length}")
            break
    else:
        print("All patients have consistent visit lengths across modalities.")

    # columns_to_check = ['HADM_ID', 'ADMIT_TIME', 'DISCH_TIME']
    # for col in columns_to_check:
    #     if not all(df[f'{col}_diag'] == df[f'{col}_drug']) or \
    #             not all(df[f'{col}_drug'] == df[f'{col}_lab']) or \
    #             not all(df[f'{col}_lab'] == df[f'{col}_proc']):
    #         raise ValueError(f"Inconsistency found in {col} across modalities")
    #
    # columns_to_drop = [f'{col}_{suffix}' for col in columns_to_check for suffix in ['drug', 'lab', 'proc']]
    # df = df.drop(columns=columns_to_drop)
    # columns_to_rename = {f'{col}_diag': col for col in columns_to_check}
    # df = df.rename(columns=columns_to_rename)

    raw_patient['DOB'] = pd.to_datetime(raw_patient['DOB'])
    raw_admission = raw_admission.sort_values(by=['SUBJECT_ID', 'ADMITTIME'])

    # Get the last admission date for each patient
    last_admission_times = raw_admission.groupby('SUBJECT_ID')['ADMITTIME'].last()

    # Merge the last admission times with the patient dataframe
    merged_df = raw_patient.set_index('SUBJECT_ID').join(last_admission_times)
    merged_df['ADMITTIME'] = pd.to_datetime(merged_df['ADMITTIME'])
    merged_df['DOB'] = pd.to_datetime(merged_df['DOB'])

    # Calculate age at last admission
    merged_df['age'] = merged_df.apply(lambda row:
                                       (row['ADMITTIME'].year - row['DOB'].year -
                                        ((row['ADMITTIME'].month, row['ADMITTIME'].day) <
                                         (row['DOB'].month, row['DOB'].day))),
                                       axis=1)
    merged_df['age'] = merged_df['age'].apply(lambda x: -1 if x <= 0 or x > 100 else x)
    merged_df['age_missing'] = merged_df['age'].apply(lambda x: 1 if x == -1 else 0)

    # Merge age data into your main DataFrame df
    df = df.merge(merged_df[['age', 'age_missing']],
                  left_on='SUBJECT_ID',
                  right_index=True,
                  how='left')
    df = df.merge(raw_admission[['ETHNICITY', 'MARITAL_STATUS', 'RELIGION']],
                  left_on='SUBJECT_ID',
                  right_index=True,
                  how='left')
    df = df.merge(raw_patient[['GENDER']],
                  left_on='SUBJECT_ID',
                  right_index=True,
                  how='left')

    #fill nan with UNKOWN
    df = df.fillna('UNKOWN')

    one_hot_gender = pd.get_dummies(df['GENDER'], prefix='gen')
    one_hot_ethnicity = pd.get_dummies(df['ETHNICITY'], prefix='eth')
    one_hot_marital_status = pd.get_dummies(df['MARITAL_STATUS'], prefix='mstat')
    one_hot_religion = pd.get_dummies(df['RELIGION'], prefix='rel')

    df = pd.concat([df, one_hot_gender, one_hot_ethnicity, one_hot_marital_status, one_hot_religion], axis=1)
    df['Demographic'] = df.apply(lambda row: [row['age'], row['age_missing']] +
                                              row[one_hot_gender.columns.tolist() +
                                                  one_hot_ethnicity.columns.tolist() +
                                                  one_hot_marital_status.columns.tolist() +
                                                  one_hot_religion.columns.tolist()].tolist(), axis=1)

    df = df.drop(columns=one_hot_gender.columns.tolist() +
                         one_hot_ethnicity.columns.tolist() +
                         one_hot_marital_status.columns.tolist() +
                         one_hot_religion.columns.tolist() + ['age_missing'])
    #Rename age to AGE
    df = df.rename(columns={'age': 'AGE'})

    #change ADMIT_TIME and DISCH_TIME to datetime
    #values in ADMIT_TIME and DISCH_TIME are in list, so we need to convert them to datetime in day
    df['ADMIT_TIME'] = df['ADMIT_TIME'].apply(lambda x: [pd.to_datetime(i).date() for i in x])
    df['DISCH_TIME'] = df['DISCH_TIME'].apply(lambda x: [pd.to_datetime(i).date() for i in x])

    df['time_gaps'] = df['ADMIT_TIME'].apply(mimic_data_gen.calculate_time_gaps_with_last_visit)
    df['consecutive_time_gaps'] = df['time_gaps'].apply(mimic_data_gen.compute_time_gaps)

    #create a index mapping for all the codes with its index
    #get uinque codes in df['DIAGNOSES']
    all_codes_dict = {}
    for patient in df['DIAGNOSES']:
        for visit in patient:
            for code in visit:
                if code not in all_codes_dict:
                    all_codes_dict[code] = len(all_codes_dict)
    #save the code index mapping to csv file
    if short_ICD:
        with open('diagnosis_to_int_mapping_3dig.csv', 'w') as f:
            for code, int_val in all_codes_dict.items():
                f.write(f"{code},{int_val}\n")
    else:
        with open('diagnosis_to_int_mapping_5dig.csv', 'w') as f:
            for code, int_val in all_codes_dict.items():
                f.write(f"{code},{int_val}\n")

    #create a new DIAGNOSES_int column in df, which is the integer representation of the DIAGNOSES column
    df['DIAGNOSES_int'] = df['DIAGNOSES'].apply(lambda x: [[all_codes_dict[code] for code in visit] for visit in x])

    #do the same thing for DRG_CODE, LAB_ITEM, PROC_ITEM
    all_drg_dict = {}
    for patient in df['DRG_CODE']:
        for visit in patient:
            for code in visit:
                if code not in all_drg_dict:
                    all_drg_dict[code] = len(all_drg_dict)
    if drug_agg:
        with open('drug_to_int_mapping_3dig.csv', 'w') as f:
            for code, int_val in all_drg_dict.items():
                f.write(f"{code},{int_val}\n")
    else:
        with open('drug_to_int_mapping_5dig.csv', 'w') as f:
            for code, int_val in all_drg_dict.items():
                f.write(f"{code},{int_val}\n")

    df['DRG_CODE_int'] = df['DRG_CODE'].apply(lambda x: [[all_drg_dict[code] for code in visit] for visit in x])

    all_lab_dict = {}
    for patient in df['LAB_ITEM']:
        for visit in patient:
            for code in visit:
                if code not in all_lab_dict:
                    all_lab_dict[code] = len(all_lab_dict)
    if short_ICD:
        with open('lab_to_int_mapping_3dig.csv', 'w') as f:
            for code, int_val in all_lab_dict.items():
                f.write(f"{code},{int_val}\n")
    else:
        with open('lab_to_int_mapping_5dig.csv', 'w') as f:
            for code, int_val in all_lab_dict.items():
                f.write(f"{code},{int_val}\n")

    df['LAB_ITEM_int'] = df['LAB_ITEM'].apply(lambda x: [[all_lab_dict[code] for code in visit] for visit in x])

    all_proc_dict = {}
    for patient in df['PROC_ITEM']:
        for visit in patient:
            for code in visit:
                if code not in all_proc_dict:
                    all_proc_dict[code] = len(all_proc_dict)
    if short_ICD:
        with open('proc_to_int_mapping_3dig.csv', 'w') as f:
            for code, int_val in all_proc_dict.items():
                f.write(f"{code},{int_val}\n")
    else:
        with open('proc_to_int_mapping_5dig.csv', 'w') as f:
            for code, int_val in all_proc_dict.items():
                f.write(f"{code},{int_val}\n")
    df['PROC_ITEM_int'] = df['PROC_ITEM'].apply(lambda x: [[all_proc_dict[code] for code in visit] for visit in x])

    code_time_gaps = []

    for patient, patient_timegaps in zip(df['DIAGNOSES_int'], df['consecutive_time_gaps']):
        patient_code_timegaps = mimic_data_gen.compute_code_time_gaps(patient, patient_timegaps)
        code_time_gaps.append(patient_code_timegaps)

    df['code_time_gaps'] = code_time_gaps

    drug_time_gaps = []
    for patient, patient_timegaps in zip(df['DRG_CODE_int'], df['consecutive_time_gaps']):
        patient_code_timegaps = mimic_data_gen.compute_code_time_gaps(patient, patient_timegaps)
        drug_time_gaps.append(patient_code_timegaps)
    df['drug_time_gaps'] = drug_time_gaps

    lab_time_gaps = []
    for patient, patient_timegaps in zip(df['LAB_ITEM_int'], df['consecutive_time_gaps']):
        patient_code_timegaps = mimic_data_gen.compute_code_time_gaps(patient, patient_timegaps)
        lab_time_gaps.append(patient_code_timegaps)
    df['lab_time_gaps'] = lab_time_gaps

    proc_time_gaps = []
    for patient, patient_timegaps in zip(df['PROC_ITEM_int'], df['consecutive_time_gaps']):
        patient_code_timegaps = mimic_data_gen.compute_code_time_gaps(patient, patient_timegaps)
        proc_time_gaps.append(patient_code_timegaps)
    df['proc_time_gaps'] = proc_time_gaps

    df.to_csv('mimic_4moda_with_timegaps.csv', index=False)
    train_r = 0.75
    test_r = 0.1
    val_r = 0.15
    seed = 1234
    train, remaining = train_test_split(df, train_size=train_r, random_state=seed)
    test, val = train_test_split(remaining, train_size=test_r / (test_r + val_r), random_state=seed)
    toy = train.head(128)

    if short_ICD:
        train.to_csv('train_' + '3dig' + 'mimic.csv', index=False)
        test.to_csv('test_' + '3dig' + 'mimic.csv', index=False)
        val.to_csv('val_' + '3dig' + 'mimic.csv', index=False)
        toy.to_csv('toy_' + '3dig' + 'mimic.csv', index=False)

    else:
        train.to_csv('train_' + '5dig' + 'mimic.csv', index=False)
        test.to_csv('test_' + '5dig' + 'mimic.csv', index=False)
        val.to_csv('val_' + '5dig' + 'mimic.csv', index=False)
        toy.to_csv('toy_' + '5dig' + 'mimic.csv', index=False)


if __name__ == '__main__':
    main(True, True, 3, True)
    main(False, True, 5, False)


