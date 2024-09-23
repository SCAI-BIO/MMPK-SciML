########################################################################################################################
# Data preprocessing Suni project                                                                                      #
# Created by: Olga Teplytska                                                                                           #
# last edited: 11.06.24 (Final correction data)                                                                        #
# this has been written in Python 3.11 and may not work the same in future versions!                                   #
#                                                                                                                      #
# data used:                                                                                                           #
# Datensatz_AF_FK_Sunitinib_final_raw.csv: dataset for every split with set column to be split on, corrected           #
# Datensatz_AF_FK_Sunitinib_final_aug_split_10.csv: augmentation datasets split_1 - split_10                           #
########################################################################################################################

import pandas as pd
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

data = pd.read_csv("Datensatz_AF_FK_Sunitinib_final_raw.csv", delimiter=';', encoding="utf-8") # only clinical data
# data = pd.read_csv("Datensatz_AF_FK_Sunitinib_final_aug_split_10.csv", delimiter=';', encoding="utf-8") # augmented data split 10

# drop ID ET0800 (no measurements), Baseline measurements and BQL measurements 
# Drop rows where 'DV' column is empty, NaN, or None
data = data.drop(data[
                     (data['PATIENT_ID'] == 'ET0800') |
                     (data['CYC'] == 'BASELINE') |
                     (data['DV'] == 'BQL') |
                     (pd.isna(data['DV']))].index).reset_index(drop=True)


# convert doses to numbers which are falsely strings
data['DOS'] = pd.to_numeric(data['DOS'], errors='coerce')

# convert dose to ng for consistency
data['DOS'] = data['DOS'] * 1000

# drop columns to drop permanently: times, cycle, biomarkers, metabolite concentrations and all SNP (single nucleotide polymorphisms)
# and body surface area and blood counts, because there is too much missing data
columns_to_permanently_drop = ['PATIENT_ID', 'CYC', 'STUDY', 'SMOKE', 'TIME', 'DAT_MEAS', 'DV_MET',
                               'DAT_BP', 'TIME_BP', 'BP_SYS', 'BP_DIA', 'PULS',
                               'ALT', 'AST', 'ALB', 'ALKP', 'CAL', 'SCREA', 'LDH', 'BILI', 'CYP1', 'ABCR1',
                               'ABCR2', 'ABCR3', 'VEGFA1', 'VEGFA2', 'VEGFA3', 'VEGFA4',
                               'KDR1', 'FLT1', 'FLT2', 'FLT3', 'IL8', 'HAPAB', 'HAPVG','sVEGFR2','sVEGFR3']

data = data.drop(columns=columns_to_permanently_drop)

# Save preprocessed data
filename = 'suni_data_split_0_aug_preproc.csv' # only clinical data
# filename = 'suni_data_split_10_100_aug_preproc.csv'# augmented data split 10
data.to_csv(filename, index=False)
