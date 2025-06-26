import json
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import nibabel as nib


array_type = np.float32

def __convert_dict__(meta_data):
    out_data = []
    for this_data in meta_data:
        if not this_data:
            this_data = 0
        out_data.append(this_data)
    return out_data
class CSVDDataset(Dataset):
  
    def __init__(self, PATIENTS_ARCHIVES_path, patient_lst_txt_path,
                 use_radiomics=False, use_label=True, transform=None, target_transform=None):
        self.directory = PATIENTS_ARCHIVES_path
        self.transform = transform
        self.target_transform = target_transform
        self.__init_items__(use_radiomics, use_label)
        with open(patient_lst_txt_path, 'r', encoding='utf-8') as f:
            patient_lst = f.readlines()
            self.patients = [patient[:-1] if patient[-1] == '\n' else patient for patient in patient_lst]
        for indpt, pt in enumerate(self.patients):
            pt_dir = os.path.join(self.directory, pt)
            if self.use_label:
                label = int(os.listdir(os.path.join(pt_dir, 'Label'))[0][:-4]) - 1
                self.labels.append(label)
            if self.use_radiomics:
                radiomics_directory = os.path.join(pt_dir, 'Normed_Radiomics')
                radiomics_subdirectory = os.listdir(radiomics_directory)
                radiomics = []
                for subdirectory in radiomics_subdirectory:
                    if subdirectory != 'WMH':
                        continue
                    subdirectory_path = os.path.join(radiomics_directory, subdirectory)
                    if not self.radiomics:
                        radiomics_file_lst = os.listdir(subdirectory_path)
                    for radiomics_file in radiomics_file_lst:
                        radiomics.append(os.path.join(subdirectory_path, radiomics_file))
                self.radiomics.append(radiomics)
        print('Data Set Size = {}'.format(self.__len__()))
        print('Label Distribution:')
        self.__print_dist__()
    def __getitem__(self, index):
        patients = dict()
        if self.use_label:
            patients['Label'] = self.labels[index]
        if self.use_radiomics:
            patients['Radiomics'] = []
            patients['Radiomics_Names'] = []
            for radio_ind in self.radiomics[index]:
                radiomics_info = np.loadtxt(radio_ind).astype(array_type)
                patients['Radiomics'].append(radiomics_info)
                patients['Radiomics_Names'].append(radio_ind)
        patients['Name'] = self.patients[index]
        return patients
    def __init_items__(self, use_radiomics,use_label):
        self.use_radiomics = use_radiomics
        self.use_label = use_label
        if self.use_radiomics:
            self.radiomics = []
        if self.use_label:
            self.labels = []
    def __len__(self):
        return len(self.patients)
    def __print_dist__(self):
        for i in range(max(self.labels) + 1):
            print('{:.2f}% Label = {}'.format((np.count_nonzero(np.array(self.labels) == i) * 100 / len(self.labels)),
                                              i)) 