# -*- coding: utf-8 -*-
"""
Created on Wed May 29 02:34:13 2024

@author: gabriel
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import librosa
import os

from utils import normalize, gen_spectro

# CLASS DATASET
class ClassDataset(Dataset):
    def __init__(self, base_path, annotation_file, parameters):
        self.parameters = parameters
        self.base_path = base_path
        self.annotations = pd.read_csv(annotation_file)
        #self.gen_spectro = gen_spectro
        self.transform = normalize
                
        #parameters spectrograms
        self.nfft = parameters['nfft']
        self.window_size = parameters['window_size']
        self.overlap = parameters['overlap']
        self.sample_rate = parameters['sample_rate']
        self.labels_list = parameters['labels_list']        
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        filepath = os.path.join(self.base_path, self.annotations['filename'][index]) 
        start_time = self.annotations['start'][index]
        end_time = self.annotations['end'][index]
        input_data, _ = librosa.load(filepath, sr=self.sample_rate, offset=start_time, duration=end_time-start_time)
        
        spectro,_ = gen_spectro(input_data, self.sample_rate, self.window_size, self.overlap, self.nfft)
        if self.transform is not None:
            spectro = self.transform(spectro, self.parameters)
            
        input_label = torch.zeros(len(self.labels_list))
        for j in range(len(self.labels_list)):
            input_label[j] = self.annotations[self.labels_list[j]][index]
            
        return (spectro, input_label)
    
    def __getstart_time__(self, index):
        dataset = self.annotations['start'][index] 
        return dataset
    def __getend_time__(self, index):
        dataset = self.annotations['end'][index] 
        return dataset
    def __getfilename__(self, index):
        dataset = self.annotations['filename'][index] 
        return dataset
    
    
