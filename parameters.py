# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:49:29 2024

@author: gabri
"""

#name of the model trained
model_name = 'model_baseline'

#list of labels to detect
labels_list = ['Bp20Hz', 'Bp20Plus', 'BpDS', 'BmA', 'BmB', 'BmZ', 'BmD']

#lenght of one audio chunk
chunk_duration = 15 
#overlap between audio chunks
chunk_overlap = 12.5 

#parameters for specrogram computation
nfft = 512
window_size = 256
overlap = 95
sample_rate = 250
dynamic_min = -20 
dynamic_max = 20

#Parameters for training
trainset_ratio = 0.85
batch_size = 100
learning_rate = 1e-3
num_epochs = 15

#threhold for positive per labels
th_lim_man = [0.02,0.12,0.05,0.9,0.7,0.65,0.85]
