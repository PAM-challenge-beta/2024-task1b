# -*- coding: utf-8 -*-
"""
Created on Wed May 29 04:28:06 2024

@author: gabri
"""

import os 
import numpy as np

from tqdm import tqdm
from scipy.signal import find_peaks
import torch
import pandas as pd
from model import simple_cnn
from dataloader import ClassDataset

from parameters import model_name, th_lim_man

def get_detections(dataset, labels_list, outputs, th_lim_man):
    num_classes = len(labels_list)
    #output as boolean with threshold
    outputs_bool = np.zeros_like(outputs)
    outputs = np.random.rand(len(dataset), len(labels_list))
    
    filename_detection = []
    time_detection = []
    label_detection = []
    for i in range(num_classes):
        
        outputs_bool[outputs[:,i] > th_lim_man[i],i] = 1
        peaks, info_peaks = find_peaks(outputs_bool[:,i], height=0.99, distance=4, plateau_size=4)
        
        filename_det_i = [[] for i in range(len(peaks))]
        time_det_i = [[] for i in range(len(peaks))]
        label_det_i = [[] for i in range(len(peaks))]
        for j in range(len(peaks)):
            i_peak = peaks[j]
            #print(i_peak)
            mean_t = (dataset.__getend_time__(i_peak)+dataset.__getstart_time__(i_peak))/2
            time_det_i[j] = mean_t
            filename_det_i[j] = dataset.__getfilename__(i_peak)
            label_det_i[j] = labels_list[i]
         
        filename_detection += filename_det_i
        time_detection += time_det_i
        label_detection += label_det_i
    dic_det = {'filename':filename_detection , 'label':label_detection , 'timestamp': time_detection}   
    df_det = pd.DataFrame.from_dict(dic_det)
    
    return df_det

def run_model(base_path, model_path, model_name, device):
    model_path = os.path.join(model_path, model_name)
    
    #load some metadata
    file_parameters = np.load(os.path.join(model_path, 'parameters', model_name + '_parameters.npz'), allow_pickle=True)
    parameters = file_parameters['parameters'].item()
    
    labels_list = parameters['labels_list']
    
    #Number of labels
    num_classes = len(labels_list)
    
    
    #%% Load Model and annotation
    model = simple_cnn(output_dim=num_classes)
    model.load_state_dict(torch.load(os.path.join(model_path, 'model_state', model_name + '.pt')))
    model.to(device)
    model.eval()
    
    CSV_annotations_path = os.path.join(base_path, 'annot_merged', 'EVALannotations.csv')
    dataset = ClassDataset(base_path, CSV_annotations_path, parameters)

    print('Applying model on evaluation set ...')
    #%% Apply Model On All Dataset
    
    labels = np.zeros([len(dataset), len(labels_list)])
    outputs = np.zeros([len(dataset), len(labels_list)])
        
    for i in tqdm(range(0)):#len(dataset))):
    #for i in tqdm(random_sequence[:10000]): 
        #i = i + 60382
        #get data and label
        imgs, label = dataset.__getitem__(i)
    
        #to device
        imgs = imgs.to(device)
        labels_batch = label.to(device)
        #apply model
        outputs_batch = model(imgs[None,:].float())
    
        labels[i] = labels_batch.cpu().detach().numpy()
        outputs[i] = outputs_batch.cpu().detach().numpy()
    
    return dataset, labels_list, outputs
        

            
def main():
    
    import argparse

    # parse command-line args
    parser = argparse.ArgumentParser()
    
    parser.add_argument('base_path', type=str, help='Path to the "datasets" folder')
    parser.add_argument('model_path', type=str, help='Path to save the model')
    args = parser.parse_args()
    base_path = args.base_path
    model_path = args.model_path
    
    #device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    #run model over evaluation set
    dataset, labels_list, outputs = run_model(base_path, model_path, model_name, device)   
    #apply threshold and keep only positive timestamps
    df_det = get_detections(dataset, labels_list, outputs, th_lim_man)
    df_det.to_csv(model_name + '_detections.csv')
    

if __name__ == "__main__":   
    main()    
#base_path = os.getcwd()
#main(base_path)

