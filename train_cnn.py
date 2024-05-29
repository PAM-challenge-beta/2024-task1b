# -*- coding: utf-8 -*-
"""
Created on Wed May 29 02:32:54 2024

@author: gabri
"""

import numpy as np
import os
import pandas as pd
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import random

from dataloader import ClassDataset
from model import simple_cnn
from parameters import model_name, labels_list, trainset_ratio, batch_size, learning_rate, num_epochs, nfft, window_size, overlap, sample_rate, dynamic_min, dynamic_max


#Train Loop
def train_over_dev(device, model, criterion, optimizer, num_epochs, train_loader, validation_loader, model_name, model_path):
        #initialize loss tab for plot
        loss_tab_train = []
        loss_tab_validation = []
        model.train()
        for epoch in range(num_epochs):
            torch.cuda.empty_cache()
            model.train()
            ite = 1
            loss_sum_train = 0
            epoch_p = epoch + 1
            #Loop For over train set
            for data, labels in train_loader:
                #load data and label, send them to device
                data = data.to(device)
                labels = labels.to(device)
                #apply model
                outputs = model(data.float())
                #compute loss and backward 
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #loss mean over the epoch
                loss_sum_train += loss.item()
                PrintedLine = f"Epoch TRAIN [{epoch_p}/{num_epochs}]" + '  -- Loss = '+ str(loss_sum_train/ite) + '  --  ' + f"iteration [{ite}/{len(train_loader)}]" 
                sys.stdout.write('\r'+PrintedLine)
                ite += 1
                
            loss_tab_train.append(loss_sum_train/(ite-1))
            
            print('  ')
            torch.cuda.empty_cache()
            with torch.no_grad():
                #validation 
                model.eval()
                ite = 1
                loss_sum_validation = 0
                for data, labels in validation_loader:
                    data = data.to(device)
                    labels = labels.to(device)
                    outputs = model(data.float())
                    loss = criterion(outputs, labels)
                    loss_sum_validation += loss.item()
                    PrintedLine = f"Epoch VALID [{epoch_p}/{num_epochs}]" + '  -- Loss = '+ str(loss_sum_validation/ite) + '  --  ' + f"iteration [{ite}/{len(validation_loader)}]" 
                    sys.stdout.write('\r'+PrintedLine)
                    ite += 1
                    
            loss_tab_validation.append(loss_sum_validation/(ite-1))
            print('  ')
            
            #%% Save Model 
            #save model
            torch.save(model.state_dict(), os.path.join(model_path, 'model_state','sub_state', model_name + '_epoch'+ str(int(epoch_p))+ '.pt'))
        return loss_tab_train, loss_tab_validation
    

def train_model_main(base_path, model_path, parameters):
        
    #%% DEVICE 
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    print('Device used : ', device)
    
    #model name
    model_name = parameters['model_name']
    #label to detect
    labels_list = parameters['labels_list']
    num_classes = len(labels_list)
    #parameters trainning
    trainset_ratio = parameters['trainset_ratio']
    batch_size = parameters['batch_size']
    learning_rate = parameters['learning_rate']
    num_epochs = parameters['num_epochs']
    
    #path for save trainng data
    if not os.path.exists(model_path):
        os.makedirs(os.path.join(model_path, 'parameters'))
        os.makedirs(os.path.join(model_path, 'model_state'))
        os.makedirs(os.path.join(model_path, 'model_state','sub_state'))
          
    # Def model
    capacity = 64
    model = simple_cnn(output_dim=num_classes, c=capacity)
    model.to(device)
    params_to_update = model.parameters()
    optimizer = torch.optim.Adam(params_to_update, lr=learning_rate)
    criterion = nn.BCELoss()

    #%% def datasets
    pin_memory = True
    num_workers = 1
    drop_last = True
    CSV_annotations_path = os.path.join(base_path, 'annot_merged', 'DEVannotations.csv')
    dataset = ClassDataset(base_path, CSV_annotations_path, parameters)

    #Import annotation
    train_df_dev = pd.read_csv(os.path.join(base_path, 'annot_merged', 'DEVannotations.csv'))
    # Number of files for developpment
    NbFile = len(train_df_dev)
    random_sequence = list(np.linspace(0,NbFile-1, NbFile, dtype=int))
    random.shuffle(random_sequence)

    # Created using shuffled indices from 0 to train_size.
    train_set = torch.utils.data.Subset(dataset, random_sequence[:int(trainset_ratio*NbFile)])
    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory, drop_last=drop_last)
    
    # Created using indices from Train_size to the end.
    validation_set = torch.utils.data.Subset(dataset, random_sequence[int(trainset_ratio*NbFile):])
    validation_loader = DataLoader(dataset=validation_set, shuffle=True, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory, drop_last=drop_last)
    
    #save parameters
    parameters['model_name'] = model_name    
    np.savez(os.path.join(model_path, 'parameters', model_name + '_parameters.npz'), parameters=parameters)
    
    print('TRAINNING : ')
    #Launch the training 
    train_over_dev(device, model, criterion, optimizer, num_epochs, train_loader, validation_loader, model_name, model_path)
    print('DONE')
    
    #save model
    torch.save(model.state_dict(), os.path.join(model_path, 'model_state' + os.sep + model_name + '.pt'))
    

def main():
    
    import argparse

    # parse command-line args
    parser = argparse.ArgumentParser()
    
    parser.add_argument('base_path', type=str, help='Path to the "datasets" folder')
    parser.add_argument('model_path', type=str, help='Path to save the model')
    args = parser.parse_args()
    base_path = args.base_path
    model_path = args.model_path
    
    parameters = {'model_name':model_name, 'trainset_ratio':trainset_ratio, 'batch_size':batch_size, 'learning_rate':learning_rate, 'num_epochs':num_epochs, 'nfft':nfft, 'window_size':window_size, 'overlap':overlap, 'sample_rate':sample_rate, 'labels_list':labels_list, 'dynamic_min':dynamic_min, 'dynamic_max':dynamic_max}    
    
    train_model_main(base_path, model_path, parameters)
    
if __name__ == "__main__":   
    main()
    
    
    
#base_path = os.getcwd()
#main(base_path, parameters)