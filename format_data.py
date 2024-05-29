# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:59:56 2024

@author: gabri
"""
#%% Import Bibli
import numpy as np
import os
import glob
import pandas as pd
from tqdm import tqdm
import librosa
import random

from utils import comp_overlap
from parameters import labels_list, chunk_duration, chunk_overlap

#%% MAIN
def merge_audio_with_annotations(base_path, dataset_ID, annotation_file, labels_list, chunk_duration, chunk_overlap, minimal_area_for_positive):
    print('Processing dataset : ' + dataset_ID)
        
    #List all wav file from dataset
    path_audio_files = os.path.join(base_path, 'datasets', dataset_ID,'audio')
    list_wavfile = sorted([os.path.relpath(x, base_path) for x in glob.glob(os.path.join(path_audio_files , '*wav'))])
    
    #Load Annotation files
    xl_annotations = pd.read_csv(annotation_file)
    
    grouped_file_annotation = xl_annotations.groupby('filename')
    
    # Prepare CSV annotation for DL
    #columns_name = ['filename', 'start_time', 'end_time'] + labels_list

    filename_chunk = []
    start_time_chunk = []
    end_time_chunk = []
    labels_mat = []
    
    for id_file in tqdm(range(len(list_wavfile)), desc='Checking audio files ...'):
        
        duration_i = librosa.get_duration(filename=os.path.join(base_path, list_wavfile[id_file]))
        chunk_offset = chunk_duration - chunk_overlap
        nb_chunk = int((duration_i - chunk_duration) / chunk_offset) + 1
        
        if list_wavfile[id_file] in grouped_file_annotation.groups.keys():
            all_neg = 0
            annot_group = grouped_file_annotation.get_group(list_wavfile[id_file])
        
        else: all_neg = 1
        
        for i_chunk in range(nb_chunk):
            st_chunk = i_chunk*chunk_offset
            et_chunk = i_chunk*chunk_offset + chunk_duration
            
            labels_mat.append([0 for i in range(len(labels_list))])
                            
            if all_neg == 0: 
                for _, i_annotation in annot_group.iterrows():
                    overlap = comp_overlap(st_chunk, et_chunk, i_annotation['start'], i_annotation['end'])
                    if overlap / abs(i_annotation['end'] - i_annotation['start']) > minimal_area_for_positive:
                        labels_mat[-1][labels_list.index(i_annotation['label'])] = 1
                        
            start_time_chunk.append(st_chunk)
            end_time_chunk.append(et_chunk)
            filename_chunk.append(list_wavfile[id_file])
                
    labels_mat = np.array(labels_mat)
    dic_for_df =  {'filename':filename_chunk, 'start':start_time_chunk, 'end':end_time_chunk}
    for i_label in range(len(labels_list)):
        dic_for_df[labels_list[i_label]] = labels_mat[:,i_label]
    
    format_dataset_df = pd.DataFrame.from_dict(dic_for_df)
    return format_dataset_df

def format_data_main(base_path, dataset_ID_tab, labels_list, chunk_duration, chunk_overlap, minimal_area_for_positive):

    Nb_dataset = len(dataset_ID_tab)
    Dic_all_DF = {}
    #for each datasets
    for i in range(Nb_dataset):
        #dataset name
        dataset_ID = dataset_ID_tab[i]
        #annotation file 
        annotation_file = os.path.join(base_path, 'datasets', dataset_ID, 'annotations', 'annotations_'+dataset_ID+'.csv')
        
        format_dataset_df = merge_audio_with_annotations(base_path, dataset_ID, annotation_file, labels_list, chunk_duration, chunk_overlap, minimal_area_for_positive)

        Dic_all_DF[dataset_ID] = format_dataset_df 
    
    
    print('Merging All Datasets ...')
    final_df = pd.concat(Dic_all_DF.values(), ignore_index=True)

    # Save DataFrame as csv
    #Create Paths
    if not os.path.exists(os.path.join(base_path, 'annot_merged')):
        os.makedirs(os.path.join(base_path, 'annot_merged'))  
    final_df.to_csv(os.path.join(base_path, 'annot_merged', 'ALLannotations.csv'), index = False, header=True)
    print('OK')
    
def split_dev_eval(base_path, dataset_ID_tab, labels_list, dataset_for_dev, dataset_for_eval):
    '''
    Dev: keep all positive chunk and as many negative chunk
    Eval: keep all chunk
    '''
    train_df = pd.read_csv(os.path.join(base_path, 'annot_merged', 'ALLannotations.csv'))
    NbFile = len(train_df)
        
    ord_sequence = list(np.linspace(0,NbFile-1, NbFile, dtype=int))
    dev_set_arg = []
    eval_set_arg = []
    
    Annot_ALL = {}
    for dataset_i in dataset_ID_tab:
        Annot_ALL[dataset_i] = [[],[]]
        
    #np.nonzero(np.sum(np.array(train_df.loc[:, labels_list]), axis=1))
        
    for i in tqdm(ord_sequence):
        _, dataset, _, _ = train_df['filename'][i].split(os.sep)
        if sum(train_df.loc[i, labels_list]) > 0:
            Annot_ALL[dataset][0].append(i)
        else:
            Annot_ALL[dataset][1].append(i)
            
            
    for dataset_i in tqdm(dataset_ID_tab):
        if dataset_i in dataset_for_eval:
            eval_set_arg = Annot_ALL[dataset_i][0] + Annot_ALL[dataset_i][1]
        else:
            for i in Annot_ALL[dataset_i][0]:
                dev_set_arg.append(i)
            list_random = Annot_ALL[dataset_i][1].copy()
            random.shuffle(list_random) 
            for i in list_random[:len(Annot_ALL[dataset_i][0])]:
                dev_set_arg.append(i)
            
    
    eval_set_arg = sorted(eval_set_arg)
    dev_set_arg = sorted(dev_set_arg)
    
    train_df_dev = train_df.iloc[dev_set_arg]
    train_df_eval = train_df.iloc[eval_set_arg]
    
    # Save dataframe as .csv
    train_df_dev.to_csv(os.path.join(base_path, 'annot_merged', 'DEVannotations.csv'), index = False, header=True)
    train_df_eval.to_csv(os.path.join(base_path, 'annot_merged', 'EVALannotations.csv'), index = False, header=True)
    
    
def main():
    
    import argparse

    # parse command-line args
    parser = argparse.ArgumentParser()
    
    parser.add_argument('base_path', type=str, help='Path to the "datasets" folder')
    args = parser.parse_args()
    base_path = args.base_path

    #list of all dataset used
    dataset_ID_tab = ['BlueFinLibrary_BallenyIslands2015', 'BlueFinLibrary_ElephantIsland2013Aural',  'BlueFinLibrary_ElephantIsland2014', "BlueFinLibrary_Greenwich64S2015", 'BlueFinLibrary_MaudRise2014', 'BlueFinLibrary_RossSea2014', 'BlueFinLibrary_casey2014', 'BlueFinLibrary_casey2017', 'BlueFinLibrary_kerguelen2005', 'BlueFinLibrary_kerguelen2014', 'BlueFinLibrary_kerguelen2015']
    #datasets used for dev
    dataset_for_dev = ['BlueFinLibrary_ElephantIsland2013Aural', 'BlueFinLibrary_ElephantIsland2014', "BlueFinLibrary_Greenwich64S2015", 'BlueFinLibrary_MaudRise2014', 'BlueFinLibrary_RossSea2014', 'BlueFinLibrary_casey2014', 'BlueFinLibrary_casey2017', 'BlueFinLibrary_kerguelen2005', 'BlueFinLibrary_kerguelen2014', 'BlueFinLibrary_kerguelen2015'] 
    #datasets used for eval
    dataset_for_eval = ['BlueFinLibrary_BallenyIslands2015']
    
    format_data_main(base_path, dataset_ID_tab, labels_list, chunk_duration, chunk_overlap, minimal_area_for_positive=0.5)
    split_dev_eval(base_path, dataset_ID_tab, labels_list, dataset_for_dev, dataset_for_eval)
    
#base_path = os.getcwd() 
#main(base_path)

if __name__ == "__main__":   
    main()

