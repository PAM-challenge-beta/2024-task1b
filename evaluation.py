# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:31:25 2024

@author: gabri
"""

import pandas as pd
from parameters import labels_list

def compute_metrics(ground_truth, detections, buffer=10):
    TP = 0
    FP = 0
    FN = 0

    # Group data by filename for easier processing
    grouped_ground_truth = ground_truth.groupby('filename')
    grouped_detections = detections.groupby('filename')

    # Algorithm explanation: The true positive count will only be incremented 
    # once per detection timestamp. For instance if multiple detection timestamps 
    # aligns with only one ground truth, the true positive count will be incremented 
    # only once. Similarly, if one detection timestamp aligns with multiple ground truths, 
    # the true positive count will be incremented for each ground truth.
    for filename, tru_group in grouped_ground_truth:
        if filename in grouped_detections.groups:
            # Get corresponding detection
            det_group = grouped_detections.get_group(filename)

            # For each ground truth, check if it is a TP or FN 
            for tru_time in tru_group['timestamp']:
                if any((tru_time - buffer <= det_group['timestamp']) & (det_group['timestamp'] <= tru_time + buffer)):
                    TP += 1
                else:
                    FN += 1
        else:
            # All annotations in files without detections are false negatives
            FN += len(tru_group)

    # Now for the FP we have to loop through the detections and check if they do not match any ground truth
    # Note that we are treating each detection independently and separetly from TP and FN. This is to ensure that any detection that matches
    # a groudn truth is not counted as a FP
    for filename, det_group in grouped_detections:
        if filename in grouped_ground_truth.groups:
            # Get corresponding annotations
            ann_group = grouped_ground_truth.get_group(filename)

            # For each detection, check if it is a FP
            for det_time in det_group['timestamp']:
                # Check if detection is not within the buffer of any ground truth
                if not any((ann_group['timestamp'] - buffer <= det_time) & (det_time <= ann_group['timestamp'] + buffer)):
                    FP += 1
        else:
            # All detections in files without groudn truths are false positives
            FP += len(det_group)

    # Precision
    if TP + FP == 0:
        precision = 0  # To handle the case where no positives are predicted
    else:
        precision = TP / (TP + FP)

    # Recall
    if TP + FN == 0:
        recall = 0  # To handle the case where there are no true positives and false negatives
    else:
        recall = TP / (TP + FN)

    # F1 Score
    if precision + recall == 0:
        f1_score = 0  # To handle the case where precision and recall are both zero
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
        
    return precision, recall, f1_score


def main():
    
    import argparse

    # parse command-line args
    parser = argparse.ArgumentParser()
    
    parser.add_argument('df_gt_path', type=str, help='Path to the ground truth annotations CSV file.')
    parser.add_argument('df_det_path', type=str, help='Path to the detections CSV file.')
    
    args = parser.parse_args()
    df_gt_path = args.df_gt_path
    df_det_path = args.df_det_path
    
    df_det = pd.read_csv(df_det_path)
    df_groudtruth = pd.read_csv(df_gt_path)
    gt_groups = df_groudtruth.groupby('label')
    det_groups = df_det.groupby('label')
    p_mean, r_mean, f1_mean = 0, 0, 0
    for label in labels_list:
        if label not in list(det_groups.groups):
            print(label + ' not detected !')
            p,r,f1 = 0, 0, 0
            print(p,r,f1)
        else:
            gt = gt_groups.get_group(label)
            det = det_groups.get_group(label)
            p, r, f1 = compute_metrics(gt, det)
            print(label)
            print(p,r,f1)
        p_mean += p/len(labels_list)
        r_mean += r/len(labels_list)
        f1_mean += f1/len(labels_list)
        
    print('Evaluation by averaging all labels : ')
    print("Precision:", p_mean)
    print("Recall:", r_mean)
    print("F1 Score:", f1_mean)

#df_det_path = 'model_baseline_detections.csv'
#df_gt_path = 'annotations_eval_BlueFinLibrary_BallenyIslands2015.csv'
if __name__ == "__main__":   
    main()      