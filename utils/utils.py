import hashlib
import matplotlib.pyplot as mplot
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
from tqdm import tqdm

def md5(fname):
    filesize = os.stat(fname).st_size
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in tqdm(iter(lambda: f.read(4096), b""), total=filesize/4096):
            hash_md5.update(chunk)
            
    return hash_md5.hexdigest()

def build_files_list(root_dir, abnormal_dir='abnormal', normal_dir='normal'):
    normal_files = []
    abnormal_files = []
    
    for root, dirs, files in os.walk(top = os.path.join(root_dir)):
        for name in files:
            current_dir_type = root.split('/')[-1]
            if current_dir_type == abnormal_dir:
                abnormal_files.append(os.path.join(root, name))
            if current_dir_type == normal_dir:
                normal_files.append(os.path.join(root, name))
    print(normal_files)      
    return normal_files, abnormal_files


def generate_files_list(root_dir, abnormal_dir='abnormal', normal_dir='normal'):
    normal_files = []
    abnormal_files = []
    
    for root, dirs, files in os.walk(top = os.path.join(root_dir)):
        for name in files:
            current_dir_type = root.split('/')[-1]
            if current_dir_type == abnormal_dir:
                abnormal_files.append(os.path.join(root, name))
            if current_dir_type == normal_dir:
                normal_files.append(os.path.join(root, name))

    random.shuffle(normal_files)

    test_files = np.concatenate((normal_files[:len(abnormal_files)], abnormal_files), axis=0)
    test_labels = np.concatenate((np.zeros(len(abnormal_files)), np.ones(len(abnormal_files))), axis=0)
    
    train_files = normal_files[len(abnormal_files):]
    train_labels = np.zeros(len(train_files))
    
    return train_files, train_labels, test_files, test_labels


def generate_error_types(df, ground_truth_col='Ground Truth', prediction_col='Prediction', normal_label=0.0, anomaly_label=1.0):
    df['TP'] = 0
    df['TN'] = 0
    df['FP'] = 0
    df['FN'] = 0
    df.loc[(df[ground_truth_col] == df[prediction_col]) & (df[ground_truth_col] == normal_label), 'TP'] = 1
    df.loc[(df[ground_truth_col] == df[prediction_col]) & (df[ground_truth_col] == anomaly_label), 'TN'] = 1
    df.loc[(df[ground_truth_col] != df[prediction_col]) & (df[ground_truth_col] == normal_label), 'FP'] = 1
    df.loc[(df[ground_truth_col] != df[prediction_col]) & (df[ground_truth_col] == anomaly_label), 'FN'] = 1
    
    return df

def print_confusion_matrix(confusion_matrix, class_names, figsize = (4,3), fontsize=14):
    df_cm = pd.DataFrame(confusion_matrix, 
                        index=class_names, 
                        columns=class_names)
    
    fig = mplot.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 16}, cmap='cividis')
    except ValueError:
        raise ValueError("input ints")
    
    # Figure customization:
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    mplot.ylabel('Wynik rzeczywisty', fontsize=16)
    mplot.xlabel('Wynik prognozowany', fontsize=16)
    
    return fig