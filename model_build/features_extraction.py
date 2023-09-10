import os
import sys
import random
import time
import pickle
import pandas
import numpy as np
import librosa
import matplotlib.pyplot as mplot
import seaborn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tqdm import tqdm
from utils import sound_tools, utils
import tensorflow as tf

#import sagemaker
#import boto3
#from sagemaker.tensorflow import TensorFlow


def build_files_list(root_dir, abnormal_dir='abnormal', normal_dir='normal'):
    normal_files = []
    abnormal_files = []
    for root, dirs, files in os.walk(top = os.path.join(root_dir)):
        for name in files:
            #TODO - zlapac blad
            ##depends on os 
            current_dir_type = root.split('\\')[-1]  #windows style
          #  current_dir_type = root.split('/')[-1]  #linux style
            if current_dir_type == abnormal_dir:
                abnormal_files.append(os.path.join(root, name))
            if current_dir_type == normal_dir:
                normal_files.append(os.path.join(root, name))
    return normal_files, abnormal_files

    
def build_data_set(root_dir,PROCESSED_DATA):
    normal_files, abnormal_files = build_files_list(root_dir=root_dir)
    X = np.concatenate((normal_files, abnormal_files), axis=0)
    y = np.concatenate((np.zeros(len(normal_files)), np.ones(len(abnormal_files))), axis=0)

    train_files, test_files, train_labels, test_labels = train_test_split(X, y,
                                                                        train_size=0.85,
                                                                        random_state=508,
                                                                        shuffle=True,
                                                                        stratify=y
                                                                        )
    dataset = dict({
        'train_files': train_files,
        'test_files': test_files,
        'train_labels': train_labels,
        'test_labels': test_labels
    })

    for key, values in dataset.items():
        fname = os.path.join(PROCESSED_DATA, key + '.txt')
        with open(fname, 'w') as f:
            for item in values:
                f.write(str(item))
                f.write('\n')

    train_files = [f for f in train_files if f not in abnormal_files]
    train_labels = np.zeros(len(train_files))
    return (train_files,test_files,train_labels, test_labels)

def extract_signal_features(signal, sr, n_mels=64, frames=5, n_fft=1024, hop_length=512):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    features_vector_size = log_mel_spectrogram.shape[1] - frames + 1
    dims = frames * n_mels
    if features_vector_size < 1:
        return np.empty((0, dims), np.float32)
    
    features = np.zeros((features_vector_size, dims), np.float32)
    for t in range(frames):
        features[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t:t + features_vector_size].T
    return features

def generate_dataset(files_list, n_mels=64, frames=5, n_fft=1024, hop_length=512):
    dims = n_mels * frames
    
    for index in tqdm(range(len(files_list)), desc='Extracting features'):
        signal, sr = sound_tools.load_sound_file(files_list[index])
        
        features = extract_signal_features(
            signal, 
            sr, 
            n_mels=n_mels, 
            frames=frames, 
            n_fft=n_fft, 
            hop_length=hop_length
        )
        
        if index == 0:
            dataset = np.zeros((features.shape[0] * len(files_list), dims), np.float32)
            
        dataset[features.shape[0] * index : features.shape[0] * (index + 1), :] = features
    return dataset