import argparse
import os
import pickle
import random

import matplotlib.pyplot as mplot
import numpy as np
import tensorflow as tf

from data_exploartion import data_exploartion as de
from model_build import features_extraction as fe
from model_build import model_evaluation as me
from model_build import model_training as mt


def explore_data(paths, frames, power, n_fft, hop_length, n_mels):
    data_path = paths[0]
    if not de.check_files(data_path):
        raise RuntimeError("sprawdz w readme jakie dane i gdzie pobrac")
    normal_signal_file = os.path.join(data_path, 'fan', 'id_00', 'normal', '00000100.wav')
    abnormal_signal_file = os.path.join(data_path, 'fan', 'id_00', 'abnormal', '00000100.wav')
    initialData = de.initialCheck(data_path, frames, power, (normal_signal_file, abnormal_signal_file))
    # TODO
    sr = 16000
    print(
        f'The signals have a {initialData[0].shape} shape. At {sr} Hz, these are {initialData[0].shape[0] / sr:.0f}s signals')
    # TODO
    de.draw_STFTPlot(initialData, n_fft, hop_length)
    de.draw_sectrogram(initialData, n_fft, hop_length, normal_signal_file)
    de.draw_log_spectrogram(initialData, n_fft, hop_length, (normal_signal_file, abnormal_signal_file))
    de.draw_Mel_spectrogram(initialData, n_fft, hop_length, n_mels)


def build_model(paths, frames, power, n_fft, hop_length, n_mels):
    DATA, RAW_DATA, PROCESSED_DATA = paths
    train_files, test_files, train_labels, test_labels = fe.build_data_set(os.path.join(DATA, 'fan'), PROCESSED_DATA)
    print("liczba plików w zbiorze treningowym:")
    print(len(test_files))
    print("liczba plików w zbiorze testowym:")
    print(len(train_files))

    train_data_location = os.path.join(DATA, 'autoenkoder_data.pkl')
    if os.path.exists(train_data_location):
        print('dane już istnieją...')
        with open(train_data_location, 'rb') as f:
            train_data = pickle.load(f)  # nieużywane
    else:
        train_data = fe.generate_dataset(train_files, n_mels=n_mels, frames=frames, n_fft=n_fft, hop_length=hop_length)
        with open(os.path.join(DATA, 'autoenkoder_data.pkl'), 'wb') as f:
            pickle.dump(train_data, f)
    return train_files, test_files, train_labels, test_labels


def train_model(paths, n_mels, frame, lr, batch_size, epochs):
    training_dir, RAW_DATA, PROCESSED_DATA = paths
    mt.train(training_dir, os.path.join(training_dir, 'model/1'), n_mels, frame, lr, batch_size, epochs)


def initialize():
    print("de init start")
    random.seed(508)
    tf.random.set_seed(508)
    np.random.seed(508)
    mplot.style.use('seaborn')
    # Paths
    DATA = os.path.join('data', 'temp')
    RAW_DATA = os.path.join('data', 'raw')
    PROCESSED_DATA = os.path.join('data', 'processed')
    print(DATA)
    if not os.path.exists(DATA):
        print('Data directory does not exist, creating them.')
        os.makedirs(DATA, exist_ok=True)
        os.makedirs(RAW_DATA, exist_ok=True)
        os.makedirs(PROCESSED_DATA, exist_ok=True)
    return (DATA, RAW_DATA, PROCESSED_DATA)


def evaluate_model(paths, test_files, train_labels, test_labels, frame, power, n_fft, hop_length, n_mels):
    df = me.evaluate_model(paths, test_files, train_labels, test_labels, frame, power, n_fft, hop_length, n_mels)
    me.analyze_results(paths, df)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--n_mels', type=int, default=64)
    parser.add_argument('--frame', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--n_fft', type=int, default=4096)
    parser.add_argument('--hop_length', type=int, default=512)
    parser.add_argument('--power', type=int, default=2)
    args, _ = parser.parse_known_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    epochs = args.epochs
    n_mels = args.n_mels
    frame = args.frame
    lr = args.learning_rate
    batch_size = args.batch_size
    n_fft = args.n_fft
    hop_length = args.hop_length
    power = args.power

    try:
        paths = initialize()
        explore_data(paths, frame, power, n_fft, hop_length, n_mels)
        train_files, test_files, train_labels, test_labels = build_model(paths, frame, power, n_fft, hop_length, n_mels)
        train_model(paths, n_mels, frame, lr, batch_size, epochs)
        evaluate_model(paths, test_files, train_labels, test_labels, frame, power, n_fft, hop_length, n_mels)
    except Exception as exc:
        print(f"Exception: {exc}")
