import os
import random
import pickle
import pandas
import seaborn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as mplot
from utils import sound_tools, utils
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

def plot_curves(FP, FN, nb_samples, threshold_min, threshold_max, threshold_step):
    fig = mplot.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    min_FN = np.argmin(FN)
    min_FP = np.where(FP == np.min(FP))[0][-1]
    plot_top = max(FP + FN) + 1
    major_ticks = np.arange(threshold_min, threshold_max, 1.0 * threshold_step)
    minor_ticks = np.arange(threshold_min, threshold_max, 0.2 * threshold_step)
    ax.set_xticks(major_ticks);
    ax.set_xticks(minor_ticks, minor=True);
    ax.grid(which='minor', alpha=0.5)
    ax.grid(which='major', alpha=1.0, linewidth=1.0)
    
    mplot.plot(np.arange(threshold_min, threshold_max + threshold_step, threshold_step), FP, label='Fałszywie pozytywne', color='tab:red')
    mplot.plot(np.arange(threshold_min, threshold_max + threshold_step, threshold_step), FN, label='Fałszywie negatywne', color='tab:green')

    mplot.xlabel('Przedział błędu rekonstrukcji [%]', fontsize=16)
    mplot.ylabel('Liczba próbek', fontsize=16)
    mplot.legend()

def evaluate_model(paths,test_files,train_labels, test_labels,frames,power,n_fft,hop_length,n_mels):
    DATA,RAW_DATA,PROCESSED_DATA = paths
    loaded_model = tf.keras.models.load_model(os.path.join(DATA, 'model/1'))
    y_true = test_labels
    reconstruction_errors = []

    for index, eval_filename in tqdm(enumerate(test_files), total=len(test_files)):
        signal, sr = sound_tools.load_sound_file(eval_filename)
        # Ekstrakcja cech z sygnału:
        eval_features = sound_tools.extract_signal_features(
            signal, 
            sr, 
            n_mels=n_mels, 
            frames=frames, 
            n_fft=n_fft, 
            hop_length=hop_length
        )
        # Pobieranie predykcji jako wyniku działania autoenkodera:
        prediction = loaded_model.predict(eval_features)
        # Obliczanie błędu rekonstrukcji: ['predictions']
        mse = np.mean(np.mean(np.square(eval_features - prediction), axis=1))
        reconstruction_errors.append(mse)

    data = np.column_stack((range(len(reconstruction_errors)), reconstruction_errors))
    bin_width = 0.25
    bins = np.arange(min(reconstruction_errors), max(reconstruction_errors) + bin_width, bin_width)
    fig = mplot.figure(figsize=(12,4))
    mplot.hist(data[y_true==0][:,1], bins=bins, color='tab:blue', alpha=0.6, label='Sygnały oznaczone jako normalne', edgecolor='#FFFFFF')
    mplot.hist(data[y_true==1][:,1], bins=bins, color='tab:red', alpha=0.6, label='Sygnały oznaczone jako anomelie', edgecolor='#FFFFFF')
    mplot.xlabel("Błąd rekonstrukcji")
    mplot.ylabel("Liczba próbek")
    mplot.title('Rozkład błędów rekonstrukcji', fontsize=16)
    mplot.legend()
    mplot.show()

    threshold_min = 5.0
    threshold_max = 10.0
    threshold_step = 0.50

    normal_x, normal_y = data[y_true==0][:,0], data[y_true==0][:,1]
    abnormal_x, abnormal_y = data[y_true==1][:,0], data[y_true==1][:,1]
    x = np.concatenate((normal_x, abnormal_x))

    fig, ax = mplot.subplots(figsize=(24,8))
    mplot.scatter(normal_x, normal_y, s=15, color='tab:green', alpha=0.3, label='Sygnały oznaczone jako normalne')
    mplot.scatter(abnormal_x, abnormal_y, s=15, color='tab:red', alpha=0.3,   label='Sygnały oznaczone jako anomelie')
    mplot.fill_between(x, threshold_min, threshold_max, alpha=0.1, color='tab:orange', label='Zakres')
    mplot.hlines([threshold_min, threshold_max], x.min(), x.max(), linewidth=0.5, alpha=0.8, color='tab:olive')
    mplot.legend(loc='upper left')
    mplot.title('Wykres pokazujący przedział ', fontsize=16)
    mplot.xlabel('Numer próbki')
    mplot.ylabel('Błąd rekonstrukcji')
    mplot.xlim([0,850])
    mplot.show()

    thresholds = np.arange(threshold_min, threshold_max + threshold_step, threshold_step)

    df = pandas.DataFrame(columns=['Signal', 'Ground Truth', 'Prediction', 'Reconstruction Error'])
    df['Signal'] = test_files
    df['Ground Truth'] = test_labels
    df['Reconstruction Error'] = reconstruction_errors

    FN = []
    FP = []
    for th in thresholds:
        df.loc[df['Reconstruction Error'] <= th, 'Prediction'] = 0.0
        df.loc[df['Reconstruction Error'] > th, 'Prediction'] = 1.0
        df = utils.generate_error_types(df)
        FN.append(df['FN'].sum())
        FP.append(df['FP'].sum())
    
    plot_curves(FP, FN, nb_samples=df.shape[0], threshold_min=threshold_min, threshold_max=threshold_max, threshold_step=threshold_step)
    return df

def analyze_results(paths,df):
    DATA,RAW_DATA,PROCESSED_DATA = paths
    th = 6.5
    df.loc[df['Reconstruction Error'] <= th, 'Prediction'] = 0.0
    df.loc[df['Reconstruction Error'] > th, 'Prediction'] = 1.0
    df['Prediction'] = df['Prediction'].astype(np.float32)
    df = utils.generate_error_types(df)
    tp = df['TP'].sum()
    tn = df['TN'].sum()
    fn = df['FN'].sum()
    fp = df['FP'].sum()

    df['Ground Truth'] = 1 - df['Ground Truth']
    df['Prediction'] = 1 - df['Prediction']

    utils.print_confusion_matrix(confusion_matrix(df['Ground Truth'], df['Prediction']), class_names=['anomalia', 'dobra\n próbka']);
    
    df.to_csv(os.path.join(PROCESSED_DATA, 'results_autoencoder.csv'), index=False)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    miss_rate = fn / (tp + fn)
    print(f"""Podstawowe wskaźniki kwalifikacji wyników:
    - precyzja: {precision*100:.1f}%
    - czułość: {recall*100:.1f}%
    - dokładność: {accuracy*100:.1f}%
    - wynik F1: {f1_score*100:.1f}%
    - fałszywie ujemne: {miss_rate*100:.1f}%""")