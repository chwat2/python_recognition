import os
import random
import sys
import numpy 
import hashlib
import matplotlib.pyplot as mplot
import librosa
import librosa.display
import IPython.display as ipython

from utils import sound_tools, utils

def initialize ():
    print("de init start")
    random.seed(508)
    numpy.random.seed(508)
    mplot.style.use('seaborn')
    # Paths
    DATA           = os.path.join('data', 'temp')
    RAW_DATA       = os.path.join('data', 'raw')
    PROCESSED_DATA = os.path.join('data', 'processed')
    print(DATA)
    if not os.path.exists(DATA):
        print('Data directory does not exist, creating them.')
        os.makedirs(DATA, exist_ok=True)
        os.makedirs(RAW_DATA, exist_ok=True)
        os.makedirs(PROCESSED_DATA, exist_ok=True)
    return (DATA,RAW_DATA,PROCESSED_DATA)

def checkFiles (data_path):
    first_file = os.path.join(data_path, 'fan', 'id_00', 'normal', '00000000.wav')
    return os.path.exists(first_file)

def initialCheck (data_path, n_mels, frames, power):
    normal_signal_file = os.path.join(data_path, 'fan', 'id_00', 'normal', '00000100.wav')
    abnormal_signal_file = os.path.join(data_path, 'fan', 'id_00', 'abnormal', '00000100.wav')
    normal_signal, sr = sound_tools.load_sound_file(normal_signal_file)
    abnormal_signal, sr = sound_tools.load_sound_file(abnormal_signal_file)
    return (normal_signal,abnormal_signal)

def drawSTFTPlot (signals,n_fft,hop_length):
    blue = '#1520A6' #azure
    red = '#ff1a1a'
    normal_signal,abnormal_signal = signals

    D_normal = numpy.abs(librosa.stft(normal_signal[:n_fft], n_fft=n_fft, hop_length=n_fft + 1))
    D_abnormal = numpy.abs(librosa.stft(abnormal_signal[:n_fft], n_fft=n_fft, hop_length=n_fft + 1))

    fig = mplot.figure(figsize=(12, 6))
    mplot.plot(D_normal, color=blue, alpha=0.9, label='Sygnał oznaczony jako normalny');
    mplot.plot(D_abnormal, color=red, alpha=0.9, label='Sygnał oznaczony jako anomalia');
    mplot.title('Transformata Fouriera dla pierwszych 256ms')
    mplot.xlabel('Częstotliwość [Hz]')
    mplot.ylabel('Amplituda')
    mplot.legend()
    mplot.xlim(0, 200);
    mplot.show()

def DrawMelSectrogram(signals):
    n_fft = 4096
    D_normal = np.abs(librosa.stft(normal_signal[:10*n_fft], n_fft=n_fft, hop_length=hop_length))
    dB_normal = sound_tools.get_magnitude_scale(normal_signal_file)

    fig = mplot.figure(figsize=(12, 6))
    librosa.display.specshow(D_normal, sr=sr, x_axis='time', y_axis='linear', cmap='viridis');
    mplot.title('Sygnał oznaczony jako normalny\nSFTP dla pierwszych 2560ms')
    mplot.ylim(0, 500)
    mplot.xlabel('Czas [s]')
    mplot.ylabel('Częstotliwość [Hz]')
    mplot.colorbar()
    mplot.show()