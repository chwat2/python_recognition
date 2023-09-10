import os
import random
import sys
import numpy as np
import hashlib
import matplotlib.pyplot as mplot
import librosa
import librosa.display
import IPython.display as ipython

from utils import sound_tools, utils

def checkFiles (data_path):
    first_file = os.path.join(data_path, 'fan', 'id_00', 'normal', '00000000.wav')
    return os.path.exists(first_file)

def initialCheck (data_path, frames, power, files):
    normal_signal_file, abnormal_signal_file = files
    normal_signal, sr = sound_tools.load_sound_file(normal_signal_file)
    abnormal_signal, sr = sound_tools.load_sound_file(abnormal_signal_file)
    return (normal_signal,abnormal_signal,sr)

def draw_STFTPlot (signals,n_fft,hop_length):
    blue = '#1520A6' #azure
    red = '#ff1a1a'
    normal_signal,abnormal_signal,sr = signals

    D_normal = np.abs(librosa.stft(normal_signal[:n_fft], n_fft=n_fft, hop_length=n_fft + 1))
    D_abnormal = np.abs(librosa.stft(abnormal_signal[:n_fft], n_fft=n_fft, hop_length=n_fft + 1))

    fig = mplot.figure(figsize=(12, 6))
    mplot.plot(D_normal, color=blue, alpha=0.9, label='Sygnał oznaczony jako normalny');
    mplot.plot(D_abnormal, color=red, alpha=0.9, label='Sygnał oznaczony jako anomalia');
    mplot.title('Transformata Fouriera dla pierwszych 256ms')
    mplot.xlabel('Częstotliwość [Hz]')
    mplot.ylabel('Amplituda')
    mplot.legend()
    mplot.xlim(0, 200);
    mplot.show()

def draw_sectrogram(signals,n_fft,hop_length,normal_signal_file):
    normal_signal,abnormal_signal,sr = signals

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

    D_normal = np.abs(librosa.stft(normal_signal, n_fft=n_fft, hop_length=hop_length))
    D_abnormal = np.abs(librosa.stft(abnormal_signal, n_fft=n_fft, hop_length=hop_length))

    fig2 = mplot.figure(figsize=(24, 6))
    mplot.subplot(1, 2, 1)
    librosa.display.specshow(D_normal, sr=sr, x_axis='time', y_axis='linear', cmap='viridis');
    mplot.title('Machine #id_00 - sygnał normalny')
    mplot.xlabel('Time (s)')
    mplot.ylabel('Frequency (Hz)')
    mplot.colorbar();

    mplot.subplot(1, 2, 2)
    librosa.display.specshow(D_abnormal, sr=sr, x_axis='time', y_axis='linear', cmap='viridis');
    mplot.title('Machine #id_00 - anomalia')
    mplot.xlabel('Time (s)')
    mplot.ylabel('Frequency (Hz)')
    mplot.colorbar();
    mplot.show()

def draw_log_spectrogram (signals,n_fft,hop_length,signal_files):
    normal_signal,abnormal_signal,sr = signals
    normal_signal_file,abnormal_signal_file = signal_files
    dB_normal = sound_tools.get_magnitude_scale(normal_signal_file, n_fft=n_fft, hop_length=hop_length)
    dB_abnormal = sound_tools.get_magnitude_scale(abnormal_signal_file, n_fft=n_fft, hop_length=hop_length)

    fig = mplot.figure(figsize=(24, 6))

    mplot.subplot(1, 2, 1)
    librosa.display.specshow(dB_normal, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
    mplot.title('sygnał normalny')
    mplot.colorbar(format="%+2.f dB")
    mplot.xlabel('Time (s)')
    mplot.ylabel('Frequency (Hz)')

    mplot.subplot(1, 2, 2)
    librosa.display.specshow(dB_abnormal, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
    mplot.title('anomalia')
    mplot.ylabel('Częstotliwość [Hz]')
    mplot.colorbar(format="%+2.f dB")
    mplot.xlabel('Czas [s]')
    mplot.ylabel('Częstotliwość [Hz]')

    mplot.show()

def draw_Mel_spectrogram (signals,n_fft,hop_length,n_mels):
    normal_signal,abnormal_signal,sr = signals
    normal_mel = librosa.feature.melspectrogram(y=normal_signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    normal_S_DB = librosa.power_to_db(normal_mel, ref=np.max)
    abnormal_mel = librosa.feature.melspectrogram(y=abnormal_signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    abnormal_S_DB = librosa.power_to_db(abnormal_mel, ref=np.max)

    fig = mplot.figure(figsize=(24, 6))
    mplot.subplot(1, 2, 1)
    librosa.display.specshow(normal_S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='viridis');
    mplot.title('Machine #id_00 - sygnał normalny')
    mplot.xlabel('Time (s)')
    mplot.ylabel('Frequency (Hz)')
    mplot.colorbar(format='%+2.0f dB');

    mplot.subplot(1, 2, 2)
    librosa.display.specshow(abnormal_S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='viridis');
    mplot.title('Machine #id_00 - anomalia')
    mplot.xlabel('Time (s)')
    mplot.ylabel('Frequency (Hz)')
    mplot.colorbar(format='%+2.0f dB');

    mplot.show()