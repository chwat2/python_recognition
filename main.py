import os
from utils import sound_tools, utils
from dataExploartion import dataExploartion as de

def main():
    print("init start")
    paths = de.initialize()
    data_path = paths[0]
    if not de.checkFiles(data_path):
        raise RuntimeError("upload files to proper place")
    print("init end")
    normal_signal_file = os.path.join(data_path, 'fan', 'id_00', 'normal', '00000100.wav')
    abnormal_signal_file = os.path.join(data_path, 'fan', 'id_00', 'abnormal', '00000100.wav')

    frames = 5
    power = 2.0
    initialData = de.initialCheck(data_path,frames,power,(normal_signal_file,abnormal_signal_file))
    #TODO
    sr = 16000
    print(f'The signals have a {initialData[0].shape} shape. At {sr} Hz, these are {initialData[0].shape[0]/sr:.0f}s signals')
    n_fft = 4096
    hop_length = 512
    n_mels = 64

    de.drawSTFTPlot(initialData,n_fft,hop_length)
    de.drawSectrogram(initialData,n_fft,hop_length,normal_signal_file)
    de.drawLogSpectrogram(initialData,n_fft,hop_length,(normal_signal_file,abnormal_signal_file))
    de.drawMelSpectrogram(initialData,n_fft,hop_length,n_mels)

if __name__ == "__main__":
    print("start")
    try:
        main()
    except Exception as exc:
        print(f"Exception: {exc}")