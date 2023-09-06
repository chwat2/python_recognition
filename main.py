from utils import sound_tools, utils
from dataExploartion import dataExploartion as de

def main():
    print("init start")
    paths = de.initialize()
    if not de.checkFiles(paths[0]):
        raise RuntimeError("upload files to proper place")
    print("init end")
    n_mels = 64
    frames = 5
    power = 2.0
    testSingals = de.initialCheck(paths[0],n_mels,frames,power)
    #TODO
    sr = 16000
    print(f'The signals have a {testSingals[0].shape} shape. At {sr} Hz, these are {testSingals[0].shape[0]/sr:.0f}s signals')
    n_fft = 4096
    hop_length = 512
    de.drawSTFTPlot(testSingals,4096,512)

if __name__ == "__main__":
    print("start")
    try:
        main()
    except Exception as exc:
        print(f"Exception: {exc}")