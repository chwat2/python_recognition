#TODO

1) Download files, unpack and place in data/temp
2) run
python3 main.py --epochs 16 --n_mels 64 --frame 5 --learning-rate 0.01 --batch-size 128 --n_fft 4096 --hop_length 512

please note that this runs on windows - for linux - line 30 in model_build.py needs to be made os agnostic 