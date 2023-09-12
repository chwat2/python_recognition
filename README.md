### Data

Download files, unpack and place in data/temp
https://zenodo.org/record/3384388/files/-6_dB_fan.zip?download=1

### Python - install dependencies

    python -m venv myenv
    ### on windows
    myenv\Scripts\activate.bat  
    ###
    pip install -r requirements.txt
    python3 main.py --epochs 16 --n_mels 64 --frame 5 --learning-rate 0.001 --batch-size 512 --n_fft 4096 --hop_length 512

##### For linux users

please note that this runs on windows - for linux - line 30 in features_extraction.py needs to be made os agnostic

### TODO

1) fix linux-style path - line 30 in features_extraction.py
2) automatic download and unpack of the zip
3) check if model exists before training
4) requiremetns.txt i osobne srodowisko 
