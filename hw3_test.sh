#!/bin/bash
wget https://github.com/NTU-speech-lab/hw3-Leng-Kai/releases/download/0.0.0/my_model_best.pth
python3 hw3_test.py $1 $2
