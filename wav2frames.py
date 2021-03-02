import numpy as np 
import matplotlib.pyplot as plt 
import pickle
import os
import librosa
import librosa.display
import scipy as sci
import argparse

def get_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--filename_in', type=str)
	parser.add_argument('--filename_out', type=str)
	return parser.parse_args()

args = get_arguments()

len_window = 4096 #Specified length of analysis window
hop_length_ = 1024 #Specified percentage hop length between windows

filename_in = args.filename_in
filename_out = args.filename_out
data_path = os.path.join(os.getcwd(),filename_in)
y, sr = librosa.load(data_path, sr=44100)

D = librosa.stft(y,n_fft=4096, window='hann')
print(D.shape)
temp = D[:,:]
phase = np.angle(temp)
temp = np.abs(temp)
temp = temp / (temp.max(axis=0)+0.000000001)
print(temp.max(axis=0))
temp = np.transpose(temp)
phase = np.transpose(phase)
print(np.shape(temp))
output = temp[~np.all(temp == 0, axis=1)]
out_phase = phase[~np.all(temp == 0, axis=1)]
print(np.shape(output))
np.save(filename_out+'.npy',output)
np.save(filename_out+'_phase.npy',out_phase)
