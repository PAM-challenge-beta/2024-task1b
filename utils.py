# -*- coding: utf-8 -*-
"""
Created on Wed May 29 02:42:27 2024

@author: gabri
"""

from scipy.signal import welch, butter, sosfilt
import numpy as np
import torch 

def gen_spectro(data, sample_rate, window_size, overlap, Nfft):
    sos = butter(10, 10, 'hp', fs=sample_rate, output='sos')
    data = sosfilt(sos, data)
    
    data = (data - np.mean(data)) / (np.std(data)  + 1e-16)     
    
    Noverlap = int(window_size * overlap / 100)
    win = np.hamming(window_size)
    if Nfft < (window_size):
        scale_psd = 2.0 * sample_rate
    else:
        scale_psd = 2.0 / ((win * win).sum())
           
    Nbech = np.size(data)
    Noffset = window_size - Noverlap
    Nbwin = int((Nbech - window_size) / Noffset)
    Freq = np.fft.rfftfreq(Nfft, d = 1 / sample_rate)
    Sxx = np.zeros([np.size(Freq), Nbwin])
    for idwin in range(Nbwin):
        if Nfft < (window_size):
            x_win = data[idwin * Noffset:idwin * Noffset + window_size]
            _, Sxx[:, idwin] = welch(x_win, fs=sample_rate, window='hamming', nperseg=Nfft,
                                            noverlap=int(Nfft / 2) , scaling='sectrum')
        else:
            x_win = data[idwin * Noffset:idwin * Noffset + window_size] * win
            Sxx[:, idwin] = (np.abs(np.fft.rfft(x_win, n=Nfft)) ** 2)
        Sxx[:, idwin] *= scale_psd

    return Sxx, Freq

def normalize(spectro, parameters):
    dynamic_min = parameters['dynamic_min']
    dynamic_max = parameters['dynamic_max']
    
    #Normalize Lvl
    spectro = 10*np.log10(spectro+1e-15)
    spectro[spectro < dynamic_min] = dynamic_min
    spectro[spectro >  dynamic_max] = dynamic_max
    spectro = (spectro - dynamic_min) /  (dynamic_max-dynamic_min)
    
    #plt.pcolor(spectro)
    #plt.savefig('test'+str(np.random.randint(10))+'.png')
    X, Y = np.shape(spectro)
    
    spectro = torch.from_numpy(spectro.reshape((X, Y)))
    spectro = spectro[None,:,:]
    return spectro  

def comp_overlap(min1, max1, min2, max2):
    return max(0, min(max1, max2) - max(min1, min2))
