import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from scipy import optimize

sns.set()
def read_data(folder_path):
    filenames = os.listdir(folder_path)
    for filename in filenames:
        if(filename.endswith('.tbl')):
            first_file = filename
            break

    df_data = ascii.read(folder_path + first_file).to_pandas()

    for filename in filenames:
        if(filename.endswith('.tbl') and not filename == first_file):
            data = ascii.read(folder_path + filename).to_pandas()
            df_data = df_data.append(data)

    return df_data

def plot_data(data_x, data_y, label_x='Time', label_y='Flux', title=''):
    plt.figure(1,dpi=300)
    sns.lineplot(data_x, data_y)
    plt.xlabel('Time')
    plt.ylabel('Flux')
    plt.title(title)
    plt.show()

def remove_noise(df,data):
    df['MEDIAN'] = data.rolling(10).median()

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}
