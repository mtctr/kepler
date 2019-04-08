import os
import pandas as pd
import seaborn as sns
import numpy as np
import copy
import matplotlib.pyplot as plt
from astropy.io import ascii
from scipy import optimize

''' Constants with the column names '''
col = ['SAP_FLUX','PDCSAP_FLUX']
ecol = ['SAP_FLUX_ERR','PDCSAP_FLUX_ERR']
col2 = ['F','FPDC']   # Names for the modified columns.
ecol2 = ['EF','EFPDC']

sns.set()
def read_data(folder_path):
    """
    Read and process all the kepler data inside a folder based on the path provided

    folder_path: path of the folder containing the data filenames

    returns:

    df_array: array with data frames for each quarter
    periods: array with the period calculated with raw data for each quarter
    periods_normalized: array with the period calculated with raw data for each quarter
    """
    periods = []
    df_array = []
    filenames = os.listdir(folder_path)
    for filename in filenames:
        if(filename.endswith('.tbl')):
            data = ascii.read(folder_path + filename).to_pandas()
            data = data[['TIME','SAP_FLUX','PDCSAP_FLUX','SAP_FLUX_ERR','PDCSAP_FLUX_ERR','CADENCENO']].dropna()
            data = normalize_data(data)

            remove_noise(data, data.PDCSAP_FLUX,'PDC_RAW_MEDIAN')
            remove_noise(data, data.FPDC,'PDC_NORM_MEDIAN')

            res = get_signal_parameters(data.dropna().TIME, data.dropna().PDC_RAW_MEDIAN)
            periods = np.append(periods, res["period"])

            df_array = np.append(df_array, data)

    return {"df_array": df_array, "periods": periods}

def normalize_data(data):
    r = copy.deepcopy(data)
    for c,ec,c2,ec2 in zip(col,ecol,col2,ecol2):
        medf = np.median(r[c])
        norm = r[c] / medf - 1
        enorm = r[ec] / medf
        r[c2] = norm
        r[ec2] = enorm
    return r

def plot_data(data_x, data_y, label_x='Time', label_y='Flux', title=''):
    plt.figure(1,dpi=300)
    sns.lineplot(data_x, data_y)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    plt.show()

def remove_noise(df,data,col_name='MEDIAN'):
    """
    inputs:
    df: dataframe containing the data
    data: data to be adjusted
    field_name: name of the column to be added to the dataframe
    """
    df[col_name] = data.rolling(10).median()

def get_signal_parameters(tt, yy):
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
