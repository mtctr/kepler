import os
import requests
import copy
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import subprocess

from astropy.io import ascii
from scipy import optimize
from collections import Counter
from itertools import groupby
from statistics import mode, median, mean, stdev
from tqdm import tqdm
from pwkit import pdm

""" Constants with the column names """
col = ["SAP_FLUX", "PDCSAP_FLUX"]
ecol = ["SAP_FLUX_ERR", "PDCSAP_FLUX_ERR"]
col2 = ["F", "FPDC"]  # Names for the modified columns.
ecol2 = ["EF", "EFPDC"]
""" Constants for paths"""
BASE_URL = "https://exoplanetarchive.ipac.caltech.edu"
BASE_PATH = "datasets/light-curves/"

sns.set()
def process_data(folder_path):
    """
    Read and process all the kepler data inside a folder based on the path provided

    folder_path: path of the folder containing the data filenames
    """
    filenames_tbl = get_filenames(folder_path,'tbl')
    filenames_fits = get_filenames(folder_path,'fits')
    for filename in filenames_fits:
        os.remove(folder_path+filename)
    for idx, filename in enumerate(filenames_tbl):
        data = ascii.read(folder_path + filename).to_pandas()
        data = data.interpolate(method='linear', columns=['TIME','SAP_FLUX','PDCSAP_FLUX','SAP_FLUX_ERR','PDCSAP_FLUX_ERR','CADENCENO'])        
        data = data[['TIME','SAP_FLUX','PDCSAP_FLUX','SAP_FLUX_ERR','PDCSAP_FLUX_ERR','CADENCENO']].dropna()
        data = normalize_data(data)

        remove_noise(data, data.PDCSAP_FLUX,'PDC_RAW_FILT')
        remove_noise(data, data.FPDC,'PDC_NORM_FILT')

        data = data.dropna()    
        data.to_csv(folder_path+filename.replace('.tbl','.csv'),index=False)
        os.remove(folder_path+filename)

def get_filenames(folder_path,extension):
    return list(f for f in os.listdir(folder_path) if f.endswith('.' + extension))

def normalize_data(data):
    r = copy.deepcopy(data)
    for c, ec, c2, ec2 in zip(col, ecol, col2, ecol2):
        medf = np.median(r[c])
        norm = r[c] / medf - 1
        enorm = r[ec] / medf
        r[c2] = norm
        r[ec2] = enorm
    return r


def plot_data(data_x, data_y, label_x="Time", label_y="Flux", title=""):
    plt.figure(1, dpi=300)
    sns.lineplot(data_x, data_y)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    plt.show()


def remove_noise(df, data, col_name="FILT"):
    """
    inputs:
    df: dataframe containing the data
    data: data to be adjusted
    field_name: name of the column to be added to the dataframe
    """
    df[col_name] = data.rolling(170).mean()

def get_signal_parameters(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(
        ff[np.argmax(Fyy[1:]) + 1]
    )  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2 * 2.0 ** 0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess, maxfev=7000)
    A, w, p, c = popt
    f = w / (2.0 * np.pi)
    fitfunc = lambda t: A * np.sin(w * t + p) + c
    return {
        "amp": A,
        "omega": w,
        "phase": p,
        "offset": c,
        "freq": f,
        "period": 1.0 / f,
        "fitfunc": fitfunc,
        "maxcov": np.max(pcov),
        "rawres": (guess, popt, pcov),
    }


def round_elements(list, n_places):
    return [round(elem, n_places) for elem in list]


def remove_outliers(list):
    mean = np.mean(list, axis=0)
    sd = np.std(list, axis=0)
    new_list = [x for x in list if (x > mean - 2 * sd)]
    new_list = [x for x in new_list if (x < mean + 2 * sd)]
    return new_list


def get_period(t,y,u,periods_list):
    periods = round_elements(periods_list, 3)
    _pdm = pdm.pdm(t,y,u,periods,10)
    period = _pdm.pmin
    return period


def remove_single_quotes(path):
    file_in = path
    file_out = path + "-ed"
    with open(r"" + file_in + "", "r") as infile, open(
        r"" + file_out + "", "w"
    ) as outfile:
        data = infile.read()
        data = data.replace("'", "")
        outfile.write(data)
    os.remove(file_in)
    os.rename(file_out, file_out.replace("-ed", ""))


def get_download_url(page_text):
    big = page_text[page_text.find("<big>") : page_text.find("</big>")]
    link = big[big.find("/cgi") : big.find('t"') + 1]
    return link


def download_files(kic):
    r = requests.post(
        BASE_URL + "/cgi-bin/IERDownload/nph-IERDownload",
        data={
            "id": kic,
            "inventory_mode": "id_single",
            "idtype": "source",
            "dataset": "kepler",
            "resultmode": "webpage",
        },
    )
    link = get_download_url(r.text)
    url = BASE_URL + link
    response = requests.get(url, stream=True)
    kic_str = str(kic)
    kic_file_name = str(kic) + ".bat"
    folder_path = BASE_PATH + kic_str + "/"
    path = BASE_PATH + kic_str + "/" + kic_file_name

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(path, "wb") as handle:
        for data in tqdm(response.iter_content()):
            handle.write(data)
    print(".bat downloaded")
    remove_single_quotes(path)

    print("downloading kepler files")
    os.chdir(r"" + folder_path + "")
    subprocess.call([kic_file_name])
    os.chdir("../../../")
    print("kepler files downloaded")
    return folder_path
    
