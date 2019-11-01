import utils
import time

def get_data_frames(kic):
    df_list = []
    filenames = utils.get_filenames(utils.BASE_PATH + str(kic), "csv")
    for filename in filenames:        
        data = utils.pd.read_csv(utils.BASE_PATH + str(kic) + "/" + filename)
        df_list.append(data)
    return df_list

def get_full_light_curve(kic):
    light_curve = utils.pd.DataFrame()
    filenames = utils.get_filenames(utils.BASE_PATH + str(kic), "csv")
    for filename in filenames:        
        data = utils.pd.read_csv(utils.BASE_PATH + str(kic) + "/" + filename)    
        light_curve = light_curve.append(data)
    return light_curve

def get_period(kic):
    frequencies = []
    df_list = []
    filenames = utils.get_filenames(utils.BASE_PATH + str(kic), "csv")
    if len(filenames) <= 1:
        return {"period": 0.0, "fap": 0.0, "theta": 0.0, "periods": []}
 
    for idx, filename in enumerate(filenames):
        if (idx > 2):
            data = utils.pd.read_csv(utils.BASE_PATH + str(kic) + "/" + filename)
            try:
                freq = utils.get_freq_LS(data.TIME.to_numpy(),data.PDCSAP_FLUX.to_numpy(),data.EFPDC.to_numpy())
                frequencies.append(freq)
            except Exception as e:
                print(e)
                print(idx)
                print(kic)

            df_list.append(data)
    
    df = utils.pd.DataFrame()
    for _df in df_list:
        df = df.append(_df)        
          
    t = df.TIME.to_numpy()
    y = df.FPDC.to_numpy()
    dy = df.EFPDC.to_numpy()
    
    period1 = utils.get_period(t, y, dy, frequencies)
    period2 = utils.get_period(t, y, dy)    
    
    periods = [period1, period2]
    nbins = 3
    
    if period2 < 0.09 or period2 > 100:
        period = period1
        theta = None
    else:
        try:  
            period, theta = utils.get_period_pdm(t, y, dy, periods, nbins)
        except:
            period = utils.median(periods) 
            theta = None   
    
    df = None
    data = None
    df_list = []
    return {"period": period, "theta": theta, "periods": periods}

def download_data(kic):
    try:
        folder_path = utils.download_files(kic)
        utils.process_data(folder_path)
    except Exception as e:
        print(e)
        return e

def get(kic):
    try:
        download_data(kic)        
        data = get_period(kic)
        return data
    except Exception as e:
        print(e)
        return e

def get_all():
    return utils.pd.read_csv("datasets/keplerstellar.csv",low_memory=False)

def get_kois():
    return utils.pd.read_csv("datasets/kepler-objects-of-interest.csv")

def get_binaries():
    return utils.pd.read_csv("datasets/kepler-eclipsing-binary-catalog.csv")

def get_all_without_transit():
    kepids = list(get_all().kepid)
    kepids_koi = list(get_kois())
    kepids_binary_stars = list(get_binaries().KIC)
    kepids = [x for x in kepids if x not in kepids_koi]
    kepids = [x for x in kepids if x not in kepids_binary_stars]
    return kepids

