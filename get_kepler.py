import utils

def read_csv(kic):
    periods = []
    df_list = []
    filenames = utils.get_filenames(utils.BASE_PATH+str(kic),'csv')
    for idx, filename in enumerate(filenames):
        data = utils.pd.read_csv(utils.BASE_PATH+str(kic)+'/'+filename)
        try:
            res = utils.get_signal_parameters(data.dropna().TIME, data.dropna().PDC_RAW_MEDIAN)
            periods.append(res["period"])
        except Exception as e:
            print(e)
            print(idx)

        df_list.append(data)

    period = utils.get_period(periods)
    return {"df_list": df_list, "period": period}

def get(kic):
    try:
        folder_path = utils.download_files(kic)
        utils.process_data(folder_path)
        data = read_csv(kic)
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

