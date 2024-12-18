import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import datetime


def load_data(FILE):
    root = ET.parse(FILE).getroot()
    round2min = 5

    # get different data types.
    dataframe_cgm = get_cgm(root, round2min)
    #dataframe_fingerstick = get_fingerstick(root)

    dataframe_gsr = get_gsr(root, round2min)
    dataframe_hr = get_hr(root, round2min)
    dataframe_st = get_st(root, round2min)

    dataframe_basal = get_basal(root, round2min)
    dataframe_bolus = get_bolus(root, round2min)
    dataframe_temp_basal = get_temp_basal(root, round2min)

    dataframe_meal = get_meal(root, round2min)
    dataframe_exercise = get_exercise(root, round2min)
    dataframe_sleep = get_sleep(root, round2min)
    dataframe_work = get_work(root, round2min)

    # set time as the index.
    dataframe_cgm = dataframe_cgm.set_index('ts')
    #dataframe_fingerstick = dataframe_fingerstick.set_index('ts')

    dataframe_gsr = dataframe_gsr.set_index('ts')
    dataframe_hr = dataframe_hr.set_index('ts')
    dataframe_st = dataframe_st.set_index('ts')

    dataframe_basal = dataframe_basal.set_index('ts')
    dataframe_bolus = dataframe_bolus.set_index('ts')
    dataframe_temp_basal = dataframe_temp_basal.set_index('ts')

    dataframe_meal = dataframe_meal.set_index('ts')
    dataframe_exercise = dataframe_exercise.set_index('ts')
    dataframe_sleep = dataframe_sleep.set_index('ts')
    dataframe_work = dataframe_work.set_index('ts')

    data_frames = [dataframe_cgm, dataframe_gsr, dataframe_hr, dataframe_st, dataframe_basal,
                   dataframe_bolus, dataframe_temp_basal, dataframe_meal, dataframe_sleep, dataframe_work, dataframe_exercise ] #, dataframe_fingerstick, ]
    df = data_frames[0]
    for df_ in data_frames[1:]:
        df = df.join(df_,how="outer")
    # print(df.to_csv('temp.csv'))
    return df
