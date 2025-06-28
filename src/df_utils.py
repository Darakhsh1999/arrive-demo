import datetime
import constants
import pandas as pd


def convert_currency(x: pd.Series) -> pd.Series:
    if x['currency'] != "SEK":
        return x['parking_fee'] * constants.CURRENCY_TO_SEK[x['currency']]
    return x['parking_fee']

def time_diff_convert(x: pd.Series) -> pd.Series:
    """Calculate the time difference between parking_end and parking_start. With the format date format "YYYY-MM-DD HH:MM:SS" """
    start_date = datetime.datetime.strptime(x['parking_start_time'], '%Y-%m-%d %H:%M:%S')
    end_date = datetime.datetime.strptime(x['parking_end_time'], '%Y-%m-%d %H:%M:%S')
    time_diff = end_date - start_date
    return time_diff

def get_weekday(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x['parking_start_time']).dayofweek
    