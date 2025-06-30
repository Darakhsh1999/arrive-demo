import datetime
import constants
import pandas as pd


def convert_currency(x: pd.Series) -> pd.Series:
    if x['currency'] != "SEK":
        return x['parking_fee'] * constants.CURRENCY_TO_SEK[x['currency']]
    return x['parking_fee']

def get_weekday(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x['parking_start_time']).dayofweek
    