import datetime
import constants
import pandas as pd


def convert_currency(x: pd.Series) -> pd.Series:
    if x['currency'] != "SEK":
        return x['parking_fee'] * constants.CURRENCY_TO_SEK[x['currency']]
    return x['parking_fee']

def get_weekday(x: pd.Series) -> int:
    """ Returns 1 if weekday, 0 if weekend """
    weekday = pd.to_datetime(x['parking_start_time']).dayofweek

    if weekday in [5, 6]:
        return 0
    return 1
    