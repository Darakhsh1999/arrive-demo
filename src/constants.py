""" Currency conversion rates to SEK (standard) 28.06.2025 """

# Hard coded but ideally tied to some API
DKK_TO_SEK = 1.49
EUR_TO_SEK = 11.11
NOK_TO_SEK = 0.94

CURRENCY_TO_SEK = {
    'DKK': DKK_TO_SEK,
    'EUR': EUR_TO_SEK,
    'NOK': NOK_TO_SEK
}

AREA_TYPE_MAPPING = {
    'AboveGroundGarage': 0,
    'SurfaceLot': 1,
    'Administrative': 2,
    'OnStreet': 3,
    'UndergroundGarage': 4,
    'EVC': 5,
    'CameraParkArea': 6
}


ACCOUNT_TYPE_MAPPING = {
    'private': 0,
    'corporate': 1
}
    