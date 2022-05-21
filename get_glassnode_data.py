#!/usr/bin/env python
# -*- coding: utf-8 -*-

from glassnode import GlassnodeClient
# from time import time

import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime


if __name__ == '__main__':

    gn = GlassnodeClient()
    api_key = '27B7d366MLdyKjAIP15Og5BncOj'
    gn.set_api_key(api_key)

    tomorrow_time = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    print('tomorrow_time: ', tomorrow_time)

    filename = './glassnode_indicator_v2.xlsx'
    read_excel_file = pd.read_excel(filename)

    keys = read_excel_file.keys()
    for en,cn ,api_url,exchange,coin_type in tqdm(zip(read_excel_file[keys[0]],read_excel_file[keys[1]],
                                            read_excel_file[keys[2]],read_excel_file[keys[3]],
                                            read_excel_file[keys[4]])):
        print('en: ', en, 'cn: ',cn)

        if pd.isnull(coin_type):
            coin_type = 'BTC'

        if pd.isnull(exchange):
            # print('NAN!!!')
            gn.get_and_store(
                api_url,
                a=coin_type,
                s='2010-07-18',
                u=tomorrow_time,
                i='24h',
                indicator=en
            )

        else:
            print('exchange: ', exchange)
            gn.get_and_store(
                api_url,
                a=coin_type,
                s='2010-07-18',
                u=tomorrow_time,
                i='24h',
                e = exchange,
                indicator=en
            )











