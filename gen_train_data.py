#!/usr/bin/env python
# -*- coding: utf-8 -*-


from time import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import json



if __name__ == '__main__':

    folder_path = './data_glassnode/*.csv'
    data_path = glob.glob(folder_path)

    df = pd.DataFrame()

    for path in data_path:
        print('file path: ', path)

        file_name =  path.split('/')[-1:][0]
        indicator = file_name.split('.')[0]
        data = pd.read_csv(path,index_col="t",parse_dates=True)

        # print(data)
        if indicator == 'Spent-Output-Age-Bands' or indicator == 'Options-ATM-Implied-Volatility-(All)':
            # print(type(data['o']))
            # print(data['o'])
            # print('index: ', data.index)
            # print(data.head(5))
            data['o'] = data['o'].apply(lambda x: eval(x))
            # print(data.head(5))
            df_sub = pd.json_normalize(data['o']).set_index(data.index)
            print(df_sub.head(5))
            df = pd.concat([df, df_sub], axis=1)
            continue

        if indicator == 'URPD':
            # data = data.drop(['partitions'],axis=1) #删除列
            print(data)

            current_supply_list = []

            for i,row in data.iterrows():
                partitions = row['partitions']
                partitions_list = json.loads(partitions)
                # print('partitions_ori type: ', type(partitions_list[0]))
                # print('partitions_list: ', partitions_list)
                ath_price_ori = row['ath_price']
                ath_price = float(ath_price_ori)
                # print('ath_price: ', ath_price)
                current_price_ori = row['current_price']
                total_supply_ori = row['total_supply']
                current_price = float(current_price_ori)
                # print('current_price: ',current_price)

                price_list = []
                for i in range(0, 100):
                    price = i * ath_price / 100
                    price_list.append(price)

                price_list_show = []
                partitions_list_show = []
                position = 0
                for i in range(1, 100, 2):
                    price_list_show.append(price_list[i])
                    partitions_list_show.append(partitions_list[i])

                # print('price_list_show: ',len(price_list_show) )

                for i in range(0, 49):
                    if current_price >= price_list_show[i] and current_price <= price_list_show[i + 1]:
                        position = i
                        break
                if current_price == ath_price:
                    position = 49
                # print('position: ', position)
                current_supply = partitions_list_show[position]
                current_supply_list.append(current_supply)
                # print('current_supply:', current_supply)

            # for i in range( len(current_supply_list) ):
            #     data = data.append({'current_supply': current_supply_list[i]})

            t = data.reset_index()['t']
            df_sub = pd.DataFrame(list(zip(t, current_supply_list)),columns=['t','current_supply'])
            df_sub = df_sub.set_index('t')
            print(df_sub)
            df = pd.concat([df, df_sub], axis=1)

            continue

        data.rename(columns={'v': indicator} , inplace = True)
        df = pd.concat([df, data], axis=1)

        # print(df)

    df.to_excel('glassnode_data.xlsx')

