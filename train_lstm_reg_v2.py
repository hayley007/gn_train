#!/usr/bin/env python

from time import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import random
import json

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential,save_model,load_model
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from sklearn.metrics import recall_score,precision_score,accuracy_score,f1_score,roc_curve,classification_report,auc,precision_recall_curve,confusion_matrix

def cal_tf(pred, limit, labels):
    result = np.zeros((2, 2))

    print('pred: ',pred)
    for i,p in enumerate(pred):
        if p>= limit and labels[i]==1:
            result[1][1] += 1
        elif p>=limit and labels[i]==0:
            result[1][0] += 1
        elif p<limit and labels[i]==1:
            result[0][1] += 1
        else:
            result[0][0] += 1
    return result


def model_result(test_y,y_predict,y_score):
    limit_dict = dict()
    for i in range(50, 100, 5):
        limit_dict[str(i / 100)] = np.zeros((2, 2))

    for limit in limit_dict:
        limit_dict[limit] += cal_tf(y_score, float(limit), test_y)

    for limit in limit_dict:
        tf = limit_dict[limit]
        print("limit=%s tp=%s fp=%s fn=%s tn=%s" % (limit, tf[1][1], tf[1][0], tf[0][1], tf[0][0]))
        try:
            p = float(tf[1][1]) / float(tf[1][1] + tf[1][0])
            r = float(tf[1][1]) / float(tf[1][1] + tf[0][1])
            f_score = 2 * p * r / (1 * p + r)
            print("limit=%s p=%s r=%s f_score=%s" % (limit, p,r,f_score))
        except:
            continue
    print(' ')

    # accuracyscore = accuracy_score(test_y, y_predict)
    #f1 = f1_score(test_y,y_predict,average=None)
    #print('f1_score',f1)
    #fpr,tpr,threshods = roc_curve(test_y,y_score,pos_label = 1.0)
    #ks = np.max(np.abs(tpr-fpr))
    #aucscore = auc(fpr,tpr)
    #precisionf,recallf,threshods2 = precision_recall_curve(test_y,y_score,pos_label=1.0)
    precision = precision_score(test_y,y_predict,average='weighted')
    print('precision:',precision)
    recall = recall_score(test_y,y_predict,average='weighted')
    print('recall:',recall)
    # confusion = confusion_matrix(test_y,y_predict,labels=[1,0])
    ''' 
    print('f1_score',f1)
    print('precision:',precision)
    print('recall:',recall)
    print('auc:',aucscore)
    print('accuracyscore:',accuracyscore)
    print('K-S:',ks)
    print(classification_report(test_y,y_predict))   
    print(np.array([[confusion[0,0],confusion[1,0]],[confusion[0,1],confusion[1,1]]]))
    '''


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)



if __name__ == '__main__':

    df = pd.read_excel('./glassnode_data.xlsx')
    keys = df.columns.values.tolist()
    print('keys: ',keys)



    df = df.fillna(0)
    dt = datetime(2022, 2, 1)
    df_train = df[df['t'] < dt]
    df_test = df[df['t'] >= dt ]
    # print(df)
    print(df_test)

    del_key_list = ['t' ]
    for key in del_key_list:
        print(key)
        idx = keys.index(key)
        print('idx: ', idx )
        del keys[idx]


    featrue_set = ['Price', 'Mt-Gox-Balance', 'Accumulation-Balance', 'Binance-Inflow-Volume', 'UTXOs-Spent', 'Exchange-Balance',
     'Binance-Outflow-Volume-', 'Futures-Long-Liquidations-(Total)', 'Wrapped-BTC-(WBTC)-Balance',
     'Addresses-with-Balance-10', 'Highly-Liquid-Supply', 'Supply-Held-by-Entities-with-Balance-100---1k',
     'Sending-Addresses', 'New-Addresses-', 'Supply-Held-by-Entities-with-Balance-10k---100k',
     'Addresses-with-Balance-01', 'Transfer-Volume-in-Profit', 'UTXOs', 'asol', 'Accumulation-Addresses',
     'Spent-Volume-6m-12m', 'Number-of-Whales', 'Exchange-Outflow-Volume', 'Mempool-Transaction-Count',
     'Addresses-with-Balance-100', 'Spent-Volume-1w-1m', 'Short-Term-Holder-Supply-in-Loss', 'UTXOs-in-Loss',
     'Liquid-Supply', 'Options-Volume', 'Receiving-Addresses', 'Futures-Volume', '1m', '1w', '3m', '6m',
     'Coin-Days-Destroyed', 'Futures-Perpetual-Funding-rate', 'Lightning-Network-Channel-Size-(Mean)',
     'Transfer-Volume-(Total)', 'US-Month-over-Month-Price-Change', '1d_1w', '1h', '1h_24h', '1m_3m', '1w_1m', '1y_2y',
     '2y_3y', '3m_6m', '3y_5y', '5y_7y', '6m_12m', '7y_10y', 'more_10y', 'STH-SOPR', 'Luna-Foundation-Guard-Balance',
     'Spent-Volume-3m-6m', 'Exchange-Inflow-Volume', 'Hodler-net-position', 'USDT-Balance', 'Transfer-Volume-in-Loss',
     'Herfindahl-Index', 'Addresses-with-Balance-1', 'Total-Addresses', 'Coinbase-Inflow-Volume',
     'Miner-Unspent-Supply', 'Gini-Coefficient', 'Coinbase-Outflow-Volume-', 'Long-Term-Holder-Supply-in-Loss',
     'Options-Open-Interest', 'Spent-Volume-3y-5y', 'Supply-Held-by-Entities-with-Balance-100k',
     'EU-Month-over-Month-Price-Change', 'Futures-Short-Liquidations-(Total)', 'NVT-Signal',
     'Lightning-Network-Number-of-Nodes', 'Realized-Profit', 'Miner-Net-Position-Change', 'Addresses-with-Balance-10k',
     'Futures-Long-Liquidations-Dominance', 'Futures-Open-Interest', 'Withdrawing-Addresses',
     'Supply-Held-by-Entities-with-Balance-1k---10k', 'Spent-Volume-1m-3m', 'LTH-SOPR', 'Lightning-Network-Capacity',
     'BUSD-Balance', 'Liquid-Supply-Change', 'Percent-UTXOs-in-Profit', 'UTXOs-in-Profit', 'USDC-Balance', 'CVDD',
     'Addresses-with-Balance-1k', 'NVT-Ratio-', 'Illiquid-Supply-Change', 'Active-Addresses', 'Depositing-Addresses',
     'current_supply', 'Futures-Estimated-Leverage-Ratio', 'Percent-Addresses-in-Profit', 'Miner-Balance',
     'Asia-Month-over-Month-Price-Change', 'Addresses-with-Non-Zero-Balance', 'Short-Term-Holder-Supply']
    print('feature set: ',featrue_set)
    print('feature set len: ', len(featrue_set) )

    train_df = df_train.loc[:, featrue_set]
    # df_train = df_train.drop(['t'], axis=1)  # 删除列
    train_np = train_df.values
    test_df = df_test.loc[:, featrue_set]
    # df_test = df_test.drop(['t'], axis=1)  # 删除列
    test_np = test_df.values
    print(test_np)


    #Normalize data first
    sc = MinMaxScaler(feature_range=(0, 1))
    # sc = StandardScaler()
    training_set_scaled = sc.fit_transform(train_np)
    testing_set_scaled = sc.fit_transform(test_np)


    train_X, y = createXY(training_set_scaled, 3)
    test_X, y_test = createXY(testing_set_scaled, 3)

    print("trainX Shape-- ", train_X.shape)
    print("trainY Shape-- ", y.shape)

    # reshape输入为LSTM的输入格式 reshape input to be 3D [samples, timesteps, features]
    # train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    # test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print('train_X type: ',type(train_X))
    print('train_x.shape, train_y.shape, test_x.shape, test_y.shape')
    print(train_X.shape, y.shape, test_X.shape, y_test.shape)

    ##模型定义 design network
    model = Sequential()
    # model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(LSTM(256, activation='tanh', recurrent_activation='hard_sigmoid' ,return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(LSTM(256, activation='tanh',recurrent_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    # model.compile(loss='mae', optimizer='adam')
    model.compile(loss='mae', optimizer='rmsprop' )

    # 模型训练 fit network
    history = model.fit(train_X, y, epochs=80, batch_size=64, validation_data=(test_X, y_test), verbose=2,
                        shuffle=False) #ok


    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # # 进行预测 make a prediction
    # yhat = model.predict(test_X)
    print('Evaluate...')
    # score =  model.evaluate(test_X,y_test,batch_size = 64)

    # make a prediction
    yhat = model.predict(test_X)
    print('yhat shape: ',yhat.shape)
    # test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    prediction_copies_array = np.repeat(yhat, test_X.shape[2], axis=-1)
    inv_yhat = sc.inverse_transform(np.reshape(prediction_copies_array, (len(yhat), test_X.shape[2])))[:, 0]

    test_y = np.repeat(y_test, test_X.shape[2], axis=-1)
    inv_y = sc.inverse_transform(np.reshape(test_y, (len(y_test), test_X.shape[2])))[:, 0]

    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    print('ori:',inv_y)
    print('predict: ',inv_yhat)

    save_model(model, "./glass_reg.h5")







