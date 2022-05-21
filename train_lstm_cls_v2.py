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
from tensorflow.python.keras.regularizers import l2,l1
from sklearn.metrics import recall_score,precision_score,accuracy_score,f1_score,roc_curve,classification_report,auc,precision_recall_curve,confusion_matrix

def cal_tf(pred, limit, labels):
    result = np.zeros((2, 2))

    # print('pred: ',pred)
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

    print('predict: ', y_score[-5:])
    print('true label: ', test_y[-5:])
    print('pred cls: ', y_predict[-5:])

    print('predict: ', y_score)
    print('true label: ', test_y)
    print('pred cls: ', y_predict)

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

    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, :dataset.shape[1]])

    return np.array(dataX)



if __name__ == '__main__':

    df = pd.read_excel('./glassnode_data.xlsx')
    keys = df.columns.values.tolist()
    print('keys: ',keys)

    df["dif"] = df["Price"].diff(1).dropna()
    # df['label'] = [1 if x < 0 else 0 for x in df['dif']] # 1：跌； 0：涨
    df['label'] = [1 if x >= 0 else 0 for x in df['dif']]  # 1：涨 0：跌


    del_indicator_list = ['dif', 'Price']
    # 删除列
    for indicator in del_indicator_list:
        df = df.drop([indicator], axis=1)  # 删除列

    df = df.fillna(0)
    dt = datetime(2022, 2, 1)
    df_train = df[df['t'] < dt]
    df_test = df[df['t'] >= dt ]
    # print(df)
    print('df_test: ',df_test)

    del_key_list = ['t' , 'Price']
    for key in del_key_list:
        print(key)
        idx = keys.index(key)
        print('idx: ', idx )
        del keys[idx]

    featrue_set = ['label']
    featrue_set.extend(keys)
    print('feature set: ', featrue_set)

    train_df = df_train.loc[:, featrue_set]
    train_np = train_df.values
    y = train_np[:, 0]
    y = y[1:]
    print('y type: ', type(y))
    y_nd = train_np[:, 0]
    y_ser = pd.Series(y_nd)
    print('label count: ', y_ser.value_counts())
    # y = y_nd.tolist()
    # y = [int(x) for x in y]
    x_train = train_np[:, 1:]


    test_df = df_test.loc[:, featrue_set]
    test_np = test_df.values
    y_test = test_np[:, 0]
    y_test = y_test[1:]
    # y_test_nd = test_np[:, 0]
    # y_test = y_test_nd.tolist()
    # y_test = [int(x) for x in y_test]
    x_test = test_np[:, 1:]
    print('test_np',test_np)


    #Normalize data first
    #sc = MinMaxScaler(feature_range=(0, 1))
    sc = StandardScaler()
    training_set_scaled = sc.fit_transform(x_train)
    testing_set_scaled = sc.fit_transform(x_test)

    n_step = 2
    train_X = createXY(training_set_scaled, n_step)
    test_X  = createXY(testing_set_scaled, n_step)
    y_step = n_step -1
    y = y[y_step:]
    y_test = y_test[y_step:]

    print('train_X:',train_X)
    print('y:', y)

    # reshape输入为LSTM的输入格式 reshape input to be 3D [samples, timesteps, features]
    # train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    # test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print('train_X type: ',type(train_X))
    print('train_x.shape, train_y.shape, test_x.shape, test_y.shape')
    print(train_X.shape, y.shape, test_X.shape, y_test.shape)

    ##模型定义 design network
    model = Sequential()
    # model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(LSTM(256, activation='tanh',recurrent_activation='hard_sigmoid', return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2]) ))
    model.add(LSTM(256, activation='tanh',recurrent_activation='hard_sigmoid'))
    model.add(Dropout(0.6))
    model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # 模型训练 fit network
    history = model.fit(train_X, y, epochs=15, batch_size=64, validation_data=(test_X, y_test), verbose=2,
                        shuffle=False)

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # # 进行预测 make a prediction
    # yhat = model.predict(test_X)
    print('Evaluate...')
    score = model.evaluate(test_X, y_test, batch_size=64)

    save_model(model, "./glass_cls.h5")
    y_pred = model.predict(test_X)

    y_score = model.predict_proba(test_X).flatten().tolist()
    y_predict = model.predict_classes(test_X).flatten().tolist()

    model_result(y_test, y_predict, y_score)



