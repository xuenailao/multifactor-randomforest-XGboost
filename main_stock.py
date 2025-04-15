# -*- coding: utf-8 -*-


import datetime
import numpy as np
import pandas as pd
import pickle
from preprocessing import *
from model import *
from backtest import *
from sklearn.model_selection import StratifiedKFold, cross_val_score  # 导入交叉检验算法
from sklearn.feature_selection import SelectPercentile, f_classif  # 导入特征选择方法库
from sklearn.pipeline import Pipeline  # 导入Pipeline库
from sklearn.metrics import accuracy_score  # 准确率指标
from sklearn.metrics import roc_auc_score
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #加载数据
    price = pickle.load(open('./data/price.pkl', 'rb'))
    factor_origl_data = pickle.load(open('./data/factor_origl_data.pkl', 'rb'))
    turnover = pickle.load(open('./data/turnover.pkl', 'rb'))
    stock_close_f = pickle.load(open('./data/stock_close_f.pkl', 'rb'))
    stock_pchg_f = pickle.load(open('./data/stock_pchg_f.pkl', 'rb'))
    industry_se = pickle.load(open('./data/industry_se.pkl', 'rb'))
    price_all = pickle.load(open('./data/price_all.pkl', 'rb'))
    market_close = pickle.load(open('./data/market_close.pkl', 'rb'))
    
    factor_solve_data = {}  
    factor_solve1_data = {} #原始因子因子经过变形和预处理的变量，字典形式，key是datetime，value是dataframe
    dateList = list(factor_origl_data.keys())  #2016-12-30到2020-12-31交易日列表
    model_type = 1 #模型类型，1为随机森林，2为xgboost
    model_name = '' #模型名字
    select_weight = 0.005 #选择特征的分数差
    para_weight = 0.005 #选择超参数的分数差
    cv_num = 5 #交叉验证的个数
    pro = 3*12  #划分训练与测试集，训练集的月份淑女
    money_init = 10000000 #初始金额
    score_methods = ['roc_auc', 'f1']  #选择模型的标准
    
    for date in dateList:
        #因子初始化变形
        factor_solve_data[date] = initialize_df(factor_origl_data[date],date, turnover[date], stock_close_f[date], stock_pchg_f[date])
        #因子预处理
        factor_solve1_data[date] = data_preprocessing(factor_solve_data[date],factor_origl_data[date], industry_se[date], date)

    with open('./data/factor_solve_data.pkl', 'wb') as f: #保存文件
        pickle.dump(factor_solve_data, f)
    with open('./data/factor_solve1_data.pkl', 'wb') as f:
        pickle.dump(factor_solve1_data, f)
        
    train_data, test_data = train_test_data(factor_solve1_data, dateList, price, pro) #划分训练集与测试集
    y_train = train_data['label']  #训练标签
    X_train = train_data.copy() #训练的因子数据
    del X_train['pchg']
    del X_train['label']
    
    if model_type == 1: #随机森林参数
        parameters = [50, 100, 200, 300, 400, 500]
        model_name = 'randomforest'
    elif model_type == 2: #xgboost参数
        parameters = [1,2,3,4,5,6,7,8]
        model_name = 'xgboost'
    
    #训练模型，返回model和数据选择器transform，并保存
    model, transform = train_model(model_type, X_train, y_train, select_weight, cv_num, parameters, score_methods, para_weight, model_name)
    pickle.dump(model, open("./model/"+model_name+"/model.dat","wb"))
    pickle.dump(transform, open("./model/"+model_name+"/transform.dat","wb"))
    #测试数据
    test_all, test_sample_accuracy, test_sample_roc_auc, test_sample_date = test_model(dateList[pro+1:], test_data, transform, model, model_name)
    
    #回测类
    a = Account(money_init, start_date=dateList[pro+1], end_date=dateList[-1])
    a.BackTest(test_all, price, "open", "open", 15, price_all, market_close, model_name)