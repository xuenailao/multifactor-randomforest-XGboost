# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score  
from sklearn.feature_selection import SelectPercentile, f_classif 
from sklearn.pipeline import Pipeline  
from sklearn.metrics import accuracy_score  
from sklearn.metrics import roc_auc_score
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import time, datetime

def train_test_data(factor_data, dateList, price, pro):
    train_data=pd.DataFrame() #训练集数据
    for date in dateList[1:pro+1]:
        traindf=factor_data[date]
        start = price[date]['close']
        end1 = price[dateList[dateList.index(date)-1]]['close']
        traindf['pchg']=start/end1-1 #从收盘价得到收益率数据，并提出NAN
        traindf=traindf.dropna()   
        traindf=traindf.sort_values(by='pchg' , ascending=True)
        #只选取pchg排前30%与后30%的数据作为正负例训练，剔除噪声
        traindf=traindf.iloc[:int(len(traindf['pchg'])/10*3),:].append(traindf.iloc[int(len(traindf['pchg'])/10*7):,:])
        traindf['label']=list(traindf['pchg'].apply(lambda x:1 if x>np.mean(list(traindf['pchg'])) else 0))   #label  
        if train_data.empty:
            train_data=traindf
        else:
            train_data=train_data.append(traindf)
        
    # 测试集数据        
    test_data={}
    for date in dateList[pro+1:]:
        testdf=factor_data[date]
        stockList=list(testdf.index)
        start = price[date]['close']
        end1 = price[dateList[dateList.index(date)-1]]['close']
        testdf['pchg']=start/end1-1 #计算pchg
        testdf=testdf.dropna()   
        #选取前后各30%的股票，剔除中间的噪声
        testdf=testdf.sort_values(by='pchg' , ascending=True) 
        testdf=testdf.iloc[:int(len(testdf['pchg'])/10*3),:].append(testdf.iloc[int(len(testdf['pchg'])/10*7):,:])
        testdf['label']=list(testdf['pchg'].apply(lambda x:1 if x>np.mean(list(testdf['pchg'])) else 0)) #label
        test_data[date]=testdf
        
    return train_data,test_data

#特征选择，返回特征百分比
def select_feature(model_pipe, X_train, y_train, select_weight, model_name):
    score_means = list()
    score_stds = list()
    percentiles = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100) #特征百分比候选序列
    for percentile in percentiles:
        model_pipe.set_params(ANOVA__percentile=percentile) #设立训练管道，使用ANOVA筛选特征
        this_scores = cross_val_score(model_pipe, X_train, y_train, cv=5, n_jobs=-1) #交叉验证
        score_means.append(this_scores.mean())
        score_stds.append(this_scores.std())
    s = max(score_means)
    pos = score_means.index(s)
    pos1=pos
    for i in range(pos-1,-1,-1): #寻找合适的百分比参数
        if (s-score_means[i])>select_weight:
            pos1 = i
            break
    plt.errorbar(percentiles, score_means, np.array(score_stds)) #画score mean与std随着百分比的变化趋势图
    plt.title('Performance of the model-Anova varying the percentile of features selected')
    plt.xlabel('Percentile')
    plt.ylabel('Prediction rate')
    plt.axis('tight')
    plt.savefig("./model/"+str(model_name)+"/feature_percent.jpg")
    plt.show()
    
    if pos1 == pos:
        return percentiles[int(pos/2)]
    else:
        return percentiles[pos1+1]

#筛选超参数
def select_para(cv_num, parameters, score_methods,para_weight, model_type, model_pipe, X_train, y_train):
    cv = StratifiedKFold(cv_num) #交叉验证划分
    last = 0
    mean_list = []
    std_list = []
    for parameter in parameters:
        t1 = time.time()  # 记录训练开始的时间
        score_list = list()  # 建立空列表用于存放不同交叉检验下各个评估指标的详细数据
        print ('set parameters: %s' % parameter)  
        for score_method in score_methods:  
            if model_type == 1:
                model_pipe.set_params(model__n_estimators=parameter)
            elif model_type == 2:
                model_pipe.set_params(model__max_depth=parameter)            
            score_tmp = cross_val_score(model_pipe, X_train, y_train, scoring=score_method, cv=cv, n_jobs=-1)  # 使用交叉检验计算指定指标的得分
            score_list.append(score_tmp) 
        score_matrix = pd.DataFrame(np.array(score_list), index=score_methods)  
        score_mean = score_matrix.mean(axis=1).rename('mean')  # 计算每个评估指标的均值
        score_std = score_matrix.std(axis=1).rename('std')  # 计算每个评估指标的标准差
        score_pd = pd.concat([score_matrix, score_mean, score_std], axis=1) 
        mean_list.append(score_mean["roc_auc"])  
        std_list.append(score_std)  
        print (score_pd.round(4))  # 打印结果
        print ('-' * 60)
        t2 = time.time()  
        tt = t2 - t1  # 计算时间间隔
        print ('time: %s' % str(tt))  
    s = max(mean_list)  #筛选合适的超参数并返回
    pos = mean_list.index(s)
    pos1=pos
    for i in range(pos-1,-1,-1):
        if (s-mean_list[i])>para_weight:
            pos1 = i
            break
    if pos1 == pos:
        return parameters[int(pos/2)]
    else:
        return parameters[pos1+1]

#训练模型
def train_model(model_type, X_train, y_train, select_weight, cv_num, parameters, score_methods, para_weight, model_name):
    transform = SelectPercentile(f_classif)  #先进行特征筛选
    if model_type == 1:
        model = RandomForestClassifier()
    elif model_type == 2:
        model = XGBClassifier()
    model_pipe = Pipeline(steps=[('ANOVA', transform), ('model', model)])  # 建立由特征选择和分类模型构成的“管道”对象
    p = select_feature(model_pipe, X_train, y_train, select_weight, model_name)
    print(p)
    
    #再进行超参数筛选
    transform = SelectPercentile(f_classif,percentile=p)  
    if model_type == 1:
        model = RandomForestClassifier()
    elif model_type == 2:
        model = XGBClassifier()
    model_pipe = Pipeline(steps=[('ANOVA', transform), ('model', model)])  
    para = select_para(cv_num, parameters, score_methods,para_weight, model_type, model_pipe, X_train, y_train)
    print(para)

    #根据参数选择处理训练因子X_train得到X_train_final
    transform.fit(X_train, y_train)  
    X_train_final = transform.transform(X_train)  
    if model_type == 1:
        model = RandomForestClassifier(n_estimators=para,random_state=0)
        model1 = RandomForestClassifier(n_estimators=para,random_state=0)
    elif model_type == 2:
        model = XGBClassifier(max_depth=para,subsample=0.9,random_state=0)
        model1 = XGBClassifier(max_depth=para,subsample=0.9,random_state=0)
    model1.fit(X_train, y_train)  # 原始数据训练模型
    model.fit(X_train_final, y_train)  #特征选择后的数据训练模型

    fig = plt.figure(figsize= (15,10)) #画图，特征重要性
    n_features = X_train.shape[1]
    plt.barh(range(n_features),model1.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),X_train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.savefig('./model/'+str(model_name)+'/feature_importance.png')
    
    return model, transform

#测试数据
def test_model(dateList, test_data, transform, model, model_name):
    test_sample_predict={} #记录预测结果，字典形式，key为date，value为预测标签
    test_all = {} #记录预测结果，字典形式，key为date，value为dataframe（因子+预测概率+预测标签）
    test_sample_accuracy=[]  #记录预测ACC
    test_sample_roc_auc=[] #记录预测ROC\AUC
    test_sample_date=[] #记录预测交易日
    for date in dateList:
        y_test=test_data[date]['label']
        X_test=test_data[date].copy()
        del X_test['pchg']
        del X_test['label']    
        X_test_final = transform.transform(X_test)  # 对测试数据集做特征选择
        y_pred_tmp = model.predict(X_test_final) #预测
        y_pred = pd.DataFrame(y_pred_tmp, columns=['label_predict'])  
        y_pred_proba = pd.DataFrame(model.predict_proba(X_test_final), columns=['pro1', 'pro2'])  
        y_pred.set_index(X_test.index,inplace=True)
        y_pred_proba.set_index(X_test.index,inplace=True)
        predict_pd = pd.concat((X_test, y_pred, y_pred_proba), axis=1)# 将预测标签、预测数据和原始数据X合并
        print ('Predict date:')
        print (date)    
        print ('AUC:')
        print (roc_auc_score(y_test,y_pred)) 
        print ('Accuracy:')
        print (accuracy_score(y_test, y_pred))    
        print ('-' * 60)       
        #记录结果
        test_sample_date.append(date)
        test_sample_predict[date]=y_pred_tmp
        test_all[date]=predict_pd
        test_sample_accuracy.append(accuracy_score(y_test, y_pred))   
        test_sample_roc_auc.append(roc_auc_score(y_test,y_pred))
    print ('AUC mean info')
    print (np.mean(test_sample_roc_auc))
    print ('-' * 60)    
    print ('ACCURACY mean info')
    print (np.mean(test_sample_accuracy))
    print ('-' * 60)  

    #测试集特征关联度，画图
    factor_predict_corr=pd.DataFrame()
    for date in dateList:
        test_feature=test_data[date].copy()
        del test_feature['pchg']
        del test_feature['label']
        test_feature['predict']=list(test_sample_predict[date])
        factor_predict_corr[date]=test_feature.corr()['predict']    
    factor_predict_corr=factor_predict_corr.iloc[:-1]
    fig = plt.figure(figsize= (15,10))
    ax = fig.add_subplot(111)
    sns.set()
    ax = sns.heatmap(factor_predict_corr)
    fig.savefig('./model/'+str(model_name)+'/corr.png')
    
    #ACC与AUC测试绘图
    xs_date = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in test_sample_date]
    ys_auc = test_sample_roc_auc
    ys_score = test_sample_accuracy
    f = plt.figure(figsize= (15,10))
    sns.set(style="whitegrid")
    data1 = pd.DataFrame(ys_auc, xs_date, columns={'AUC'})
    data2 = pd.DataFrame(ys_score, xs_date, columns={'accuracy'})
    data = pd.concat([data1,data2],sort=False)
    fig = sns.lineplot(data=data, palette="tab10", linewidth=2.5)
    fig1 = fig.get_figure()
    fig1.savefig('./model/'+str(model_name)+'/test_roc_auc_acc.png')
    
    return test_all, test_sample_accuracy, test_sample_roc_auc, test_sample_date
