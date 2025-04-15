# -*- coding: utf-8 -*-


#因子数据预处理
import statsmodels.api as sm
import pandas as pd
import math
import numpy as np

#从factor_orgil_data得到需要的因子数据，返回factor_solve_data[date]，date为指定日期
def initialize_df(df,date,data_turnover_ratio, stock_close, stock_pchg):
    stocklist = list(df.index)
    
    df['net_assets']=df['total_assets']-df['total_liability']#净资产
    df_new = pd.DataFrame(index=stocklist)
        
    #估值因子
    df_new['EP'] = df['pe_ratio'].apply(lambda x: 1/x) #市盈率因子
    df_new['BP'] = df['pb_ratio'].apply(lambda x: 1/x) #市净率
    df_new['SP'] = df['ps_ratio'].apply(lambda x: 1/x) #营业收入/总市值
    df_new['DP'] = df['dividend_payable']/df['market_cap'] #DP因子
    df_new['RD'] = df['development_expenditure']/df['market_cap'] #RD因子
    df_new['CFP'] = df['pcf_ratio'].apply(lambda x: 1/x) #CFP因子
    
    #杠杆因子
    
    df_new['CMV'] = np.log(df['circulating_market_cap'])#对数流通市值
    df_new['financial_leverage']=df['total_assets']/df['net_assets']#总资产/净资产
    df_new['debtequityratio']=df['total_non_current_liability']/df['net_assets']#非流动负债/净资产
    df_new['cashratio']=df['cash_to_current_liability']#现金比率=(货币资金+有价证券)÷流动负债
    df_new['currentratio']=df['current_ratio']#流动比率=流动资产/流动负债*100%
    df_new['ocf_g_q']=df['net_operate_cash_flow']/df['net_operate_cash_flowlast_year']-1#经营性现金流(YTD)同比增长率
    df_new['roe_g_q']=df['roe']/df['roelast_year']-1#ROE(YTD)同比增长率

    #动量因子，（1个月、3个月、6个月、12个月）
    df_new['return_1m']=stock_close.iloc[-1]/stock_close.iloc[-20]-1
    df_new['return_3m']=stock_close.iloc[-1]/stock_close.iloc[-60]-1
    df_new['return_6m']=stock_close.iloc[-1]/stock_close.iloc[-120]-1
    df_new['return_12m']=stock_close.iloc[-1]/stock_close.iloc[-240]-1
    
    #波动率（1个月、3个月、6个月、12个月）
    df_new['std_1m']=stock_pchg.iloc[-20:].std()
    df_new['std_3m']=stock_pchg.iloc[-60:].std()
    df_new['std_6m']=stock_pchg.iloc[-120:].std()
    df_new['std_12m']=stock_pchg.iloc[-240:].std()
    
    #股价
    df_new['ln_price']=np.log(stock_close.iloc[-1])

    #换手率（1个月、3个月、6个月、12个月）
    df_new['turn_1m']=np.mean(data_turnover_ratio.iloc[-20:])
    df_new['turn_3m']=np.mean(data_turnover_ratio.iloc[-60:])
    df_new['turn_6m']=np.mean(data_turnover_ratio.iloc[-120:])
    df_new['turn_12m']=np.mean(data_turnover_ratio.iloc[-240:])
    df_new['bias_turn_1m']=np.mean(data_turnover_ratio.iloc[-20:])/np.mean(data_turnover_ratio)-1 #归一化
    df_new['bias_turn_3m']=np.mean(data_turnover_ratio.iloc[-60:])/np.mean(data_turnover_ratio)-1
    df_new['bias_turn_6m']=np.mean(data_turnover_ratio.iloc[-120:])/np.mean(data_turnover_ratio)-1
    df_new['bias_turn_12m']=np.mean(data_turnover_ratio.iloc[-240:])/np.mean(data_turnover_ratio)-1

    
    #财务质量因子
    df_new['operationcashflowratio_ttm']=df['net_operate_cash_flow_ttm']/df['net_profit_ttm']#经营性现金流/净利润TTM
    df_new['operationcashflowratio_q']=df['net_operate_cash_flow']/df['net_profit']#经营性现金流/净利润YTD    
    df_new['NI'] = df['net_profit_to_total_operate_revenue_ttm']# 净利润与营业总收入之比
    df_new['GPM'] = df['gross_income_ratio'] #销售毛利率
    df_new['grossprofitmargin_q']=df['gross_profit_margin']#毛利率YTD
    df_new['ROE'] = df['roe_ttm'] #ROE_TTM
    df_new['roe_q']=df['roe']#ROE_YTD
    df_new['roa_q']=df['roa']#ROA_YTD
    df_new['ROA'] = df['roa_ttm'] #ROA_TTM
    df_new['asset_turnover'] = df['total_asset_turnover_rate']  #总资产周转率
    df_new['assetturnover_q']=df['operating_revenue']/df['total_assets']#资产周转率YTD 营业收入/总资产
    df_new['net_operating_cash_flow'] = df['net_operating_cash_flow_coverage']  #净利润现金含量

    #成长因子
    df_new['Sales_G_q'] = df['operating_revenue_growth_rate'] #营收增长率
    df_new['Profit_G_q'] = df['net_profit_growth_rate'] #净利润增长率
    
    #技术指标
    df_new['RSI']=df['RSI']
    df_new['DIF']=df['DIF']
    df_new['DEA']=df['DEA']
    df_new['MACD']=df['MACD']
    
    return df_new

def filter_extreme_MAD(data,n): #中位数去极值，n规定了中位数的偏差范围
    median = data.quantile(0.5)
    new_median = ((data - median).abs()).quantile(0.50)
    max_range = median + n*new_median
    min_range = median - n*new_median
    for n in data.columns:
        data.loc[data[n] > max_range[n],n] = max_range[n]
        data.loc[data[n] < min_range[n],n] = min_range[n] 
    return data

def neutralize(data,date,market_cap, industry_se): #中性化，使用行业和市值因子中性化，market_cap:市值因子
    columns = list(data.columns)
    if isinstance(industry_se,pd.Series):
        industry_se = industry_se.to_frame()
    if isinstance(market_cap,pd.Series):
        market_cap = market_cap.to_frame()

    index = list(data.index)
    industry_se1 = np.array(industry_se.loc[index,'code'].tolist())
    industry_dummy = sm.categorical(industry_se1,drop=True)
    industry_dummy = pd.DataFrame(industry_dummy,index=index)
    market_cap = np.log(market_cap.loc[index])
    x = pd.concat([industry_dummy,market_cap],axis=1)
    model = sm.OLS(data,x)
    result = model.fit()
    y_fitted =  result.fittedvalues
    neu_result = data - y_fitted
    return neu_result

def standardize(data,ty=2):#标准化函数,ty为标准化类型:1 MinMax,2 Standard,3 maxabs 
    if int(ty)==1:
        re = (data - data.min())/(data.max() - data.min())
    elif ty==2:
        re = (data - data.mean())/data.std()
    elif ty==3:
        re = data/10**np.ceil(np.log10(data.abs().max()))
    return re

def get_industry_name(i_Constituent_Stocks, value):#取股票对应行业
    return [k for k, v in i_Constituent_Stocks.items() if value in v]


def replace_nan_indu(factor_data, industry_se, date):#缺失值处理，
    industry_code = list(set(industry_se['code']))
    data_temp=pd.DataFrame(index=industry_code,columns=factor_data.columns)
    for i in industry_code: #求个每个行业的平均值
        industry_stock = industry_se[industry_se['code']==i].index
        data_temp.loc[i]=np.mean(factor_data.loc[industry_stock,:])
    for factor in data_temp.columns:
        null_industry=list(data_temp.loc[pd.isnull(data_temp[factor]),factor].keys())
        for i in null_industry: #对行业缺失的使用所有平均值代替
            data_temp.loc[i,factor]=np.mean(data_temp[factor])
        null_stock=list(factor_data.loc[pd.isnull(factor_data[factor]),factor].keys())
        for i in null_stock:  #NAN处理，使用行业的平均值代替
            if i in industry_se.index:
                factor_data.loc[i,factor]=data_temp.loc[industry_se.loc[i,'code'],factor] 
            else:
                factor_data.loc[i,factor]=np.mean(factor_data[factor])
    return factor_data

#数据预处理
def data_preprocessing(factor_data,factor_orgil_data, industry_se, date):
    #去极值
    factor_data=filter_extreme_MAD(factor_data, 5)
    #缺失值处理
    factor_data=replace_nan_indu(factor_data,industry_se,date)
    #中性化处理
    factor_data=neutralize(factor_data, date,factor_orgil_data["market_cap"], industry_se)
    #标准化处理
    factor_data=standardize(factor_data)
    return factor_data