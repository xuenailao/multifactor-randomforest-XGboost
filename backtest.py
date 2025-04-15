# -*- coding: utf-8 -*-

#回测框架
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from empyrical import alpha_beta,annual_return,annual_volatility,calmar_ratio,downside_risk,max_drawdown,sharpe_ratio,simple_returns,sortino_ratio,value_at_risk,simple_returns,capture,down_capture,up_capture

class Account:

    def __init__(self, money_init, start_date='', end_date='', stop_loss_rate=-0.03, stop_profit_rate=0.05,
                 max_hold_period=5):
        self.cash = money_init  # 现金
        self.stock_value = 0  # 股票价值
        self.market_value = money_init  # 总市值
        self.stock_id = []  # 记录持仓股票id
        self.buy_date = []  # 记录持仓股票买入日期
        self.stock_num = []  # 记录持股股票剩余持股数量
        self.stock_price = []  # 记录股票的买入价格
        self.start_date = start_date #回测开始日期
        self.end_date = end_date #回测结束日期
        self.stock_asset = []  # 持仓数量
        self.dateList = [] #交易日历
        self.risk_free = 0.0000743 #无风险利率
        self.door = 0.00002  #过户费
        self.buy_rate = 0.0003  # 买入费率
        self.buy_min = 5  # 最小买入费率
        self.sell_rate = 0.0003  # 卖出费率
        self.sell_min = 5  # 最大买入费率
        self.stamp_duty = 0.001  # 印花税
        self.max_hold_period = max_hold_period  # 最大持股周期
        self.hold_day = []  # 股票持股时间
        self.cost = []  # 记录真实花费
        self.stop_loss_rate = stop_loss_rate  # 止损比例
        self.stop_profit_rate = stop_profit_rate  # 止盈比例        
        self.victory = 0  # 记录交易胜利次数
        self.defeat = 0  # 记录失败次数
        self.cash_all = [money_init]  # 记录每天收盘后所持现金
        self.stock_value_all = [0.0]  # 记录每天收盘后所持股票的市值
        self.market_value_all = [money_init]  # 记录每天收盘后的总市值
        #回测结果指标
        self.values=pd.Series(index=["max_drawdown","annual_return","annual_volatility","calmar_ratio","sharpe_ratio","downside_risk","sortino_ratio","capture","down_capture","up_capture"])
        self.info = pd.DataFrame(columns=['ts_code', 'name', 'buy_price', 'buy_date', 'buy_num', 'sell_price', 'sell_date','profit']) #交易信息
        
    # 股票买入
    def buy_stock(self, buy_date, stock_id, stock_price, buy_num): #买入日期、股票ID、股票价格、买入数量
        tmp_len = len(self.info)
        if stock_id not in self.stock_id:
            self.stock_id.append(stock_id) #更新相关买入信息
            self.buy_date.append(buy_date)
            self.stock_price.append(stock_price)
            self.hold_day.append(1)
            self.info.loc[tmp_len, 'ts_code'] = stock_id
            self.info.loc[tmp_len, 'buy_price'] = stock_price
            self.info.loc[tmp_len, 'buy_date'] = buy_date

            #计算市值、手续费等
            tmp_money = stock_price * buy_num
            service_change = tmp_money * self.buy_rate
            door_fax = tmp_money * self.door
            if service_change < self.buy_min:  #有最小买入费率限制
                service_change = self.buy_min
            tmp_cash = self.cash - tmp_money - service_change - door_fax
            while tmp_cash < 0 and buy_num > 0: #由于之前的买入数量为预估，需要考虑手续费，保证cash大于等于0
                buy_num = buy_num - 100
                tmp_money = stock_price * buy_num
                service_change = tmp_money * self.buy_rate
                door_fax = tmp_money * self.door
                if service_change < self.buy_min:
                    service_change = self.buy_min
                tmp_cash = self.cash - tmp_money - service_change - door_fax
            self.cash = tmp_cash  #更新信息
            self.info.loc[tmp_len, 'buy_num'] = buy_num
            self.stock_num.append(buy_num)
            self.cost.append(tmp_money + service_change + door_fax)
            service_change += door_fax
            info = str(buy_date) + '  买入 ' + ' (' + stock_id + ') ' \
                   + str(int(buy_num)) + '股，股价：'+str(stock_price)+',花费：' + str(round(tmp_money, 2)) + ',手续费：' \
                   + str(round(service_change, 2)) + ',花费：' + str(round(door_fax, 2)) + '，剩余现金：' + str(round(self.cash, 2))
            print(info)
            
#卖股票：卖的日期、卖的股票ID、卖的价格、卖的数量、卖的类型，0为正常卖（比如到期卖），1为止盈卖出，2为止损卖出
    def sell_stock(self, sell_date, stock_id, sell_price, sell_num, flag):
        if stock_id not in self.stock_id:
            raise TypeError('该股票未买入')
        idx = self.stock_id.index(stock_id)

        tmp_money = sell_num * sell_price #更新现金、市值、手续费等信息
        service_change = tmp_money * self.sell_rate
        if service_change < self.sell_min: #最小卖出费率限制
            service_change = self.sell_min
        stamp_duty = self.stamp_duty * tmp_money
        door_fax = self.door * tmp_money
        self.cash = self.cash + tmp_money - service_change - stamp_duty - door_fax
        service_change = stamp_duty + service_change + door_fax
        profit = tmp_money-service_change - self.cost[idx]
        if self.stock_num[idx] == sell_num:  #如果全部卖出直接删除相关信息
            del self.stock_num[idx]
            del self.stock_id[idx]
            del self.buy_date[idx]
            del self.stock_price[idx]
            del self.hold_day[idx]
            del self.cost[idx]
        else:
            self.stock_num[idx] = self.stock_num[idx] - sell_num
        
        #输出结果
        if flag == 0: #到期卖出
            info = str(sell_date) + '  到期卖出' + ' (' + stock_id + ') ' \
                   + str(int(sell_num)) + '股，股价：'+str(sell_price) + ',收入：' + str(round(tmp_money,2)) + ',手续费：' \
                   + str(round(service_change, 2)) + '，剩余现金：' + str(round(self.cash, 2))
            if profit > 0:
                info = info + '，最终盈利：' + str(round(profit, 2))
                self.victory += 1
            else:
                info = info + '，最终亏损：' + str(round(profit, 2))
                self.defeat += 1
        elif flag == 1: #止盈卖出
            info = str(sell_date) + '  止盈卖出' + ' (' + stock_id + ') ' \
                   + str(int(sell_num)) + '股，股价：' + str(sell_price) + ',收入：' + str(round(tmp_money, 2)) + ',手续费：' \
                   + str(round(service_change, 2)) + '，剩余现金：' + str(round(self.cash, 2)) \
                   + '，最终盈利：' + str(round(profit, 2))
            self.victory += 1
        elif flag == 2: #止损卖出
            info = str(sell_date) + '  止损卖出' + ' (' + stock_id + ') ' \
                   + str(int(sell_num)) + '股，股价：' + str(sell_price) + ',收入：' + str(round(tmp_money, 2)) + ',手续费：' \
                   + str(round(service_change, 2)) + '，剩余现金：' + str(round(self.cash, 2)) \
                   + '，最终亏损：' + str(round(profit, 2))
            self.defeat += 1

        print(info)
        idx = (self.info['ts_code'] == stock_id) & self.info['sell_date'].isna()
        self.info.loc[idx, 'sell_date'] = sell_date
        self.info.loc[idx, 'sell_price'] = sell_price
        self.info.loc[idx, 'profit'] = profit
            
    # 判断是否达到卖出条件：股票ID、交易日期、股票行情数据的dataframe、卖出类型（包括按照close\open\avg卖出）
    def sell_trigger(self, stock_id, day, all_df, sell_price):
        low = all_df.loc[stock_id,'low']
        high = all_df.loc[stock_id,'high']
        open_p = all_df.loc[stock_id,'open']
        close = all_df.loc[stock_id,'close']
        avg = all_df.loc[stock_id,'avg']
        idx = self.stock_id.index(stock_id)
        if high == low:  #先判断能不能卖，对于跌停的情况，以收盘价卖出是不可以的，或者开盘即跌停也是不可以卖的
            if sell_price == "close":
                return False,3,0
            elif open_p == low:
                return False,3,0
        if sell_price == "open":  #以开盘价卖出，需要判断开盘是否处于止盈或者止损点，如果处于则按照开盘价卖出
            tmp_rate = (open_p - self.stock_price[idx]) / self.stock_price[idx]
            if tmp_rate <= self.stop_loss_rate:  # 止损卖出，开盘价卖出
                return True, 2, open_p
            elif tmp_rate >= self.stop_profit_rate:  # 止盈卖出，开盘价卖出
                return True, 1, open_p
        if sell_price == "close" or sell_price == "avg": #如果以收盘价或者平均价卖出，判断最高是否出现止盈点，最低是否出现止损点
            tmp_rate = (low - self.stock_price[idx]) / self.stock_price[idx]
            if tmp_rate <= self.stop_loss_rate:  # 止损卖出，止损价卖出
                sell_price = self.stock_price[idx] * (1 + self.stop_loss_rate - 0.01)# 假设都止损价不能马上卖出，多损失 0.01%
                return True, 2, sell_price 
            tmp_rate = (high - self.stock_price[idx]) / self.stock_price[idx]
            if tmp_rate >= self.stop_profit_rate:  # 止盈卖出，止盈价卖出
                sell_price = self.stock_price[idx] * (1 + self.stop_profit_rate)
                return True, 1, sell_price
        # 判断持股周期是否达到上限
        hold_day = self.hold_day[idx]
        if hold_day >= self.max_hold_period: 
            if sell_price == "close":# 收盘价卖出
                return True, 0, close
            elif sell_price == "open":
                return True, 0 ,open_p
            else:
                return True, 0 ,avg

        return False, 3, 0
    
    # 更新信息
    def update(self, day, all_df, price_all):
        stock_price = []
        for j in range(len(self.stock_id)):
            self.hold_day[j] = self.hold_day[j] + 1  # 更新持股时间
            if self.stock_id[j] in all_df.index:  
                close = all_df.loc[self.stock_id[j],'close']
            else:
                close = price_all.loc[self.stock_id[j],'close']
            stock_price.append(close)  #更新股票价格
            
        # 更新市值等信息
        stock_price = np.array(stock_price)
        stock_num = np.array(self.stock_num)
        self.stock_value = np.sum(stock_num * stock_price)
        self.market_value = self.cash + self.stock_value
        self.market_value_all.append(self.market_value)
        self.stock_value_all.append(self.stock_value)
        self.cash_all.append(self.cash)
        
        #利用已有的市场数据（沪深300指数）计算市场收益率
    def get_market(self, market_close):
        market_profits = market_close
        market_returns=simple_returns(market_profits)     
        return market_returns

#计算回测指标
    def cal_value(self, market_close, model_name):
        returns = pd.Series(self.market_value_all, index = self.dateList)
        NAV = returns.diff()
        NAV1 = NAV[1:]
        stock_return=simple_returns(returns)  #根据总市值计算收益率序列
        market_return=self.get_market(market_close)
        risk_free = self.risk_free
        self.values["max_drawdown"]=max_drawdown(stock_return) #最大回测
        self.values["annual_return"]=annual_return(stock_return, period = 'monthly')  #年化收益率
        self.values["annual_volatility"]=annual_volatility(stock_return, period = 'monthly')   #年华波动率
        self.values["calmar_ratio"]=calmar_ratio(stock_return, period = 'monthly')  #卡马比率
        self.values["sharpe_ratio"]=sharpe_ratio(stock_return,risk_free, period = 'monthly')   #夏普比率
        self.values["downside_risk"]=downside_risk(stock_return,risk_free, period = 'monthly')  #下行风险
        self.values["sortino_ratio"]=sortino_ratio(stock_return,risk_free, period = 'monthly')  #索提诺比率
        self.values["capture"]=capture(stock_return,market_return, period = 'monthly')  #风险
        self.values["down_capture"]=down_capture(stock_return,market_return, period = 'monthly')  #下行风险
        self.values["up_capture"]=up_capture(stock_return,market_return, period = 'monthly')  #上行风险
        #保存
        stock_return.to_excel("./results/"+model_name+"/stock_return_"+str(self.cash_all[0])+".xlsx",index=True,header=False)
        NAV.to_excel("./results/"+model_name+"/NAV_"+str(self.cash_all[0])+".xlsx",index=True,header=False)
        self.values.to_excel("./results/"+model_name+"/values_"+str(self.cash_all[0])+".xlsx",index=True,header=False)
     
    
    #回测函数，测试集数据预测概率、股票行情数据的dataframe、买入方式、卖出方式、买收益率排前几的股票、测试集所有股票的收盘价数据、
    #市场收盘价、模型名字
    def BackTest(self, buy_df, all_df, buy_price, sell_price, top_num, price_all, market_close, model_name):
        self.dateList=list(buy_df.keys())  #交易日数据
        self.dateList.insert(0,self.start_date)
        for date in self.dateList[1:]:
            flag = 1
            all_df1 = all_df[date]
            buy_df1 = buy_df[date]
            price_all1 = price_all[date]
            buy_df2 = buy_df1[buy_df1["pro2"]>0.5]   #对于预测为1（上涨）的股票概率，需要大于0.5才可以
            if buy_df2.empty:
                flag = 0
            elif buy_df2.shape[0] < top_num:  #如果不足top_num，按照实际数量计算
                top_num = buy_df2.shape[0]
            tmp_df = buy_df2.sort_values('pro2', ascending=False)  #按照pro2(预测为1）的概率进行倒序，取收益率预测排名前top_num个股票购买
            stock = list(tmp_df[0:top_num].index)
            pro = (tmp_df[0:top_num])["pro2"]
            num = defaultdict(int)  #用于记下每个股票预估的购买数量，主要是区分哪些股票表明是增仓实际上是减仓
            
             # 买股
            if flag != 0:
                weight = [p/sum(list(pro)) for p in list(pro)]  #买入权重按照预测为1的概率归一化
                for j in range(len(stock)):
                    if all_df1.loc[stock[j],"volume"] == 0:  #如果停牌需要跳过
                        continue
                    if all_df1.loc[stock[j],"high_limit"] == all_df1.loc[stock[j],"high"]: #如果涨停，按照收盘价买是不可以的，开盘即涨停也是不可以买的
                        if buy_price == "colse":
                            continue
                        elif all_df1.loc[stock[j],"high"] == all_df1.loc[stock[j],"open"]:
                            continue
                    money = self.market_value * weight[j]
                    if money > self.cash: #现金不足按照既有现金买入
                        money = self.cash
                    if money < 0:  
                        continue
                    buy_num = (money / all_df1.loc[stock[j],buy_price])//100  #买股票以100股为一手，计算买入数量
                    if stock[j] in self.stock_id: 
                        pos = self.stock_id.index(stock[j])
                        buy_num = buy_num - self.stock_num[pos]  #如果该股票还在持仓，需要判断实际为增仓还是减仓
                        num[stock[j]] = buy_num*100
                    if buy_num < 0:  #实际为减仓卖股票
                        num[stock[j]] = -buy_num*100
                        continue
                    elif buy_num == 0: #不卖也不买
                        continue
                    buy_num = buy_num * 100
                    self.buy_stock(date, stock[j], all_df1.loc[stock[j],buy_price], buy_num) #执行买入操作
                    
            for j in range(len(self.stock_id) - 1, -1, -1):  #针对持仓的股票决定是否卖出
                stock_id = self.stock_id[j]
                if stock_id not in all_df1.index:  #如果本次交易日该股票已经不再指数成分股中，全部卖出
                    sell_price1 = price_all1.loc[stock_id,sell_price]
                    self.sell_stock(date, stock_id, sell_price1, self.stock_num[j], 0)
                    continue
                if all_df1.loc[stock_id,"volume"] == 0:  #跳过停牌的股票
                    continue
                
                if self.buy_date[j] == date:  #如果该持有股票这次还需要买入，则需要修改实际卖的数量
                    if num[stock_id] >= 0:
                        continue
                    else:
                        sell_num = num[stock_id]
                else:
                    sell_num = self.stock_num[j]
                is_sell, sell_kind, sell_price1 = self.sell_trigger(stock_id, date, all_df1, sell_price)  #判断是否符合卖出机制
                if is_sell:
                    self.sell_stock(date, stock_id, sell_price1, sell_num, sell_kind)  #卖股票

            self.update(date, all_df1, price_all1) #更新持股信息
        self.cal_value(market_close, model_name)  #计算回测指标