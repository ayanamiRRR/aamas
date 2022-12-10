from agent.TradingAgent import TradingAgent
import pandas as pd
import numpy as np
from copy import deepcopy
import os


class MeanAgent(TradingAgent):
    """
    Simple Trading Agent that compares the 20 past mid-price observations with the 50 past observations and places a
    buy limit order if the 20 mid-price average >= 50 mid-price average or a
    sell limit order if the 20 mid-price average < 50 mid-price average
    """

    def __init__(self, id, name, type, symbol, starting_cash, wake_up_freq=60,
                 subscribe=False, log_orders=False, random_state=None, seed=None, market_impact=False, folder_name = None):

        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        # self.min_size = min_size  # Minimum order size
        # self.max_size = max_size  # Maximum order size
        # self.size = self.random_state.randint(self.min_size, self.max_size)
        self.size = 100
        self.wake_up_freq = wake_up_freq
        self.subscribe = subscribe  # Flag to determine whether to subscribe to data or use polling mechanism
        self.subscription_requested = False
        self.mid_list, self.avg_20_list, self.avg_50_list = [999.5]*50, [], []
        self.log_orders = log_orders
        self.state = "AWAITING_WAKEUP"
        self.order_dict_id_to_time = [{},{}]  # [ask_dict, bid_dict]
        self.market_impact = market_impact
        self.event_ls = []
        self.seed = seed
        self.folder_name = folder_name

        if not os.path.exists('./log/{}/{}'.format(folder_name,seed)):
            os.makedirs('./log/{}/{}'.format(folder_name,seed))

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)

    def kernelTerminating(self):
        super().kernelTerminating()
        np.save('./log/{}/{}/event_ls_mean_{}_{}.npy'.format(self.folder_name,self.seed,self.id,self.market_impact),np.array(self.event_ls))

    def wakeup(self, currentTime):
        """ Agent wakeup is determined by self.wake_up_freq """
        can_trade = super().wakeup(currentTime)
        if self.subscribe and not self.subscription_requested:
            super().requestDataSubscription(self.symbol, levels=1, freq=1e9)
            self.subscription_requested = True
            self.state = 'AWAITING_MARKET_DATA'
        elif can_trade and not self.subscribe:
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, currentTime, msg):
        """ Momentum agent actions are determined after obtaining the best bid and ask in the LOB """
        super().receiveMessage(currentTime, msg)
        if not self.subscribe and self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            bid, _, ask, _ = self.getKnownBidAsk(self.symbol)
            self.placeOrders(bid, ask, msg, market_impact = self.market_impact)
            self.setWakeup(currentTime + self.getWakeFrequency())
            self.state = 'AWAITING_WAKEUP'

    def placeOrders(self, bid, ask, msg, market_impact):
        """ Momentum Agent actions logic """
        if bid and ask:
            self.mid_list.append((bid + ask) / 2)
            if len(self.mid_list) > 20: self.avg_20_list.append(MeanAgent.ma(self.mid_list, n=20)[-1].round(2))
            if len(self.mid_list) > 50: self.avg_50_list.append(MeanAgent.ma(self.mid_list, n=50)[-1].round(2))
            if len(self.avg_20_list) > 0 and len(self.avg_50_list) > 0:
                if self.avg_20_list[-1] >= self.avg_50_list[-1]:
                    if self.holdings['ABM']>0:
                        size = self.holdings['ABM']
                    else:
                        size = self.size
                    self.placeMarketOrder(self.symbol, quantity=size, is_buy_order=False, tag = msg.body['refresh_count'] if market_impact else None, market_impact=market_impact)
                    self.event_ls.append([self.currentTime,2,None,size,self.holdings['CASH'],self.holdings['ABM']])
                else:
                    if self.holdings['ABM']<0:
                        size = -self.holdings['ABM']
                    else:
                        size = self.size
                    self.placeMarketOrder(self.symbol, quantity=size, is_buy_order=True, tag = msg.body['refresh_count'] if market_impact else None, market_impact=market_impact)
                    self.event_ls.append([self.currentTime,0,None,size,self.holdings['CASH'],self.holdings['ABM']])
    def getWakeFrequency(self):
        if self.freq_first_called:
            self.freq_first_called = False
            return pd.Timedelta(1,'sec')
        return pd.Timedelta(self.random_state.exponential(self.wake_up_freq),'sec')

    @staticmethod
    def ma(a, n=20):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n