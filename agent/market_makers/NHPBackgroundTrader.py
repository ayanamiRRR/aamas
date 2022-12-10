import os

from agent.TradingAgent import TradingAgent
from copy import deepcopy
import pandas as pd
import numpy as np
import torch
from util.util import log_print
import random

class NHPBackgroundTrader(TradingAgent):
    def __init__(self, id, name, type, symbol, starting_cash, lookback_length, NHPModel, log_orders=False, random_state=None, changed_position = 0, seed = None, market_impact = False, folder_name = None, mode = 1):

        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)

        self.symbol = symbol
        self.model = NHPModel
        self.subscribe = True
        self.subscribe_freq = 0
        self.subscribe_num_levels = 1
        self.log_orders = log_orders


        self.stored_lob = None
        self.event_stream = None
        self.subscription_requested = False
        self.event_count = None
        self.state = self.initialiseState()
        self.seen_quotes = [None, None]   # [ask_quote, bid_quote]
        self.order_dict_id_to_time = [{},{}]  # [ask_dict, bid_dict]
        self.event_count_ls = [0,0,0,0]
        self.dist_dict = np.load('./models/param_dict_95_INTC.npy', allow_pickle=True).item()
        self.count_2 = 0
        self.count_crash = 0
        self.event_ls = []

        self.dev_dict_original = {
            'dev_in_p_param': 0.25,
            'dev_in_v1_sub_param': 0.25,
            'dev_in_v2_sub_param': 0.25,
            'dev_in_mkt_param': 0.05,
            'dev_in_mkt_v_param': 0.25,
            'dev_in_lb_param': 2500,
            'dev_in_cs_param': 0.025,
        }
        self.changed_position = changed_position
        self.seed = seed
        self.market_impact = market_impact
        self.sensitivity_mode = True if mode == 1 else False
        self.folder_name = folder_name

        self.dev_dict = self.initialize_dev(self.dev_dict_original)
        if changed_position:
            for index,key in enumerate(self.dev_dict.keys()):
                if index+1 == changed_position:
                    self.dev_dict[key] = self.dev_dict_original[key] * self.random_state.uniform(-1,1)

        if self.sensitivity_mode:
            if not os.path.exists('./log/{}/{}'.format(seed,changed_position)):
                os.makedirs('./log/{}/{}'.format(seed,changed_position))
            np.save('./log/{}/{}/dev_dict.npy'.format(seed,changed_position), self.dev_dict)
        else:
            if not os.path.exists('./log/{}/{}'.format(folder_name,seed)):
                os.makedirs('./log/{}/{}'.format(folder_name,seed))

        self.market_order_pctg_ls = [[0.01+0.2*self.dev_dict['dev_in_mkt_param'], 0.01-0.2*self.dev_dict['dev_in_mkt_param']], [0.05+self.dev_dict['dev_in_mkt_param'], 0.05-self.dev_dict['dev_in_mkt_param']]]  # [ask,bid]
        self.cross_spread_pctg = 0.05 + self.dev_dict['dev_in_cs_param']
        self.volume_lb = 12500 + self.dev_dict['dev_in_lb_param']

        for key in self.dist_dict.keys():
            if 'limit_bid_sub_price' in key or 'limit_bid_cancel_price' in key:
                self.dist_dict[key][1] = self.dist_dict[key][1]+self.dev_dict['dev_in_p_param']
            elif 'market_bid_sub' in key:
                self.dist_dict[key][1] = self.dist_dict[key][1]+self.dev_dict['dev_in_mkt_v_param']
            elif 'limit_bid_sub_vol' in key:
                for i in range(len(self.dist_dict[key])):
                    if i == 0 or (i == 1 and len(self.dist_dict[key])==6):
                        self.dist_dict[key][i][1] = self.dist_dict[key][i][1] + self.dev_dict['dev_in_v1_sub_param']
                    else:
                        self.dist_dict[key][i][1] = self.dist_dict[key][i][1] + self.dev_dict['dev_in_v2_sub_param']


    def initialize_dev(self, dev_dict_original):
        new_dict = {}
        for index,key in enumerate(dev_dict_original.keys()):
            if self.sensitivity_mode:
                new_dict[key] = dev_dict_original[key] * self.random_state.uniform(-1,1)
            else:
                new_dict[key] = 0
        return new_dict


    def initialiseState(self):
        """ Returns variables that keep track of whether LOB and event stream have been observed. """

        return {
            "AWAITING_MARKET_DATA": True,
        }

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)

    def kernelTerminating(self):
        super().kernelTerminating()
        if self.sensitivity_mode:
            np.save('./log/{}/{}/event_ls.npy'.format(self.seed,self.changed_position),np.array(self.event_ls))
            np.save('./log/{}/{}/{}.npy'.format(self.seed,self.changed_position,self.count_crash),self.count_crash)
        else:
            np.save('./log/{}/{}/event_ls_{}.npy'.format(self.folder_name,self.seed,self.market_impact),np.array(self.event_ls))

    def wakeup(self, currentTime):
        """ Agent wakeup is determined by self.wake_up_freq """
        can_trade = super().wakeup(currentTime)
        if self.wake_up_at_mkt_open_flag:
            self.populate_the_market()
            self.wake_up_at_mkt_open_flag = False
            self.setWakeup(self.currentTime+pd.Timedelta(100,'ns'))
            return
        if self.subscribe and not self.subscription_requested and not self.first_wake and can_trade:
            super().requestDataSubscription(self.symbol, levels=self.subscribe_num_levels,
                                            freq=self.subscribe_freq)
            self.subscription_requested = True
            self.state = self.initialiseState()
            self.placeLimitOrder(symbol = self.symbol, quantity = 100, limit_price = 1000, is_buy_order=0, tag = 0,delay = 100)


    def populate_the_market(self):
        accumulated_time = 0
        for j in range(5):
            for side in [0,1]:
                target = self.random_state.randint(150,200)
                current = 0
                while current < target:
                    quantity = self.random_state.randint(1,10)
                    # time = self.random_state.randint(10000,10000000)
                    time = 1e6
                    accumulated_time += time
                    self.currentTime = self.currentTime + pd.Timedelta(time)
                    self.placeLimitOrder(symbol = self.symbol, quantity = quantity*100, limit_price = (1000+j) if side==0 else (999-j), is_buy_order=side,tag = None, delay = accumulated_time)
                    current = current + quantity

    def receiveMessage(self, currentTime, msg):
        """ Processes message from exchange. Main function is to update orders in orderbook relative to mid-price.

        :param simulation current time
        :param message received by self from ExchangeAgent

        :type currentTime: pd.Timestamp
        :type msg: str

        :return:
        """

        super().receiveMessage(currentTime, msg)
        sampled_event = None
        if msg.body['msg'] == 'MARKET_DATA' and self.state['AWAITING_MARKET_DATA'] == True:
            self.seen_quotes = [msg.body['asks'][0][0] if msg.body['asks'] else None,
                                msg.body['bids'][0][0] if msg.body['bids'] else None]

            self.event_count = deepcopy(msg.body['refresh_count'])
            if len(msg.body['state_stream']) >= 50:
                event_stream = torch.tensor(list(zip(*msg.body['event_stream']))[0]).cuda()
                dt = (np.array(list(zip(*msg.body['event_stream'][1:]))[1]) - np.array(list(zip(*msg.body['event_stream'][:-1]))[1]))
                dt_stream = torch.tensor([0] + list(map(lambda x:x.value, dt))).cuda() / 1e7
                state_stream = torch.tensor(msg.body['state_stream']).cuda()
                sampled_time_ls, sampled_event_ls = self.model.read_iterative_sampling({'event':event_stream.unsqueeze(0),'time_step':dt_stream.unsqueeze(0),'mkt_state':state_stream.unsqueeze(0)},mkt_state=True)
                sampled_event = sampled_event_ls[0].item()
                sampled_dt = sampled_time_ls[0].item() / 100
                if (not self.seen_quotes[0] and sampled_event == 3) or (not self.seen_quotes[1] and sampled_event == 1):
                    self.count_crash = self.count_crash + 1
                    sampled_event = sampled_event_ls[1].item()
                    sampled_dt = sampled_time_ls[1].item()

            else:
                sampled_event = self.random_state.randint(0,1) * 2
                sampled_dt = self.random_state.random() / 100

            self.state['AWAITING_MARKET_DATA'] = False

        if self.state['AWAITING_MARKET_DATA'] is False and sampled_event is not None:
            if len(self.orders) != len(self.order_dict_id_to_time[0])+len(self.order_dict_id_to_time[1]):
                raise Exception('internal error')
            self.pose_actions(sampled_event, sampled_dt, quotes = self.seen_quotes, volumes_last_seen = msg.body['volumes_last_seen'])
            self.state = self.initialiseState()

    def pose_actions(self,sampled_event,sampled_dt, quotes, volumes_last_seen):

        flag = 0
        cancel_according_to_history = 0.25
        self.event_count_ls[sampled_event] += 1
        try:
            current_spread = quotes[0] - quotes[1]
        except Exception:
            flag = 1
            current_spread = 1
            if quotes[0] is None:
                quotes[0] = quotes[1] + current_spread
            elif quotes[1] is None:
                quotes[1] = quotes[0] - current_spread
        if sampled_event == 0 or sampled_event == 2:
            is_bid = int(1-sampled_event/2)
            market_order_pctg = self.market_order_pctg_ls[current_spread == 1]
            if not self.sensitivity_mode:
                market_order_pctg_num = market_order_pctg[is_bid] * self.random_state.uniform(0,2)
            if self.random_state.random() > market_order_pctg_num or flag == 1:
                if current_spread >= 2:
                    if current_spread > 2:
                        price_diff = -1
                    elif current_spread == 2 and self.random_state.random() < self.cross_spread_pctg:
                        price_diff = -1
                    else:
                        price_diff = self.sample_from_dist(self.from_param_to_cdf(self.dist_dict['limit_bid_sub_price_2'][1]))
                    quantity = self.sample_from_dist(
                        self.from_param_to_cdf(self.dist_dict['limit_bid_sub_vol_2'][price_diff+1][1], m=int(
                            self.dist_dict['limit_bid_sub_vol_2'][price_diff+1][2]))) + 1
                else:
                    price_diff = self.sample_from_dist(self.from_param_to_cdf(self.dist_dict['limit_bid_sub_price_1'][1]))
                    quantity = self.sample_from_dist(self.from_param_to_cdf(self.dist_dict['limit_bid_sub_vol_1'][price_diff][1], m=int(
                    self.dist_dict['limit_bid_sub_vol_1'][price_diff][2]))) + 1

                if price_diff > 0:
                    if quotes[is_bid]+(sampled_event-1)*price_diff in volumes_last_seen.keys():
                        volume_at_arriving_level = abs(volumes_last_seen[quotes[is_bid]+(sampled_event-1)*price_diff])
                        if volume_at_arriving_level < self.volume_lb: # 15000
                            quantity = quantity + 10

                self.placeLimitOrder(symbol = self.symbol, quantity = int(quantity)*100, is_buy_order=is_bid, limit_price = quotes[is_bid] + price_diff*(-2*is_bid+1) ,tag = self.event_count, delay = sampled_dt * 1e9)
                self.event_ls.append([self.currentTime+pd.Timedelta(sampled_dt*1e9,'ns'), sampled_event, price_diff, quantity*100])
            else:
                if current_spread >=2:
                    quantity = self.sample_from_dist(
                        self.from_param_to_cdf(self.dist_dict['market_bid_sub_vol_2'][1], m=int(
                            self.dist_dict['market_bid_sub_vol_2'][2]))) + 1
                else:
                    quantity = self.sample_from_dist(
                        self.from_param_to_cdf(self.dist_dict['market_bid_sub_vol_1'][1], m=int(
                            self.dist_dict['market_bid_sub_vol_1'][2]))) + 1
                self.placeMarketOrder(symbol = self.symbol, quantity = int(quantity)*100, is_buy_order=is_bid,tag = self.event_count, delay = sampled_dt * 1e9)
                self.event_ls.append([self.currentTime+pd.Timedelta(sampled_dt*1e9,'ns'), sampled_event, None, quantity*100])

        elif sampled_event == 1 or sampled_event == 3:
            price_list = list(volumes_last_seen.keys())[1:]
            if current_spread >= 2:
                price_diff = self.sample_from_dist(self.from_param_to_cdf(self.dist_dict['limit_bid_cancel_price_2'][1]))
            else:
                price_diff = self.sample_from_dist(self.from_param_to_cdf(self.dist_dict['limit_bid_cancel_price_1'][1]))
            if sampled_event == 1:
                if quotes[1]-min(price_list)>4 and self.random_state.random() < cancel_according_to_history:
                    for key, value in self.orders.items():
                        if value.limit_price < quotes[1] - 4 and value.is_buy_order == 1:
                            order_id = key
                            self.cancelOrder(self.orders[order_id], tag=None, delay=sampled_dt/2 * 1e9)
                            break
                try:
                    order_id = min(list(self.order_dict_id_to_time[1].keys()))
                    for key, value in self.orders.items():
                        if value.limit_price == quotes[1] - price_diff:
                            order_id = key
                            break
                    self.cancel_or_partial_cancel(volumes_last_seen, order_id, sampled_event,
                                                  quotes[1] - self.orders[order_id].limit_price, self.event_count, sampled_dt)
                except Exception as e:
                    self.placeLimitOrder(symbol=self.symbol, quantity=100, is_buy_order=False,
                                         limit_price=quotes[0],tag=self.event_count, delay=sampled_dt * 1e9)
                    self.event_ls.append(
                        [self.currentTime + pd.Timedelta(sampled_dt * 1e9, 'ns'), 2, 0, 100])


            elif sampled_event == 3:
                if max(price_list) - quotes[0] > 4 and self.random_state.random() < cancel_according_to_history:
                    for key, value in self.orders.items():
                        if value.limit_price > quotes[0] + 4 and value.is_buy_order == 0:
                            order_id = key
                            self.cancelOrder(self.orders[order_id], tag=None, delay=sampled_dt/2 * 1e9)
                            break
                try:
                    order_id = min(list(self.order_dict_id_to_time[0].keys()))
                    for key, value in self.orders.items():
                        if value.limit_price == quotes[0] + price_diff:
                            order_id = key
                            break
                    self.cancel_or_partial_cancel(volumes_last_seen, order_id, sampled_event,
                                                  self.orders[order_id].limit_price - quotes[0], self.event_count, sampled_dt)
                except Exception as e:
                    self.placeLimitOrder(symbol=self.symbol, quantity=100, is_buy_order=True,
                                         limit_price=quotes[1],tag=self.event_count, delay=sampled_dt * 1e9)
                    self.event_ls.append(
                        [self.currentTime + pd.Timedelta(sampled_dt * 1e9, 'ns'), 0, 0, 100])


    def cancel_or_partial_cancel(self, volumes_last_seen, order_id, sampled_event, price_diff, tag, sampled_dt ):
        if abs(volumes_last_seen[self.orders[order_id].limit_price]) < self.volume_lb and self.orders[order_id].quantity > 500:
            quantity = self.random_state.randint(1, int(self.orders[order_id].quantity / 100) - 1) * 100
            self.cancelOrder_partial(self.orders[order_id], partial_volume = quantity, tag=tag, delay=sampled_dt * 1e9)
        else:
            quantity = self.orders[order_id].quantity
            self.cancelOrder(self.orders[order_id], tag=tag, delay=sampled_dt * 1e9)
        self.event_ls.append([self.currentTime + pd.Timedelta(sampled_dt*1e9,'ns'), sampled_event, price_diff, quantity])

    def getWakeFrequency(self):
        """ Get time increment corresponding to wakeup period. """
        return pd.Timedelta(100,'ns')

    def sample_from_dist(self,cdf):
        return int(np.argmax(cdf / self.random_state.random() >= 1))

    def from_param_to_cdf(self, a, m=5):
        x = np.arange(0, m, dtype='float')
        pmf = 1 / (x + 1) ** a
        pmf /= pmf.sum()
        return pmf.cumsum()


