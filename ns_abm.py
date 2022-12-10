import argparse
import os
import torch
import numpy as np
import pandas as pd
import sys
import datetime as dt
from dateutil.parser import parse

from Kernel import Kernel
from util import util
from util.order import LimitOrder
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle
from utils import *

from agent.ExchangeAgent import ExchangeAgent
from agent.ZeroIntelligenceAgent import ZeroIntelligenceAgent
from agent.HeuristicBeliefLearningAgent import HeuristicBeliefLearningAgent
from agent.MeanAgent import MeanAgent
from agent.MomentumAgent import MomentumAgent
from agent.market_makers.NHPBackgroundTrader import NHPBackgroundTrader
from model.LatencyModel import LatencyModel

########################################################################################################################
############################################### GENERAL CONFIG #########################################################

parser = argparse.ArgumentParser(description='Detailed options for RMSC03 config.')

parser.add_argument('-c',
                    '--config',
                    default='rmsc03',
                    help='Name of config file to execute')
parser.add_argument('-t',
                    '--ticker',
                    default = 'ABM',
                    help='Ticker (symbol) to use for simulation')
parser.add_argument('-d', '--historical-date',
                    default = '20200603',
                    type=parse,
                    help='historical date being simulated in format YYYYMMDD.')
parser.add_argument('--start-time',
                    default='09:30:00',
                    type=parse,
                    help='Starting time of simulation.'
                    )
parser.add_argument('--end-time',
                    default='10:30:00',
                    type=parse,
                    help='Ending time of simulation.'
                    )
parser.add_argument('-l',
                    '--log_dir',
                    default='rmsc03_two_hour',
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-s',
                    '--seed',
                    type=int,
                    default=101,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-v',
                    '--verbose',
                    default = False,
                    action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('--config_help',
                    action='store_true',
                    help='Print argument options for this config file')
parser.add_argument('--fund-vol',
                    type=float,
                    default=2e-10,
                    help='Volatility of fundamental time series.')
parser.add_argument('--changed_position',
                    type=int,
                    default=0,
                    help='changed param for sensitivity test.')
parser.add_argument('--mode',
                    type=int,
                    default=2,
                    help='experiment mode.')
parser.add_argument('--market_impact',
                    type=bool,
                    default=True,
                    help='include market impact or not.')
parser.add_argument('--agent_num',
                    type=list,
                    default=[0,0,0,0],
                    help='number of each type of agents.')
parser.add_argument('--wake_up_freq',
                    type=float,
                    default=30,
                    help='in secs.')


args, remaining_args = parser.parse_known_args()

if args.config_help:
    parser.print_help()
    sys.exit()

def main(seed=args.seed,agent_num=args.agent_num,market_impact=args.market_impact,gpu=0):
    args.market_impact=market_impact
    args.agent_num=agent_num
    args.seed=seed # force replace
    if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
    np.random.seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    util.silent_mode = not args.verbose
    LimitOrder.silent_mode = not args.verbose

    exchange_log_orders = True
    log_orders = None
    book_freq = 0

    simulation_start_time = dt.datetime.now()
    print("Simulation Start Time: {}".format(simulation_start_time))
    print("Configuration seed: {}\n".format(seed))
########################################################################################################################
############################################### AGENTS CONFIG ##########################################################

# Historical date to simulate.
    historical_date = pd.to_datetime(args.historical_date)
    mkt_open = historical_date + pd.to_timedelta(args.start_time.strftime('%H:%M:%S'))
    mkt_close = historical_date + pd.to_timedelta(args.end_time.strftime('%H:%M:%S'))
    agent_count, agents, agent_types = 0, [], []

# Hyperparameters
    symbol = args.ticker
    starting_cash = 10000000 # Cash in this simulator is always in CENTS.

    r_bar = 1e3  # mean fundamental value
    sigma_n = 10  # variance of oberservation of fundamental value
    kappa = 1e-12  # mean reverting speed, half life is ln(2)/kappa

# Oracle
    symbols = {symbol: {'r_bar': r_bar,
                        'kappa': 1e-12,
                        'sigma_s': 0,
                        'fund_vol': args.fund_vol,
                        'megashock_lambda_a': 2.77778e-18,
                        'megashock_mean': 100,
                        'megashock_var': 10,
                        'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))}}

    oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)
    MOMENTUM_AGENTS = True if args.agent_num[0] else False
    MEAN_AGENTS = True if args.agent_num[1] else False
    # VALUE_AGENTS = True if args.agent_num[2] else False
    ZI_AGENTS = True if args.agent_num[2] else False
    HBL_AGENTS = True if args.agent_num[3] else False
    market_impact = args.market_impact
    folder_name = '{}_{}_{}_{}_{}'.format(args.agent_num[0],args.agent_num[1],args.agent_num[2],args.agent_num[3],args.market_impact)

# 1) Exchange Agent

#  How many orders in the past to store for transacted volume computation
# stream_history_length = int(pd.to_timedelta(args.mm_wake_up_freq).total_seconds() * 100)
    stream_history_length = 10000

    agents.extend([ExchangeAgent(id=0,
                                 name="EXCHANGE_AGENT",
                                 type="ExchangeAgent",
                                 mkt_open=mkt_open,
                                 mkt_close=mkt_close,
                                 symbols=[symbol],
                                 log_orders=exchange_log_orders,
                                 pipeline_delay=0,
                                 computation_delay=0,
                                 stream_history=stream_history_length,
                                 book_freq=book_freq,
                                 wide_book=True,
                                 random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))])
    agent_types.extend("ExchangeAgent")
    agent_count += 1

# 2) NHP Agent
    model = torch.load('./models/INTC_CT4LSTM_PPP_mkt-True_coef-0_0.02rms0.002.mdl')
    agents.extend([NHPBackgroundTrader(id=agent_count,
                                       name='NHPTrader_0',
                                       type='Background Trader',
                                       symbol=symbol,
                                       starting_cash=starting_cash,
                                       lookback_length=50,
                                       NHPModel=model,
                                       log_orders=log_orders,
                                       random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')),
                                       changed_position= args.changed_position,
                                       seed = args.seed,
                                       market_impact = market_impact,
                                       folder_name = folder_name,
                                       mode = args.mode)])
    agent_count += 1
    agent_types.extend('NHPAgent')



# 3) Momentum Agents
    if MOMENTUM_AGENTS:
        num_momentum_agents = args.agent_num[0]
        agents.extend([MomentumAgent(id=j,
                                     name="MOMENTUM_AGENT_{}".format(j),
                                     type="MomentumAgent",
                                     symbol=symbol,
                                     starting_cash=starting_cash,
                                     wake_up_freq=args.wake_up_freq, # sec
                                     log_orders=log_orders,
                                     random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,dtype='uint64')),
                                     seed = args.seed,
                                     market_impact = market_impact,
                                     folder_name = folder_name)
                       for j in range(agent_count, agent_count + num_momentum_agents)])
        agent_count += num_momentum_agents
        agent_types.extend("MomentumAgent")

# 4) Mean Agents
    if MEAN_AGENTS:
        num_mean_agents = args.agent_num[1]
        agents.extend([MeanAgent(id=j,
                                     name="MEAN_AGENT_{}".format(j),
                                     type="MeanAgent",
                                     symbol=symbol,
                                     starting_cash=starting_cash,
                                     wake_up_freq=args.wake_up_freq,  # sec
                                     log_orders=log_orders,
                                     random_state=np.random.RandomState(
                                         seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')),
                                     seed=args.seed,
                                     market_impact=market_impact,
                                     folder_name=folder_name)
                       for j in range(agent_count, agent_count + num_mean_agents)])
        agent_count += num_mean_agents
        agent_types.extend("MeanAgent")


    # 5) Value Agents
    if ZI_AGENTS:
        num_zi_agents = args.agent_num[2]
        agents.extend([ZeroIntelligenceAgent(id=j,
                                             name="ZI_Agent_{}".format(j),
                                             type="ZIAgent {}",
                                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')),
                                             log_orders=log_orders,
                                             symbol=symbol,
                                             starting_cash=starting_cash,
                                             sigma_n=sigma_n,
                                             r_bar=r_bar,
                                             kappa=kappa,
                                             sigma_s=args.fund_vol,
                                             q_max=10000,  # number of lots in 100, can be long or short position
                                             sigma_pv=0,
                                             R_min=0,
                                             R_max=5,
                                             eta=1,
                                             wake_up_freq=args.wake_up_freq,
                                             seed=args.seed,
                                             market_impact=market_impact,
                                             folder_name=folder_name)
                       for j in range(agent_count, agent_count + num_zi_agents)])
        agent_count += num_zi_agents
        agent_types.extend(['ZIAgent'])

# 6) HBL Agents
    if HBL_AGENTS:
        num_hbl_agents = args.agent_num[3]
        agents.extend([HeuristicBeliefLearningAgent(id=j,
                                                     name="HBL_Agent_{}".format(j),
                                                     type="HBLAgent {}",
                                                     random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')),
                                                     log_orders=log_orders,
                                                     symbol=symbol,
                                                     starting_cash=starting_cash,
                                                     sigma_n=sigma_n,
                                                     r_bar=r_bar,
                                                     kappa=kappa,
                                                     sigma_s=args.fund_vol,
                                                     q_max=10000,  # number of lots in 100, can be long or short position
                                                     sigma_pv=0,
                                                     R_min=0,
                                                     R_max=5,
                                                     eta=1,
                                                     wake_up_freq=args.wake_up_freq,
                                                     seed=args.seed,
                                                     market_impact=market_impact,
                                                     folder_name=folder_name)
                       for j in range(agent_count, agent_count + num_hbl_agents)])
        agent_count += num_hbl_agents
        agent_types.extend("HBLAgent")


########################################################################################################################
########################################### KERNEL AND OTHER CONFIG ####################################################

    kernel = Kernel("RMSC03 Kernel", random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                      dtype='uint64')))
    kernelStartTime = mkt_open
    kernelStopTime = mkt_close# + pd.to_timedelta('00:01:00')

    defaultComputationDelay = 0

# LATENCY

    latency_rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2**32,dtype=np.int64))
    pairwise = (agent_count, agent_count)

# Latency: not used / all latency are set to zero in usage
    nyc_to_seattle_meters = 3866660
    pairwise_distances = util.generate_uniform_random_pairwise_dist_on_line(0.0, nyc_to_seattle_meters, agent_count,
                                                                            random_state=latency_rstate)
    pairwise_latencies = util.meters_to_light_ns(pairwise_distances)

    model_args = {
        'connected': True,
        'min_latency': pairwise_latencies
    }

    latency_model = LatencyModel(latency_model='deterministic',
                                 random_state=latency_rstate,
                                 kwargs=model_args
                                 )
# KERNEL
    if args.mode == 1:
        log_dir = '{}/{}'.format(str(args.seed), str(args.changed_position))
    elif args.mode == 2:
        log_dir = '{}/{}'.format(folder_name,str(args.seed))

    kernel.runner(agents=agents,
                  startTime=kernelStartTime,
                  stopTime=kernelStopTime,
                  agentLatencyModel=latency_model,
                  defaultComputationDelay=defaultComputationDelay,
                  oracle=oracle,
                  log_dir=log_dir)


    simulation_end_time = dt.datetime.now()
    print("Simulation End Time: {}".format(simulation_end_time))
    print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))

if __name__ =="__main__":
    main()
