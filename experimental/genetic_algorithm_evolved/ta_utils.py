import numpy as np
import pandas as pd
from typing import Tuple, List

def calculate_cumulative_return(ret: np.ndarray) -> List[float]:
    cum_ret_list = [ret[0]]
    n = ret.shape[0]
    for i in range(1, n):
        cum_ret = (1 + ret[i]) * (1 + cum_ret_list[-1]) - 1
        cum_ret_list.append(cum_ret)
    cum_ret_list.insert(0, np.nan)
    return cum_ret_list

def calculate_strategy_returns(signal: np.ndarray, ret: np.ndarray) -> np.ndarray:
    strat_ret = np.array(signal[0:-1]) * np.array(ret[1::])
    strat_ret = np.insert(strat_ret, 0, np.nan)
    return strat_ret

def calculate_strategy_cumulative_returns(strat_ret: np.ndarray) -> List[float]:
    strat_cum_ret = calculate_cumulative_return(strat_ret[1::])
    return strat_cum_ret

def find_trade_timing(signal: np.ndarray) -> Tuple[List[int], List[int]]:
    n = len(signal)
    enter = []
    exit = []
    for i in range(1, n):
        before_trade = signal[i - 1]
        trade = signal[i]
        if before_trade == 0 and (trade == 1 or trade == -1):
            enter.append(i)
        elif (before_trade == 1 or before_trade == -1) and trade == 0:
            exit.append(i)
    return enter, exit

def calculate_win_rate(ret: np.ndarray, enter: List[int], exit: List[int]) -> Tuple[float, List[int]]:
    if len(enter) == len(exit):
        n = len(enter)
    else:
        n = min(len(enter), len(exit))
    
    acc_ret_list = []
    win_pos = []
    
    for i in range(n):
        sec_ret = ret[enter[i] + 1:exit[i] + 1]
        acc_ret = calculate_cumulative_return(sec_ret)[-1]
        acc_ret_list.append(acc_ret)
        if acc_ret > 0:
            win_pos.append(i)
    
    try:
        winrate = len([i for i in acc_ret_list if i > 0]) / len(acc_ret_list)
    except ZeroDivisionError:
        winrate = 0
    
    return winrate, win_pos

def adjust_parameter_positions(pop: np.ndarray, position: List[int]) -> np.ndarray:
    pop = np.array(pop)
    ori_low = pop[:, position[0]]
    ori_high = pop[:, position[1]]
    
    new_high = np.where(ori_low > ori_high, ori_low, ori_high)
    new_low = np.where(ori_low > ori_high, ori_high, ori_low)
    
    pop[:, position[0]] = new_low
    pop[:, position[1]] = new_high
    return pop

def generate_rsi_signals(price: np.ndarray, rsi: np.ndarray, buy_sig: float, 
                        sell_sig: float, error_tol: float = 0.03) -> np.ndarray:
    order_status = 'no_order'
    ta_direc = 'no_direc'
    pos = [0]
    
    for i in range(1, len(price)):
        pre_rsi = rsi[i - 1]
        cur_rsi = rsi[i]
        
        if order_status == 'order_placed':
            if ta_direc == 'long':
                if (pre_rsi < cur_rsi and (abs(cur_rsi - sell_sig) / sell_sig) < error_tol) or (cur_rsi > sell_sig):
                    order_status = 'no_order'
                    ta_direc = 'no_direc'
                    pos.append(0)
                else:
                    pos.append(1)
            elif ta_direc == 'short':
                if (pre_rsi > cur_rsi and (abs(cur_rsi - buy_sig) / buy_sig) < error_tol) or (cur_rsi < buy_sig):
                    order_status = 'no_order'
                    ta_direc = 'no_direc'
                    pos.append(0)
                else:
                    pos.append(-1)
        
        elif order_status == 'no_order':
            if (cur_rsi >= sell_sig > pre_rsi) or (pre_rsi > sell_sig and pre_rsi > cur_rsi
                                               and (abs(cur_rsi - sell_sig) / sell_sig) <= error_tol):
                ta_direc = 'short'
                order_status = 'order_placed'
                pos.append(-1)
            
            elif (cur_rsi <= buy_sig < pre_rsi) or (pre_rsi < buy_sig and pre_rsi < cur_rsi
                                                and (abs(cur_rsi - buy_sig) / buy_sig) <= error_tol):
                ta_direc = 'long'
                order_status = 'order_placed'
                pos.append(1)
            else:
                order_status = 'no_order'
                pos.append(0)
    
    return np.array(pos)

def generate_rsi_stoploss_takeprofit_signals(price: np.ndarray, rsi: np.ndarray, 
                                            buy_sig: float, sell_sig: float, 
                                            stop_loss: float, take_profit: float, 
                                            error_tol: float = 0.03) -> np.ndarray:
    order_status = 'no_order'
    ta_direc = 'no_direc'
    pos = [0]
    enter_price = None
    
    for i in range(1, len(price)):
        pre_rsi = rsi[i - 1]
        cur_rsi = rsi[i]
        
        if order_status == 'order_placed':
            if ta_direc == 'long':
                unrealized_ret = (price[i] / enter_price) - 1
                if (pre_rsi < cur_rsi and (abs(cur_rsi - sell_sig) / sell_sig) < error_tol) or (cur_rsi > sell_sig) \
                        or (unrealized_ret <= stop_loss) or (unrealized_ret >= take_profit):
                    order_status = 'no_order'
                    ta_direc = 'no_direc'
                    enter_price = None
                    pos.append(0)
                else:
                    pos.append(1)
            elif ta_direc == 'short':
                unrealized_ret = 1 - (price[i] / enter_price)
                if (pre_rsi > cur_rsi and (abs(cur_rsi - buy_sig) / buy_sig) < error_tol) or (cur_rsi < buy_sig) \
                        or (unrealized_ret <= stop_loss) or (unrealized_ret >= take_profit):
                    order_status = 'no_order'
                    ta_direc = 'no_direc'
                    enter_price = None
                    pos.append(0)
                else:
                    pos.append(-1)
        
        elif order_status == 'no_order':
            if (cur_rsi >= sell_sig > pre_rsi) or (pre_rsi > sell_sig and pre_rsi > cur_rsi
                                               and (abs(cur_rsi - sell_sig) / sell_sig) <= error_tol):
                order_status = 'order_placed'
                ta_direc = 'short'
                enter_price = price[i]
                pos.append(-1)
            
            elif (cur_rsi <= buy_sig < pre_rsi) or (pre_rsi < buy_sig and pre_rsi < cur_rsi
                                                and (abs(cur_rsi - buy_sig) / buy_sig) <= error_tol):
                order_status = 'order_placed'
                ta_direc = 'long'
                enter_price = price[i]
                pos.append(1)
            else:
                order_status = 'no_order'
                pos.append(0)
    
    return np.array(pos)

def generate_sma_signals(price: np.ndarray, sma_short: np.ndarray, sma_long: np.ndarray) -> np.ndarray:
    order_status = 'no_order'
    ta_direc = 'no_direc'
    pos = [0]
    
    for i in range(1, len(price)):
        pre_sma_short = sma_short[i - 1]
        cur_sma_short = sma_short[i]
        pre_sma_long = sma_long[i - 1]
        cur_sma_long = sma_long[i]
        
        if order_status == 'order_placed':
            if ta_direc == 'long':
                if (pre_sma_long < pre_sma_short) and (cur_sma_long > cur_sma_short):
                    order_status = 'no_order'
                    ta_direc = 'no_direc'
                    pos.append(0)
                else:
                    pos.append(1)
            
            elif ta_direc == 'short':
                if (pre_sma_long > pre_sma_short) and (cur_sma_long < cur_sma_short):
                    order_status = 'no_order'
                    ta_direc = 'no_direc'
                    pos.append(0)
                else:
                    pos.append(-1)
        
        elif order_status == 'no_order':
            if (pre_sma_long < pre_sma_short) and (cur_sma_long > cur_sma_short):
                ta_direc = 'short'
                order_status = 'order_placed'
                pos.append(-1)
            elif (pre_sma_long > pre_sma_short) and (cur_sma_long < cur_sma_short):
                ta_direc = 'long'
                order_status = 'order_placed'
                pos.append(1)
            else:
                order_status = 'no_order'
                pos.append(0)
    
    return np.array(pos)
