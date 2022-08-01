import urllib.request
import importlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
from scipy import stats
import requests as re
from time import sleep
import time
import datetime as dt
from pycoingecko import CoinGeckoAPI
cg = CoinGeckoAPI()



# Working with the crypto market (NOTHING NEW HERE)
def get_market_data(gecko_id, startDate, endDate):
    dates = pd.date_range(start = startDate , end = endDate , periods = 3)
    price_df = pd.DataFrame()
    for date in range(len(dates)-1):
        start_timestamp = int(time.mktime(dt.datetime.strptime(str(dates[date])[0:10], "%Y-%m-%d").timetuple()))
        end_timestamp = int(time.mktime(dt.datetime.strptime(str(dates[date+1])[0:10], "%Y-%m-%d").timetuple()))
        test = cg.get_coin_market_chart_range_by_id(id=gecko_id,vs_currency='usd',from_timestamp=start_timestamp,to_timestamp=end_timestamp)
        timestamps = []
        for item in test['prices']:
            timestamps.append(dt.datetime.fromtimestamp(item[0]/1000))
        market_data_df = pd.DataFrame()
        market_data_df['timestamps'] = timestamps
        market_data_df['prices'] = pd.DataFrame(test['prices'][:])[1]
        market_data_df['market_caps'] = pd.DataFrame(test['market_caps'][:])[1]
        market_data_df['total_volumes'] = pd.DataFrame(test['total_volumes'][:])[1]
        market_data_df.set_index('timestamps',inplace=True)
        time_bars = market_data_df.groupby(pd.Grouper(freq='1d')).agg({'prices': 'ohlc', 
                                                                       'total_volumes': 'sum',
                                                                       'market_caps': 'last'})
        price_df = pd.concat([price_df,time_bars]) 
        time.sleep(.5)
    # Reindexing for interpolating later for missing values
    price_df_reindexed = price_df.reindex(pd.date_range(start=price_df.index.min(),
                                                  end=price_df.index.max(),
                                                  freq='1d'))
    return price_df_reindexed.interpolate(method='linear') 


# Creating multi-df for multiple coins

def create_multi_df(sec_list = ['bitcoin', 'ethereum', 'litecoin'],
                    start = "2020-06-01" , 
                    end = "2022-07-01" , ):
    """Creates a dataframe with OHLCV data from a list of the securities, reading them from the Gecko API
    "sec_list" should be a list of securities(or coins)
    "start" and "end" should be strings following the pattern: "%Y-%m-%d" or "%Y-%m-%d %H:%M:%S"
    It returns a pandas dataframe, the columns are the standard for OHLCV and the index is a multi-index:
        level_0 = "Security" , the same passed in the "sec_list"
        level_1 = "date" , datetime
    """
    multi_df = pd.DataFrame()
    first_date = pd.to_datetime(start)
    last_date = pd.to_datetime(end)
    for coin in sec_list:
        temp_df = pd.DataFrame() #to ensure an empty temporary pandas dataframe
        temp_df['close'] = get_market_data(coin, start, end)['prices']['close']
        temp_df.index.rename('date', inplace=True)
        if temp_df is None:
            continue
        temp_df["Security"] = coin #column "Security" will be used as the future level 0 index
        first_date = max(first_date , temp_df.index[0])
        last_date = min(last_date , temp_df.index[-1])
        multi_df = multi_df.append(temp_df)
    multi_df.reset_index(inplace = True)
    #limits the data to the date interval where all securities have data
    multi_df = multi_df.loc[(multi_df["date"] > first_date) & (multi_df["date"] < last_date)]
    #creates the multi-index using "Security" as level 0 and "date" as level 1
    multi_df.set_index(["Security" , "date"] , inplace = True)
    return multi_df
multi_df = create_multi_df()



def create_equal_positions_from_multidf(multi_df):
    '''
    Creates a pandas data frame with equal entries. Its necessary for some aritmetic operations in the 
    calc_rolling_profit_w_port_function function
    '''
    all_dates = multi_df.index.get_level_values('date').unique()
    all_coins = multi_df.index.get_level_values('Security').unique()
    out_df = pd.DataFrame(data = 1 , index = all_dates , columns= all_coins)
    return out_df


# THE MAIN FUNCTION!

def calc_rolling_profit_w_port_function(multi_df , screen_selection , port_function , p_func_kwargs = {} , freq = "1W" , rolling = 4 , max_sec = 15 , min_variation = 0. , info = False):
    """This function will simulate the portifolio management:
    multi_df - is the dataframe containing the OHLCV+ data from all the securities in the simulation.
    screen_selection -  is the dataframe with the selected for securities, index is date, the securities are the columns 
    and a number above 0 means security selected.
    port_function - is the function that will calculate the weights for the securities selected in the screen_selection.
    p_func_kwargs - the kwargs for the portifolio function.
    freq - is how often the portifolio will be ajusted.
    rolling - is how many 'freq' of data will be supplied to the portifolio functions. Ex.: in the defaut configurations, 
    the portifolio will be adjusted weekly and the function will use data from the 4 weeks prior to the adjustment date.
    max_sec - the maximum number os securities in the portfolio. The securities with the highest ranking (lowest number) 
    will be selected.
    min_variation - to avoid too many small adjustments in the portifolio, a minimum portifólio weights difference can be
    set;
    """
    all_dates = multi_df.index.get_level_values('date').unique() #all date with data
    readjust_dates = [all_dates[0]] + list(pd.date_range(start = all_dates[0] , end = all_dates[-1] , freq = freq)) #all dates for readjusting the portifolio
    out_weights = pd.DataFrame(index = all_dates , columns=screen_selection.columns)
    out_weights.loc[all_dates[0]] = 0 #the weights to be outputed for the first day should be zero
    last_weights = out_weights.loc[all_dates[0]] #needed for the min_variation functionality
    if info:
        original_repositioning = 0
        new_repositioning = 0
    for i in range(rolling , len(readjust_dates)):
        initial_date = readjust_dates[i - rolling]
        cur_date = readjust_dates[i]
        if (screen_selection.loc[cur_date] > 0).any():
            part_multi_df = multi_df.xs(slice(initial_date, cur_date), level=1 , drop_level = False)
            close_prices = part_multi_df.reset_index().pivot_table("close", index = "date" , columns = "Security")
            close_prices = close_prices * (screen_selection.loc[cur_date] > 0)
            close_prices = close_prices * (screen_selection.loc[cur_date] <= max_sec)
            close_prices = close_prices.loc[:, (close_prices != 0).any(axis=0)]
            cur_position = port_function(close_prices , **p_func_kwargs)
            out_weights.loc[cur_date] = cur_position
        out_weights.loc[cur_date].fillna(0 , inplace = True)
        # variation = calc_RMS(out_weights.loc[cur_date] - last_weights) #old RMS system
        variation = (out_weights.loc[cur_date] - last_weights).abs().sum()/2
        same_securities = ((out_weights.loc[cur_date] > 0) == (last_weights > 0)).all() #check if current weights and last weights have the same securities on them
        same_securities = True
        has_variation = (variation > 0)
        if has_variation & info:
            original_repositioning += 1
            new_repositioning += 1
        if (variation < min_variation) & (variation > 0) & same_securities:
            out_weights.loc[cur_date] = last_weights
            if info:
                new_repositioning -= 1
        last_weights = out_weights.loc[cur_date]
    out_weights.fillna(method="ffill" , inplace = True)
    if info:
        if original_repositioning == new_repositioning:
            print("{} repositions".format(original_repositioning))
        else:
            print("The number of repositions went from {} to {}".format(original_repositioning , new_repositioning))
    return out_weights





