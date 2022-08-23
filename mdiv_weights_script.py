import pandas as pd
import numpy as np
from time import sleep
import datetime as dt
import matplotlib.pyplot as plt
from pycoingecko import CoinGeckoAPI
cg = CoinGeckoAPI()

# Get prices from coingecko
def get_coin_price_data(gecko_id, days=30):
    test = cg.get_coin_market_chart_by_id(id=gecko_id,
                              vs_currency='usd',
                              days=days)
    timestamps = []
    for item in test['prices']:
        timestamps.append(dt.datetime.fromtimestamp(item[0]/1000))
    market_data_df = pd.DataFrame()
    market_data_df['timestamps'] = timestamps
    market_data_df['prices'] = pd.DataFrame(test['prices'][:])[1]
    market_data_df.set_index('timestamps',inplace=True)
    time_bars = market_data_df.groupby(pd.Grouper(freq='1d')).agg({'prices': 'ohlc', 
                                                                  }
                                                                 )
    return time_bars

#Creating multi_dataframe
def create_multi_df(sec_list = ['bitcoin', 'ethereum', 'litecoin'], 
                   days=300):
    """Creates a dataframe with OHLCV data from a list of the securities, reading them from the Gecko API
    "sec_list" should be a list of securities(or coins)
    It returns a pandas dataframe, the columns are the standard for OHLC and the index is a multi-index:
        level_0 = "Security" , the same passed in the "sec_list"
        level_1 = "date" , datetime
    """
    multi_df = pd.DataFrame()
    for coin in sec_list:
        temp_df = pd.DataFrame() #to ensure an empty temporary pandas dataframe
        temp_df['close'] = get_coin_price_data(coin, days=days)['prices']['close']
        temp_df.index.rename('date', inplace=True)
        if temp_df is None:
            continue
        temp_df["Security"] = coin #column "Security" will be used as the future level 0 index
        first_date = temp_df.index[0]
        last_date = temp_df.index[-1]
        multi_df = multi_df.append(temp_df)
    multi_df.reset_index(inplace = True)
    #limits the data to the date interval where all securities have data
    multi_df = multi_df.loc[(multi_df["date"] > first_date) & (multi_df["date"] < last_date)]
    #creates the multi-index using "Security" as level 0 and "date" as level 1
    multi_df.set_index(["Security" , "date"] , inplace = True)
    return multi_df

def create_equal_positions_from_multidf(multi_df):
    '''
    Creates a pandas data frame with equal entries. Its necessary for some aritmetic operations in the 
    calc_rolling_profit_w_port_function function
    '''
    all_dates = multi_df.index.get_level_values('date').unique()
    all_coins = multi_df.index.get_level_values('Security').unique()
    out_df = pd.DataFrame(data = 1 , index = all_dates , columns= all_coins)
    return out_df

# Calculates profit
def calc_rolling_profit_w_port_function(multi_df , 
                                        screen_selection , 
                                        port_function , 
                                        p_func_kwargs = {} ,
                                        freq = "1W" , 
                                        rolling = 4 ,
                                        max_sec = 15 , 
                                        min_variation = 0. ,
                                        info = False):
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

#Calculating portfolio weights
def get_weights_including_stable(portfolio_with_nonstables,
                                 portfolio_with_stables,
                                 stable_min_weight = .2,
                                 given_port_strat_position_proportion_df = pd.DataFrame() #= position_proportion_df 
                                 ):
    """
    Generates new historical weights for a portfolio strategy that uses volatility in its formulation, but including
    stable coins.
    """
    new_weights_df = pd.DataFrame()
    new_weights_df['dai'] = stable_min_weight/(1-stable_min_weight)+(1-given_port_strat_position_proportion_df.sum(axis=1))
    non_stable_assets = portfolio_with_nonstables['close'].index.get_level_values('Security').unique()
    for sec in list(non_stable_assets):
        new_weights_df[sec] = given_port_strat_position_proportion_df[sec]

    new_weights_df = new_weights_df.divide(new_weights_df.sum(axis=1), axis=0)
    return new_weights_df

def get_weights_MDiv(df, risk_target = 1):
    prices = df
    returns = prices.pct_change()
    # We calculate the covariance matrix
    covariances = returns.iloc[1:, :].cov() 
    assets = prices.columns
    assets_volatilities = np.array(returns.describe().loc['std']) # Denoted as Sigma
    # uses weights as w = Cov**(-1) . Sigma / (Sigma' . Cov**(-1) . Sigma)
    w = np.linalg.inv(covariances) @ assets_volatilities / (assets_volatilities.T @  np.linalg.inv(covariances) @ assets_volatilities)  
    w_pd = pd.Series(w, index=assets, name='weights') 
    w_long = w_pd >= 0 # Just long positions (shorts are negative)
    w_pd.loc[~w_long] = 0 
    w_unitary = w_pd/np.sum(w_pd) # Sum of weights equals 1 -> Without saving cash
    # We convert the weights to a matrix
    weights = np.matrix(w_unitary)
    # We calculate the risk of the weights distribution
    portfolio_risk = float(np.sqrt((weights * covariances.values * weights.T)))  
    if portfolio_risk > risk_target:
        weights = w_unitary * risk_target / portfolio_risk # Sum of weights less than 1
    else:
        weights = w_unitary
    return weights


def job():
    '''
    Executes everything that is necessary
    '''
    # Creating dataframes with assets' prices.
    #With stable coin.
    multi_df = create_multi_df(sec_list = ['bitcoin', 'ethereum', 'dai'],
                        days=100)
    # Creating positions withou stablecoins
    position_df = create_equal_positions_from_multidf(multi_df.loc[['bitcoin','ethereum']])

    # Creating the Maximum Diversification strategy's weights without including stablecoin.
    position_proportion_MDiv_df = calc_rolling_profit_w_port_function(multi_df.loc[['bitcoin','ethereum']] , 
                                                                 position_df ,
                                                                 get_weights_MDiv , 
                                                                 p_func_kwargs={'risk_target': 0.029},
                                                                 freq = "d" , #Rebalance frequency 
                                                                 rolling = 30) # Get_weights é p ser a port_weights


    position_proportion_df_with_stable = get_weights_including_stable(portfolio_with_nonstables = multi_df.loc[['bitcoin','ethereum']],
                                     portfolio_with_stables = multi_df,
                                     stable_min_weight = .2,
                                     given_port_strat_position_proportion_df = position_proportion_MDiv_df)

    # Reindexing to make the columns order correct
    non_stable_assets = multi_df.index.get_level_values('Security').unique()
    position_proportion_df_with_stable_biweekly_mean = position_proportion_df_with_stable.rolling(15).mean()
    weights_series = .95*position_proportion_df_with_stable_biweekly_mean.iloc[-1]
    #Fixing KACY as 5%.
    weights_series['kacy'] = .05
    return dict(weights_series)

#job()