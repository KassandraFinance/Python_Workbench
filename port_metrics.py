import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mtick
#Portfolio functions for performance analysis


def scale_data(data):
    """Scale the data in each column between 0 and 1
    """
    out_data = ((data - data.min(axis = 0))/
                (data.max(axis = 0) - data.min(axis = 0))
               )
    return out_data

def unscale_data(data , original):
    """Unscale the data in each column, back to a scale equivalent to the original.
    """
    out_data = (original.max(axis = 0) - original.min(axis = 0))*data + original.min(axis = 0)
    return out_data

def from_ret_to_log(return_df):
    """Converts normal returns to logaritmic returns
    """
    return  np.log1p(return_df)

def from_log_to_ret(return_df):
    """Converts logaritmic returns to normal returns
    """
    return np.expm1(return_df)

def logdiff_2_pctchange(df):
    """Converts columns "bench_returns" , "st_returns" and "cum_st_returns" from logaritmic returns to normal returns
    """
    df2 = df.copy()
    df2["bench_returns"] = np.expm1(df["bench_returns"])
    df2["st_returns"] = np.expm1(df["st_returns"])
    df2["cum_st_returns"] = np.expm1(df["cum_st_returns"])
    return df2

def pctchange_2_logdiff(df):
    """Converts columns "bench_returns" , "st_returns" and "cum_st_returns" from normal returns to logaritmic returns
    """
    df2 = df.copy()
    df2["bench_returns"] = np.log1p(df["bench_returns"])
    df2["st_returns"] = np.log1p(df["st_returns"])
    df2["cum_st_returns"] = np.log1p(df["cum_st_returns"])
    return df2

def calc_volatility(returns_series):
    entries = returns_series.shape[0]
    std = returns_series.std()
    return std * entries ** (1/2)

def calc_max_drawdown_lenght(drawdown_df):
    """Calculates the maximum drawdown lenght of all the columns of a dataframe.
    
    The data should be the drawdown, either in returns, logaritmic returns or price.
    The function returns a series with the result for each column of the initial dataframe
    """
    out_df = drawdown_df.copy()
    for colname , col in out_df.iteritems():
        # God knows how this works, but it does and it is much faster than a loop.
        temp = (col == 0)
        out_df[colname] = (1-temp).groupby(temp.cumsum()).cumsum()
    return out_df.max(axis = 0)

def calc_RMS(series):
    """Returns the RMS of the pandas series given as input
    """
    
    result = series.copy()
    result = result**2
    result = result.mean()
    result = result**(1/2)
    return result

def round_number_to_sig_digits(number, digits = 10):
    """Round a number to a significant digits
    """
    return round(number, digits - int(math.floor(math.log10(abs(number)))) - 1)

def calculate_profit(df ,
                     SL = 0 ,
                     SG = 0 ,
                     exit_s = False ,
                     end_s = True ,
                     slippage = 0 ,
                     comission = 0 ,
                     plot = False):
    """Calculates the profit of a strategy
    
    df - shoulf be a pandas dataframe with OHLCV data (volume is not necessary) plus a 'signal' column.
        This 'signal' column should contain:
        1 - for buy long
        0.5 - for sell short
        0 - for stay as it is
        -0.5 - for sell long
        -1 - for buy short
    SL - Stop Loss, where '0.02' equals to '2%' and '0' means 'do not used Stop Loss'
    SG - Stop Gain, the same as above
    exit_s - if the sell signal from the strategy will be used. If False, the strategy buys and stop gain and stop loss sell
    end_s - sell everything at the end
    slippage - expected difference between the close price and the actual buy price. Works the same as Stop Loss
    commission - commission taken by the exchange in both buy and sell operations. Works the same as Stop Loss
    plot - if a graph shoul be ploted.
    
    There are two outputs:
        PnL_df - is a list of all the times the strategy was positioned, containg:
            "Type" (long or short) , 'Entry_date', 'Entry_price', 'Exit_date', 'Exit_Price'
            'PnL%' (Pnl in %), 'PnL' (PnL in value), 'Exit' (reason to exit: Strategy, StopLoss or Stop Gain)
        new_df - is a dataframe with the same index as the input 'df', containg:
            "pos" - 1 when long, -1 when short and 0 when not positioned
            "bench_price" and "bench_returns" - the close price and return of the security being traded
            "st_returns"  , "cum_st_returns" , "st_price" - the return, accumulated return and equivalent price of the strategy
        *all returns are log returns

    """
    ##Checking the use of stop-gain and stop-loss
    SL_s = SL > 0
    SG_s = SG > 0
    ## Initializing data    
    PnL = []
    df2 = df.copy()
    exit_date = {"End" : df2.index[-1]}
    ## LOOP, only if there is a buy signal in the dataframe
    df2.loc[df2.index[-1] , "signal"] = 0
    if df2["signal"].abs().max() == 1:
        n_done = True
    else:
        n_done = False
    while n_done:
        init_date = df2.loc[df2["signal"].abs() == 1].index[0] # find the first "buy"
        df2 = df2.loc[init_date:]
        df2 = df2.iloc[1:]
        exit_date["Stop Loss"] = df2.index[-1]
        exit_date["Stop Gain"] = df2.index[-1]
        exit_date["Strategy"] = df2.index[-1]
        #For LONG
        if df.loc[init_date , "signal"] >= 1:
            init_price = df.loc[init_date , "close"] * (1 + slippage) * (1 + comission)
            trade_type = "Long"
            #Check for STOP LOSS
            if SL_s:
                SL_price = df.loc[init_date , "close"] * (1 - SL) * (1 + slippage)
                if SL_price > df2["low"].min():
                    exit_date["Stop Loss"] = df2.loc[df2["low"] <= SL_price].index[0]
            #Check for STOP GAIN
            if SG_s:
                SG_price = df.loc[init_date , "close"] * (1 + SG) * (1 + slippage)
                if SG_price < df2["high"].max():
                    exit_date["Stop Gain"] = df2.loc[df2["high"] >= SG_price].index[0]
            #Check for Strategy Exit
            if exit_s:  
                if df2["signal"].min() <= -0.5:
                    exit_date["Strategy"] = df2.loc[df2["signal"] <= -0.5].index[0]
            #Find de quickest exit
            exit_reason = min(exit_date , key=exit_date.get)
            #Check if the quickest exit happend before the end
            if exit_date[exit_reason] != df2.index[-1]:
                if exit_reason == "Stop Loss":
                    exit_price = SL_price * (1 - comission)
                elif exit_reason == "Stop Gain":
                    exit_price = SG_price * (1 - comission)
                else:
                    exit_price = df2.loc[exit_date[exit_reason] , "close"] * (1 - comission)
                ret_p = np.log(exit_price / init_price)
                ret = exit_price - init_price
            #or if does, if there is the end_s flag
            elif end_s:
                n_done = False
                exit_price = df2.loc[exit_date[exit_reason] , "close"] * (1 - comission)
                ret_p = np.log(exit_price / init_price)
                ret = exit_price - init_price
            #otherwise, just exit without concluding the position
            else:
                n_done = False
                exit_price = np.nan
                ret_p = np.nan
                ret = np.nan
        #For SHORT
        elif df.loc[init_date , "signal"] <= -1:
            init_price = df.loc[init_date , "close"] * (1 - slippage) * (1 - comission)
            trade_type = "Short"
            #Check for STOP LOSS
            if SL_s:
                SL_price = df.loc[init_date , "close"] / (1 - SL) * (1 - slippage)
                if SL_price < df2["high"].max():
                    exit_date["Stop Loss"] = df2.loc[df2["high"] >= SL_price].index[0]
            #Check for STOP GAIN
            if SG_s:
                SG_price = df.loc[init_date , "close"] / (1 + SG) * (1 - slippage)
                if SG_price > df2["low"].min():
                    exit_date["Stop Gain"] = df2.loc[df2["low"] <= SG_price].index[0]
            #Check for Strategy Exit
            if exit_s:  
                if df2["signal"].max() >= 0.5:
                    exit_date["Strategy"] = df2.loc[df2["signal"] >= 0.5].index[0]
            #Find de quickest exit
            exit_reason = min(exit_date , key=exit_date.get)
            #Check if the quickest exit happend before the end
            if exit_date[exit_reason] != df2.index[-1]:
                if exit_reason == "Stop Loss":
                    exit_price = SL_price * (1 + comission)
                elif exit_reason == "Stop Gain":
                    exit_price = SG_price * (1 + comission)
                else:
                    exit_price = df2.loc[exit_date[exit_reason] , "close"] * (1 + comission)
                ret_p = np.log(init_price / exit_price)
                ret = init_price - exit_price
            #or if does, if there is the end_s flag
            elif end_s:
                n_done = False
                exit_price = df2.loc[df2.index[-1] , "close"] * (1 + comission)
                ret_p = np.log(init_price / exit_price)
                ret = init_price - exit_price
            #otherwise, just exit without concluding the position
            else:
                n_done = False
                exit_price = np.nan
                ret_p = np.nan
                ret = np.nan
        #Append the positon's data to PnL2
        PnL.append([trade_type , init_date, init_price, exit_date[exit_reason], exit_price, ret_p, ret, exit_reason])
        df2 = df2.loc[exit_date[exit_reason]:]
        if df2["signal"].abs().max() < 1:
            n_done = False
            
    #Create the final Dataframes
    PnL_df = pd.DataFrame(PnL , columns =["Type" , 'Entry_date', 'Entry_price', 'Exit_date', 'Exit_Price', 'PnL%', 'PnL', 'Exit'])
    new_df = pd.DataFrame(index = df.index)
    new_df["pos"] = np.nan
    new_df.loc[df.index[0] , "pos"] = 0
    new_df.loc[PnL_df['Exit_date'] , "pos"] = 0
    new_df.loc[PnL_df.loc[PnL_df["Type"]== "Long" , 'Entry_date'] , "pos"] = 1
    new_df.loc[PnL_df.loc[PnL_df["Type"]== "Short" , 'Entry_date'] , "pos"] = -1
    new_df["pos"].fillna(method = "ffill" , inplace = True)
    new_df["pos"] = new_df["pos"].shift(1 , fill_value=0)
    new_df["bench_price"] = df["close"]
    new_df["bench_returns"] = np.log(df["close"] / df["close"].shift(1))
    new_df.loc[df.index[0] , "bench_returns"] = 0
    
    temp = new_df["bench_price"]
    temp.loc[PnL_df['Entry_date']] = list(PnL_df['Entry_price'])
    temp.loc[PnL_df['Exit_date']] = list(PnL_df['Exit_Price'])
    temp =  np.log(temp / temp.shift(1))
    temp.iloc[0] = 0
    new_df["st_returns"] = temp * new_df["pos"]
    new_df["cum_st_returns"] = new_df["st_returns"].cumsum()
    new_df["st_price"] = df["close"].iloc[0] * np.exp(new_df["cum_st_returns"])
    
    if plot:
        text_1 = ("From {} to {}\n".format(df.index[0] , df.index[-1]))
        
        profit = PnL_df["PnL%"].sum()
        market = np.log(df["close"].iloc[-1] / df["close"].iloc[0])
        
        text_2 = ("Profit: {0:.1%} (ln)\n".format(profit))
        text_3 = ("Market: {0:.1%} (ln)\n".format(market))
        print(text_1 + text_2 + text_3)
        
        fig, ax = plt.subplots(figsize = (16 , 6) , dpi = 120)       
        ax.plot(df["close"], linewidth=1, color='black', alpha = 0.8 , label='Close price')
        ax.plot(new_df["st_price"] , linewidth=1, color='blue', alpha = 0.8 , label='Result')
        plot_date = PnL_df.loc[PnL_df['Exit'] == "Strategy" , 'Exit_date']
        plot_price = df.loc[plot_date , "close"]
        ax.scatter(plot_date , plot_price , label = 'Signal Sell', color = 'yellow', alpha = 0.8 , s = 100, marker = "8")
        plot_date = PnL_df.loc[PnL_df['Exit'] == "Stop Loss" , 'Exit_date']
        plot_price = df.loc[plot_date , "close"]
        ax.scatter(plot_date , plot_price , label = 'Stop Loss sell', color = 'red', alpha = 0.8 , s = 100, marker = "8")
        plot_date = PnL_df.loc[PnL_df['Exit'] == "Stop Gain" , 'Exit_date']
        plot_price = df.loc[plot_date , "close"]
        ax.scatter(plot_date , plot_price , label = 'Stop Gain sell', color = 'green', alpha = 0.8 , s = 100, marker = "8")
        plot_date = PnL_df.loc[PnL_df['Type'] == "Long" , "Entry_date"]
        plot_price = df.loc[plot_date , "close"]
        ax.scatter(plot_date , plot_price , label = 'Buy Long', color = 'blue', alpha = 0.8 , s = 100, marker = "^")
        plot_date = PnL_df.loc[PnL_df['Type'] == "Short" , "Entry_date"]
        plot_price = df.loc[plot_date , "close"]
        ax.scatter(plot_date , plot_price , label = 'Buy Short', color = 'cyan', alpha = 0.8 , s = 100, marker = "v")
        for key, spine in ax.spines.items():
            spine.set_visible(False)
        ax.tick_params(axis='both', which='both', top = False , bottom=False, left=False)
        ax.legend(frameon=False)
        plt.show()
        
#         show_stats_table(data_dic = {"Tested":(PnL_df , new_df) })
#         plot_hist_moving_ab_table(df, plot = True)
#         plot_return_historgram_and_PDF(new_df , colormap = "cividis")
#         montly_return_table(new_df["st_returns"] , table_name="Strategy")
#         montly_return_table(new_df["bench_returns"] , table_name="Market")
    return PnL_df , new_df



def show_stats_table(data_dic = {"BTC":() , #should a tuple (PnL , df)
                                 "ETH":()
                                } , 
                                format_table = True ,
                                display_table = True , 
                                simplify = True , 
                                min_date = "1950-01-01" , #in case data must be aligned
                                max_date = "2050-01-01" , #in case data must be aligned
                                conf_interval = .7 , #to remove outliers from the statistical data (mean, std, skewness and kurtosis)
                                ):
    """Shows statistics table for multiple data

    'data_dic' is a dictionary containing:
        {'data name 1' : (Pnl_1 , df_1) , 'data name 2':  (Pnl_2 , df_2) , ..... }
        where 'PnL_x' and 'df_x' are the output of the 'calculate_profit' function
    'format_table' formats the table for easier human undestanding. If returned, the table will contain strings only
    'display_table' defines if the table should be displayed (True) or returned (False)
    'simplfy' removes the rows where all values are zero and removes the 'market' column if the previous 'market' contains the same results.
    'min_date' & 'max_date' limits the data, before calculating the stats
    'conf_interval', in order to have meaningful statistics about the return (Ret Mean, std, skeness..), the outliers must be removed. This parameter defines the interval to consider the data as valid.
    
    The stats are divide in three groups:
    - Econometrical
        - Return (ln), Return (%), Top Return (%), Annual Return, Daily Volatility, Downside Deviation, Max Drawdown, Average Drawdown, Max Drawdown Length, Sharpe, Sortino, Calmar, Momersion
        - The volatility stats (Daily Volatility, Downside Deviation) only consider the periods when the return is not zero (strategy positioned). If the input data is no daily, it will be resampled to daily before this calculations
    - Statistical
        - Ret Mean, Ret Std, Ret Skewness, Ret Kurtosis
        - It only consider the periods when the return is not zero (strategy positioned)
        - There is no resampling of the data, if the input is daily returns, the results will be daily, if the input is hourly returns, the results will be hourly stats
    - Trades
        - Trades (number), Accuracy , Best Win, Worst Loss, Average Win, Average Loss, Holding Time
    """
    # Creates the table
    stats_table_columns = pd.MultiIndex.from_product([data_dic.keys(), ['Strategy', 'Market']])
    stats_table = pd.DataFrame(columns = stats_table_columns)
    last_entry = 0

    for entry in data_dic:
        # Separetes 'PnL' and 'df', and creates a 'df' using non-log returns
        e_Pnl = data_dic[entry][0]
        e_Pnl = e_Pnl.loc[(e_Pnl["Entry_date"]> min_date) & (e_Pnl["Entry_date"] < max_date)]
        e_df = data_dic[entry][1]
        e_df = e_df.loc[min_date:max_date]
        e_df_per = logdiff_2_pctchange(e_df)
        # Group strategy data with market data
        ret_log = pd.concat([e_df["st_returns"] , e_df["bench_returns"]] , axis = 1)
        ret_per = pd.concat([e_df_per["st_returns"] , e_df_per["bench_returns"]] , axis = 1)
        # Resample the data to daily. Some statistics require daily information.
        daily_return = ret_log.resample("D").sum().fillna(method="pad")
        # daily_return_per = from_log_to_ret(daily_return) 
        # Initial calculations
        total_log_cum_ret = ret_log.cumsum(axis = 0)
        drawdown = total_log_cum_ret - total_log_cum_ret.cummax()
        drawdown = from_log_to_ret(drawdown)
        # Creates the columns for the current entry
#         stats_table.loc["Return (ln)" , (entry , slice(None))] = ret_log.sum(axis = 0).values
#         stats_table.loc["Return (%)" , (entry , slice(None))] = np.expm1(stats_table.loc["Return (ln)" , (entry , slice(None))].astype("float64"))
#         stats_table.loc["Top Return (%)" , (entry , slice(None))] = np.expm1(ret_log.cumsum(axis = 0).max()).values
        annual_return = from_log_to_ret(daily_return.mean() * 365)
        stats_table.loc["Annual Return" , (entry , slice(None))] = annual_return.values
        daily_return_pos = daily_return.replace(0,np.nan)
        d_vol = daily_return_pos.std(ddof = 0)
        stats_table.loc["Daily Volatility" , (entry , slice(None))] = d_vol.values
        dd = daily_return_pos.mask(daily_return_pos >0 , np.nan).std(ddof = 0)
        stats_table.loc["Downside Deviation" , (entry , slice(None))] = dd.values
        max_drawdown = drawdown.min()
        stats_table.loc["Max Drawdown" , (entry , slice(None))] = max_drawdown.values
#         stats_table.loc["Average Drawdown" , (entry , slice(None))] = drawdown.mean().values
        stats_table.loc["Max Drawdown Length" , (entry , slice(None))] = calc_max_drawdown_lenght(drawdown).values
        stats_table.loc["Sharpe", (entry , slice(None))] = (annual_return / (d_vol.where(d_vol != 0) * 365**0.5)).values
        stats_table.loc["Sortino", (entry , slice(None))] = (annual_return / dd.where(dd!=0) ).values
        stats_table.loc["Calmar", (entry , slice(None))] = (-annual_return / max_drawdown.where(max_drawdown != 0)).values
#         stats_table.loc["Momersion" , (entry , slice(None))] = (ret_per > 0).where(ret_per.notna()).mean(skipna = True).values
        #Statistical Results
        z_simetrical = stats.norm.ppf(conf_interval/2 +0.5)
        clean_ret_log = ret_log.where((ret_log != 0) & (ret_log.abs() < z_simetrical * ret_log.std(axis = 0)))
        stats_table.loc["Ret Mean" , (entry , slice(None))] = np.expm1(clean_ret_log.mean(axis = 0)).values
        stats_table.loc["Ret Std" , (entry , slice(None))] = np.expm1(clean_ret_log.std(axis = 0)).values
        stats_table.loc["Ret Skewness" , (entry , slice(None))] = np.expm1(clean_ret_log.skew(axis = 0)).values
        stats_table.loc["Ret Kurtosis" , (entry , slice(None))] = np.expm1(clean_ret_log.kurtosis(axis = 0)).values
        #Trade Results
        stats_table.loc["Trades" , (entry , 'Strategy')] = e_Pnl.shape[0]
        stats_table.loc["Accuracy" , (entry, 'Strategy')] = 0
        trade_n = e_Pnl.shape[0]
        if trade_n > 0: #to avoid a division by 0
            stats_table.loc["Accuracy" , (entry , 'Strategy')] = e_Pnl.loc[e_Pnl["PnL%"] > 0].shape[0] / trade_n
        stats_table.loc["Best Win" , (entry , 'Strategy')] = np.expm1(e_Pnl["PnL%"].max())
        stats_table.loc["Worst Loss" , (entry , 'Strategy')] = np.expm1(e_Pnl["PnL%"].min())
        stats_table.loc["Average Win" , (entry , 'Strategy')] = np.expm1(e_Pnl["PnL%"].loc[e_Pnl["PnL%"] > 0]).mean()
        stats_table.loc["Average Loss" , (entry , 'Strategy')] = np.expm1(e_Pnl["PnL%"].loc[e_Pnl["PnL%"] < 0]).mean()
        stats_table.loc["Holding Time" , (entry , 'Strategy')] = (e_df.loc[e_df["pos"] != 0 , "pos"].count() /
                                                                e_df["pos"].count())
        #Removes the previous "Market" if the current "Market" is the same
        if (last_entry != 0) & simplify:
            if stats_table.loc[:, (entry, 'Market')].equals(stats_table.loc[:, (last_entry, 'Market')]):
                stats_table.drop(columns = (last_entry, "Market"), inplace = True)
        last_entry = entry
    stats_table.loc["APPT", (slice(None), 'Strategy')] = stats_table.loc['Accuracy', (slice(None), 'Strategy')]\
                                                    *stats_table.loc['Average Win', (slice(None), 'Strategy')]\
                                                    /((1-stats_table.loc['Accuracy', (slice(None), 'Strategy')])\
                                                     *stats_table.loc['Average Loss', (slice(None), 'Strategy')].abs())
    # Fill the data that should be empty with "-"
    stats_table.loc[["Accuracy" , 
                     "Trades" , 
                     "Best Win" ,
                     "Worst Loss" ,
                     "Average Win" ,
                     "Average Loss" ,
                     "Holding Time" , 
#                      "APPT"
                    ] ,
                    (slice(None), 'Market')] = "-"
    # Fill missing data with 0
    stats_table.fillna(0 , inplace = True)
    # Formats the statistics table
    if format_table:
        stats_table = format_stats_table(stats_table)
    
    # Remove Trade rows if all unused
    if simplify:
        rows_w_all_zeros = (stats_table.loc[:, (slice(None), 'Strategy')] == "0.0%").all(axis = 1)
        stats_table.drop(index = stats_table.index[rows_w_all_zeros] , inplace = True)
    
    if display_table:
        market_slice = stats_table.columns[stats_table.columns.get_level_values(1) == "Market"]
        html_render = (stats_table.style
                .set_properties(**{"color":"gray"} , subset = market_slice)
                .set_properties(**{'text-align': 'center'})
                .set_table_styles([dict(selector="th" , props=[("text-align", "center")])])
        )
        display(html_render)
        # fig = z_datavis.set_table(stats_table)
        # z_datavis.fig_plot(fig)
        return None
    else:
        return stats_table


def format_stats_table(table):
    stats_table = table.copy()
    percent_style_1 = [#"Return (ln)" ,
#                        "Return (%)" ,
                       #"Top Return (%)" ,
                       "Annual Return" , 
                       "Daily Volatility" , 
                       #"Downside Deviation" , 
                       "Ret Std" , 
                       "Max Drawdown" ,
                       #"Average Drawdown" ,
                       #"Momersion"
    ]
    percent_style_2 = ["Ret Mean"]
    percent_style_3 = ["Accuracy" ,
                       "Best Win" , 
                       "Worst Loss" ,
                       "Average Win" ,
                       "Average Loss" ,
                       "APPT",
                       "Holding Time"]
    float_style = ["Calmar" ,
                   "Sortino" , 
                   "Sharpe" ,
                   "Ret Skewness" ,
                   "Ret Kurtosis"]
    int_style = ["Max Drawdown Length" , "Trades"]
    securities = stats_table.columns.get_level_values(0).unique()
    strategy_slice = pd.MultiIndex.from_product([securities, ['Strategy']])
    stats_table.loc[percent_style_1] = stats_table.loc[percent_style_1].applymap(lambda x:"{:.1%}".format(x))
    stats_table.loc[percent_style_2] = stats_table.loc[percent_style_2].applymap(lambda x:"{:.3%}".format(x))
    stats_table.loc[percent_style_3 , strategy_slice] = stats_table.loc[percent_style_3 , strategy_slice].applymap(lambda x:"{:.1%}".format(x))
    stats_table.loc[float_style] = stats_table.loc[float_style].astype("float").applymap(lambda x:"{:.3}".format(x))
    stats_table.loc[int_style , strategy_slice] = stats_table.loc[int_style , strategy_slice].astype("int" , errors = "ignore")
    return stats_table
    
def calculate_portfolio_profit(multi_df , 
                                position_proportion_df ,
                                SL = 0 ,
                                SG = 0 ,
                                slippage = 0 ,
                                comission = 0 ,
                                trade_type = "long" , #(long , short , both)
                                plot = False):
    if (slippage != 0) or (comission != 0):
        print("The results do not include Slippage nor commission discounts, as those are not yet implemented ", 
              "in the portifolio profit function.")
    data_dic = {"Portifolio":None}
    port_df = pd.DataFrame(0 , index = multi_df.index.get_level_values('date').unique() , columns=["st_returns" , 
                                                                                                   "bench_returns" ,
                                                                                                   "cum_st_returns" ,
                                                                                                   "pos"])
    repositions_dates = (position_proportion_df != position_proportion_df.shift(1)).any(1)
    repositions_dates = position_proportion_df.loc[repositions_dates].index[1:]
    port_PnL = pd.DataFrame(0 , index = list(range(repositions_dates.shape[0])) , columns=["Entry_date" , "PnL%"])
    port_PnL["Entry_date"] = repositions_dates
    sec_list = multi_df.index.get_level_values('Security').unique()
    sec_number = len(sec_list)
    for sec in sec_list:
        df = multi_df.loc[sec]
        signal = (position_proportion_df[sec] > 0).astype("int").diff()
        signal = signal.replace(-1 , -0.5)# valid only for "long"
        df["signal"] = signal
        PnL_df , df_new = calculate_profit(df ,
                                        SL = SL ,
                                        SG = SG ,
                                        exit_s = True ,
                                        end_s = True ,
                                        slippage = 0 ,
                                        comission = 0 ,
                                        plot = False)
        # st_returns = from_ret_to_log(from_log_to_ret(df_new["st_returns"] * position_proportion_df[sec].shift(1)))
        # cum_st_returns = st_returns.cumsum()
        # bench_returns = from_ret_to_log(from_log_to_ret(df_new["bench_returns"] * 1/sec_number))
        # port_df["st_returns"] = port_df["st_returns"] + st_returns
        # port_df["cum_st_returns"] = port_df["cum_st_returns"] + cum_st_returns
        # port_df["bench_returns"] = port_df["bench_returns"] + bench_returns
        st_returns = from_log_to_ret(df_new["st_returns"]) * position_proportion_df[sec].shift(1)
        bench_returns = from_log_to_ret(df_new["bench_returns"]) * 1/sec_number
        port_df["st_returns"] = port_df["st_returns"] + st_returns
        port_df["bench_returns"] = port_df["bench_returns"] + bench_returns
        port_df.loc[df_new["pos"] == 1 , "pos"] = 1
        data_dic[sec] = (PnL_df , df_new)
    port_df["st_returns"] = from_ret_to_log(port_df["st_returns"])
    port_df["cum_st_returns"] = port_df["st_returns"].cumsum()
    port_df["bench_returns"] = from_ret_to_log(port_df["bench_returns"])

    data_dic["Portifolio"] = (port_PnL , port_df)
    if plot:
        show_stats_table(data_dic)
        f, ax = plt.subplots(2, 1, sharex = True ,
                                           figsize = (16 , 9) , 
                                           gridspec_kw={'height_ratios': [3 , 1]} , 
                                          dpi = 120)
        colors = plt.get_cmap("nipy_spectral" , 512)
        colors = ListedColormap(colors(np.linspace(0.1, 0.9, 256)))
      
        ax[0].set_title("Cumulated Return")    
        ax[0].set_prop_cycle(color = colors(np.linspace(0 , 1 , sec_number)))
        for sec in sec_list:
            plot_data = data_dic[sec][1]["bench_price"]/data_dic[sec][1]["bench_price"][0]-1
            ax[0].plot(plot_data, label = "Gains with B&H of " + sec)
        plot_data = from_log_to_ret(data_dic["Portifolio"][1]["cum_st_returns"])
        ax[0].plot(plot_data , c = "black" , label = "Gains of the full portifolio")
        ax[0].axhline(color = "black" , alpha = 0.1)
        # Signal
        ax[1].set_title("Portifolio Composition") 
        ax[1].set_prop_cycle(color = colors(np.linspace(0 , 1 , sec_number)))
        ax[1].stackplot(position_proportion_df.index  , position_proportion_df.T , labels = sec_list)
        # ax[1].plot(position_proportion_df)
        for i in [0,1]:
            percent_tick = mtick.StrMethodFormatter('{x:+.0%}')
            ax[i].yaxis.set_major_formatter(percent_tick)
            for key, spine in ax[i].spines.items():
                spine.set_visible(False)
            column_number = round(len(sec_list)/15 + 0.5)
            ax[i].legend(frameon=False , ncol = column_number)
            ax[i].tick_params(axis='both', which='both', bottom=False, left=False)

    return data_dic


