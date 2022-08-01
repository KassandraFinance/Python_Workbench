import pandas as pd
import numpy as np
import requests as re
from time import sleep
import datetime as dt
from datetime import timezone
import scipy.optimize as sc
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns


#PORTFOLIO METRICS
def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns*weights)*365
    std = np.sqrt(
            np.dot(weights.T,np.dot(covMatrix, weights))
           )*np.sqrt(365)
    return returns, std

def portfolioVariance(weights, meanReturns, covMatrix): #It's the portfolio's STD (Volatility) instead of VARIANCE! 
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]

def portfolioReturn(weights, meanReturns, covMatrix):
        return portfolioPerformance(weights, meanReturns, covMatrix)[0]

#PORTFOLIO STRATEGIES
def maximum_Sharpe(meanReturns, covMatrix, riskFreeRate = 0, constraintSet=(0,1)):
    "Minimize the negative SR, by altering the weights of the portfolio"
    #We'll define the negative Sharpe Ratio for minimize its' function, that is, maximize it.
    #PS: that is because scipy.optimize works only with minimization.
    def negativeSR(weights, meanReturns, covMatrix, riskFreeRate = 0):
        pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix)
        return - (pReturns - riskFreeRate)/pStd
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate) #Or mi, sigma and rf respectively (from SR:=(mi-rf)/sigma)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) #All the weights sum up to 1.
    bound = constraintSet #Numeric Interval
    bounds = tuple(bound for asset in range(numAssets)) #N-dim cube
    result = sc.minimize(negativeSR, #Function to be minimized
                         numAssets*[1./numAssets], #Initial guess (equal weights portfolio)
                         args=args,
                         method='SLSQP',
                         bounds=bounds, 
                         constraints=constraints)
    return result

def minimum_variance(meanReturns, covMatrix, constraintSet=(0,1)):
    """Minimize the portfolio variance by altering the 
     weights/allocation of assets in the portfolio"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) #Same here
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(portfolioVariance,
                         numAssets*[1./numAssets],
                         args=args,
                         method='SLSQP', 
                         bounds=bounds, 
                         constraints=constraints)
    return result

#STRATEGY WITH TARGET RETURNS
def efficientOpt(meanReturns, covMatrix, returnTarget, constraintSet=(0,1)):
    """For each returnTarget, we want to optimise the portfolio for min variance"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    #We'll set the target return on the constraints below
    constraints = ({'type':'eq', 'fun': lambda x: portfolioReturn(x, meanReturns, covMatrix) - returnTarget}, 
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    effOpt = sc.minimize(portfolioVariance, 
                         numAssets*[1./numAssets], 
                         args=args, 
                         method = 'SLSQP', 
                         bounds=bounds, 
                         constraints=constraints)
    return effOpt

#STRATEGY'S RETURNS
def calculatedResults(meanReturns, covMatrix, riskFreeRate, constraintSet=(0,1)):
    """Read in mean, cov matrix, and other financial information
        Output, Max SR , Min Volatility, efficient frontier """
    # Max Sharpe Ratio Portfolio
    maxSR_Portfolio = maximum_Sharpe(meanReturns, 
                                     covMatrix, 
                                     riskFreeRate=riskFreeRate, 
                                     constraintSet=constraintSet)
    maxSR_returns, maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], 
                                                    meanReturns,
                                                    covMatrix)
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'], 
                                    index=meanReturns.index, 
                                    columns=['allocation'])
    maxSR_allocation.allocation = [round(i*100,0) for i in maxSR_allocation.allocation]
    
    # Min Volatility Portfolio
    minVol_Portfolio = minimum_variance(meanReturns, 
                                        covMatrix, 
                                        constraintSet=constraintSet)
    minVol_returns, minVol_std = portfolioPerformance(minVol_Portfolio['x'], 
                                                      meanReturns,
                                                      covMatrix)
    minVol_allocation = pd.DataFrame(minVol_Portfolio['x'],
                                     index=meanReturns.index,
                                     columns=['allocation'])
    minVol_allocation.allocation = [round(i*100,0) for i in minVol_allocation.allocation]

    # Efficient Frontier
    efficientList = []
    assets_weights_with_target = []
    targetReturns = np.linspace(minVol_returns, maxSR_returns, 50)
    for target in targetReturns:
        efficientList.append(efficientOpt(meanReturns, 
                                          covMatrix,
                                          target,
                                          constraintSet=constraintSet)['fun'])
        assets_weights_with_target.append(
            [target, efficientOpt(meanReturns, 
                                  covMatrix,
                                  target, 
                                  constraintSet=constraintSet)['x']]
        )
    
    minVol_returns, minVol_std = round(minVol_returns*100,3), round(minVol_std*100,3)
    maxSR_returns, maxSR_std = round(maxSR_returns*100,3), round(maxSR_std*100,3)
    
    return maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList, targetReturns, assets_weights_with_target 
   
#GETTING MARKET DATA (FROM COINGECKO)

def get_market_data(gecko_id, start, end):
    """Returns historical data for a given asset, which includes mktcap, close price and volume"""

    url = f'https://api.coingecko.com/api/v3/coins/{gecko_id}/market_chart?vs_currency=usd&days=max&interval=daily'
    answer = re.get(url).json()
    sleep(.2)
    price_list = []
    index = []
    mktcap_list = []
    volume_list = []
    df = pd.DataFrame() 
    if answer == {'error': 'Could not find coin with the given id'}:
        print(protocol+' was not found!')
        return df
    for num, item in enumerate(answer['prices']):
        price_list.append(answer['prices'][num][1])
        mktcap_list.append(answer['market_caps'][num][1])
        volume_list.append(answer['total_volumes'][num][1])
        index.append(dt.datetime.fromtimestamp(int(answer['prices'][num][0])/1000).strftime("%Y-%m-%d"))
    df['Price'] = pd.Series(price_list)
    df['Marketcap'] = pd.Series(mktcap_list)
    df['Volume'] = pd.Series(volume_list)
    df.index = pd.to_datetime(index)
    return df[ start: end]

def get_prices(coin_list, start, end): #Returns desired historical prices for coins in coin_list
    prices_df = pd.DataFrame()
    for coin in coin_list:
        price = get_market_data(coin,  start=start, end=end)['Price']
        prices_df[f'{coin}_price'] = price
    return prices_df#.fillna(method='bfill')

#PLOT FUNCTIONS

def EF_graph(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    """Returns a graph ploting the min vol, max sr and efficient frontier"""
    maxSR_returns, maxSR_std, maxSR_allocation, \
    minVol_returns, minVol_std, minVol_allocation, \
    efficientList, targetReturns, assets_weights_with_target \
    = calculatedResults(meanReturns=meanReturns, 
                        covMatrix=covMatrix,
                        riskFreeRate=riskFreeRate, 
                        constraintSet=constraintSet)
    #Max SR
    MaxSharpeRatio = go.Scatter(
        name='Maximum Sharpe Ratio',
        mode='markers',
        x=[maxSR_std],
        y=[maxSR_returns],
        marker=dict(color='red',size=14,line=dict(width=3, color='black'))
    )
    #Min Vol
    MinVol = go.Scatter(
        name='Minimum Volatility',
        mode='markers',
        x=[minVol_std],
        y=[minVol_returns],
        marker=dict(color='green',size=14,line=dict(width=3, color='black'))
    )
    #Efficient Frontier
    EF_curve = go.Scatter(
        name='Efficient Frontier',
        mode='lines',
        x=[round(ef_std*100, 2) for ef_std in efficientList],
        y=[round(target*100, 2) for target in targetReturns],
        line=dict(color='black', width=4, dash='dashdot')
    )
    data = [MaxSharpeRatio, MinVol, EF_curve]
    layout = go.Layout(
        title = 'Portfolio Optimisation with the Efficient Frontier',
        yaxis = dict(title='Annualised Return (%)'),
        xaxis = dict(title='Annualised Volatility (%)'),
        showlegend = True,
        legend = dict(
            x = 0.75, y = 0, traceorder='normal',
            bgcolor='#E2E2E2',
            bordercolor='black',
            borderwidth=2),
        width=800,
        height=600)
    fig = go.Figure(data=data, layout=layout)
    return fig.show()

def plot_performance(prices,
             startDate = '2022-01-27', 
             endDate = '2022-04-01',
             title = '',
             y_label = 'Asset Performance',
             save_fig = False,
             fig_name = 'something.png or jpeg'):
    '''
    Returns a figure that can be saved on the file directory
    '''
    index_monthly = pd.date_range(startDate, endDate, freq='1D')
    x= pd.to_datetime(prices[startDate:endDate].index)
    plt.figure(figsize=(16, 9), dpi=300)
    plt.title(label=title)
    np.random.seed(10)
    for col in prices.columns:
        plt.plot(x,prices[col][startDate:endDate]/prices[col][startDate], #Normalize to start at 1
                 color=(np.random.random(), 
                        np.random.random(),
                        np.random.random()),
                 label=f"{col}")
    plt.legend(loc='best', prop={'size': 15})
    plt.xticks(fontsize=14)
    plt.xlabel('Date', fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.ticklabel_format(style='plain', useOffset=False, axis='y')
    if save_fig:
        plt.savefig(fig_name)
    plt.show()