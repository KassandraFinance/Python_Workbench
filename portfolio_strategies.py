#Minimum Variance
import numpy as np
import pandas as pd

def get_weights_MV(df, risk_target = 1):
    '''
    Generates the weights for the Minimum Variance (MV) portfolio
    '''
    prices = df
    returns = prices.pct_change()
    # We calculate the covariance matrix
    covariances = returns.iloc[1:, :].cov()
#     assets = list(df.index.get_level_values(level='Security').unique())
    assets = prices.columns
    ones_vector = np.ones(len(assets)) # Denoted as 1
    # Gives weights as w = Cov**(-1) . 1 / (1' . Cov**(-1) . 1)
    w = np.linalg.inv(covariances) @ ones_vector / (ones_vector.T @  np.linalg.inv(covariances) @ ones_vector)
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

def get_weights_EW(df, risk_target = 1):
    '''
    Generates the weights for the Equal Weights (EW) portfolio
    '''
    
    prices = df
    returns = prices.pct_change()
    # We calculate the covariance matrix
    covariances = returns.iloc[1:, :].cov()
    w_unitary = [1/len(df.columns)]*len(df.columns)
    w_unitary = np.array(w_unitary)
    # We convert the weights to a matrix
    weights = np.matrix(w_unitary)
    # We calculate the risk of the weights distribution
    portfolio_risk = float(np.sqrt((weights * covariances.values * weights.T)))  
    if portfolio_risk > risk_target:
        weights = w_unitary * risk_target / portfolio_risk # Sum of weights less than 1
    else:
        weights = w_unitary
    return weights


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


def get_weights_IVol_power(df, power = 2, risk_target = 1):
    prices = df
    returns = prices.pct_change()
    covariances = returns.cov()
    assets_variances = np.array(returns.describe().loc['std'])**power # Denoted as Sigma
    assets = prices.columns
    ones_vector = np.ones(len(assets))
    #  used weights as w = 1/sigma_portfolio / sum(1/asset_i_volatility)
    w = (ones_vector / assets_variances) / (ones_vector / assets_variances).sum()  
    w_pd = pd.Series(w, index=assets, name='weights') 
    portfolio_risk = float(np.sqrt(w.T @ covariances @ w))  
    if portfolio_risk > risk_target:
        weights = w_pd * risk_target / portfolio_risk # Sum of weights less than 1
    else:
        weights = w_pd
    return weights

def get_weights_TG(df, risk_free = .0, risk_target = 1):
    prices = df
    returns = prices.pct_change()
    # We calculate the covariance matrix
    covariances = returns.iloc[1:, :].cov() 
    assets = prices.columns
    assets_expected_returns = np.array(returns.describe().loc['mean']) # Denoted as E
    ones_vector = np.ones(len(assets)) # Denoted as 1
    rf_vector = risk_free * ones_vector # Vector containing risk free reference
    #  used weights as w = Cov**(-1) . (E - rf.1) / (1' . Cov**(-1) . (E - rf.1)  )
    w = np.linalg.inv(covariances) @ (assets_expected_returns -  rf_vector) / (ones_vector.T @  np.linalg.inv(covariances) @ (assets_expected_returns -  rf_vector) ) 
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
