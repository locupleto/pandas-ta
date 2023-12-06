import pandas as pd
import numpy as np
import math
from pandas import Series
from pandas_ta.overlap import jma, ema, zlma
from scipy.stats import norm

def rwd(high: Series, low: Series, close: Series, 
        min_length=8, max_length=65, bars_per_year=252,
        smooth_type='ezls',
        smooth_length=3, 
        smooth_gain=6,
        probability_output=False) -> pd.DataFrame: 
    """
    Random Walk Deviation (RWD)

    Description:
    Random Walk Deviation (RWD) is a technical analysis indicator inspired 
    by Cynthia Kase's "The Best Momentum Indicators" (1997) and E. Michael 
    Poulos's Random Walk Index (RWI). RWI quantifies the extent to which 
    price ranges over a fixed period deviate from a random walk, where a 
    larger-than-expected range indicates a trend. RWI employs Average True 
    Range (ATR) for volatility measurement. RWD, in contrast, aims to 
    closely replicate the Cynthia Kase's proprietary Kase Peak Oscillator (KPO) 
    and similarily to KPO, the RWD aims to be adaptive to market changes as 
    well as more statistically correct. The main differences between RWI and 
    RWD are:

    - RWD is adaptive. It finds the largest upward and downward deviation 
        within the specified range for each bar. (RWI uses one fixed period)  
    - RWD uses the standard deviation of logarithmic price changes, unlike 
        RWI which relies on simpler linear ATRs.
    - Four different smoothing functions can be configured

    RWD calculates by iterating over all lengths within a specified 
    interval, determining the maximum upward and downward deviation from 
    the expected random walk in log price returns. The RWD is expressed  
    on a standard deviations scale.

    Parameters:
    - open, high, low and close as pandas Series.
    - bars_per_year: Number of bars per year (default 252 for daily data).
    - min_length: Min length to calculate max deviation (default 8 days).
    - max_length: Max length for deviation calculation (default 65 days).

    Output Columns:
    - UP_LENGTH: Cycle length associated with maximum upward deviation.
    - DN_LENGTH: Cycle length associated with maximum downward deviation.
    - RWDh: Max upward random walk deviation
    - RWDl: Max downward random walk deviation
    - RWD: Difference between maximum directional deviations.
    - optional smoothed RWD

    Returns:
    A DataFrame containing the maximum upward and downward deviations, 
    expressed in standard deviations, along with other metrics. It is also
    possible to get the output expressed in probabilities (range -1.0 to +1.0)
    """

    params_suffix = f"{min_length}-{max_length}_{bars_per_year}"
    smooth_suffix = f"{smooth_type}_{smooth_length}_{smooth_gain}"

    length_up_col = f"UP_LENGTH_{params_suffix}"
    max_rw_deviation_up_col = f"RWDh_{params_suffix}"
    length_dn_col = f"DN_LENGTH_{params_suffix}"
    max_rw_deviation_dn_col = f"RWDl_{params_suffix}"
    rw_deviation_peak_col = f"RWD_PEAK_{params_suffix}"
    rw_deviation_peak_smooth_col = f"{rw_deviation_peak_col}_{smooth_suffix}"
    log_return_stddev_col = f"__LOG_RETURN_STDDEV"
 
    column_names = [length_up_col, max_rw_deviation_up_col, 
                    length_dn_col, max_rw_deviation_dn_col, 
                    rw_deviation_peak_col]
    
    # Creating a DataFrame from the Series
    df = pd.DataFrame({'high': high, 'low': low, 'close': close})

    # Create a fresh price DataFrame
    for col in column_names:
        df[col] = np.nan  # Initialize new columns with NaN 
    
    if 'close' in df and 'high' in df and 'low' in df:
        # create a numpy matrix with the necessary price time series lines
        p = df[['high', 'low', 'close']].values
        HIGH = 0
        LOW = 1
        
        # calculate the historical volatility for each length in the interval
        stddev = [None] * (max_length + 1) 
        for n in range(min_length, max_length + 1):
            col_iter_name = f"{log_return_stddev_col}_{n}"
            log_return_volatility(df, n)
            stddev[n] = np.array(df[col_iter_name].to_numpy())

        # drop stddev temp cols from the df
        columns_to_drop = [col for col in df.columns 
                           if col.startswith(log_return_stddev_col)]
        columns_to_drop.append('LOG_RETURNS')
        df.drop(columns=columns_to_drop, inplace=True)


        # four new timeseries lines for the upward deviation calculations
        accord_length_up_value = np.empty(len(df.index))
        accord_length_up_value[:] = 0
        accord_ratio_up_value = np.empty(len(df.index))
        accord_ratio_up_value[:] = 0
        max_rwd_up_value = np.empty(len(df.index))
        max_rwd_up_value[:] = -9999
        max_rwd_up_value[:max_length] = 0
        
        # four new timeseries lines for the downward deviation calculations
        accord_length_dn_value = np.empty(len(df.index))
        accord_length_dn_value[:] = 0
        accord_ratio_dn_value = np.empty(len(df.index))
        accord_ratio_dn_value[:] = 0
        max_rwd_dn_value = np.empty(len(df.index))
        max_rwd_dn_value[:] = 9999
        max_rwd_dn_value[:max_length] = 0  
        
        # peak value times series line (similar to KasePeakOscillator)
        random_walk_peak_value = np.empty(len(df.index))
        random_walk_peak_value[:] = 0
        
        # loop over the entire df starting from the first calculable data...
        for i in range(max_length, len(df.index)): 
                      
            # find the extreme values and accords within a rolling range window
            for n in range(min_length, max_length + 1):
                expected_yearly_volatility = (stddev[n][i] * math.sqrt(n) 
                                              * (bars_per_year / 252))
                if expected_yearly_volatility > 0:
                    
                    # Upward logarithmic random walk deviation calculations
                    if p[i-n][LOW] != 0: 
                        # log(High/Low[n])/(LogPriceReturnStdDev * sqrt(n))          
                        lrwd_up = (math.log(p[i][HIGH] / p[i-n][LOW]) 
                                   / expected_yearly_volatility)       
                    else:
                        lrwd_up = 0 
                    if lrwd_up > 0:
                        accord_ratio_up_value[i] = \
                            accord_ratio_up_value[i] + 1 
                    
                    if lrwd_up > max_rwd_up_value[i]:
                        max_rwd_up_value[i] = lrwd_up
                        accord_length_up_value[i] = n                      
                    
                    # downward logarithmic random walk deviation calculations
                    if p[i][LOW] != 0:
                        # -log(High[n]/low)/(log_price_return_stddev * sqrt(n))     
                        lrwd_dn = (-math.log(p[i-n][HIGH] / p[i][LOW])
                                    / expected_yearly_volatility)
                    else:
                        lrwd_dn = 0; 
                    if lrwd_dn < 0:
                        accord_ratio_dn_value[i] = \
                            accord_ratio_dn_value[i] + 1
      
                    if lrwd_dn < max_rwd_dn_value[i]:
                        max_rwd_dn_value[i] = lrwd_dn
                        accord_length_dn_value[i] = n  
                                   
            accord_ratio_up_value[i] = (accord_ratio_up_value[i] 
                                        / (max_length - min_length + 1))
            accord_ratio_dn_value[i] = (accord_ratio_dn_value[i] 
                                        / (max_length - min_length + 1))
            
            # rw_deviation_peak: max_rwd_up_value[i] + max_rwd_dn_value[i]
            # unless both values in the same direction, then skip the smallest
            if max_rwd_dn_value[i] > 0:
                random_walk_peak_value[i] = max_rwd_up_value[i]
            elif max_rwd_up_value[i] < 0:
                random_walk_peak_value[i] = max_rwd_dn_value[i]
            else:
                random_walk_peak_value[i] = (max_rwd_up_value[i] 
                                         + max_rwd_dn_value[i]) 
    
        # if specified to do so, convert the results from Z to probabilities
        if probability_output:
            max_rwd_up_value = norm.cdf(np.nan_to_num(max_rwd_up_value)) * 2 - 1
            max_rwd_dn_value = norm.cdf(np.nan_to_num(max_rwd_dn_value)) * 2 - 1
            random_walk_peak_value = norm.cdf(np.nan_to_num(random_walk_peak_value)) * 2 - 1

        # populate df with the nine calculated RWI-related time series cols
        df[length_up_col] = \
            pd.Series(accord_length_up_value, index=df.index) 
        df[max_rw_deviation_up_col] = \
            pd.Series(max_rwd_up_value, index=df.index)        
        df[length_dn_col] = \
            pd.Series(accord_length_dn_value, index=df.index) 
        df[max_rw_deviation_dn_col] = \
            pd.Series(max_rwd_dn_value, index=df.index)
        df[rw_deviation_peak_col] = \
            pd.Series(random_walk_peak_value, index=df.index) 
        
    # optional smoothing
    if smooth_type == 'zlema':
        df[rw_deviation_peak_smooth_col]= \
                zlma(df[rw_deviation_peak_col], 
                        length=smooth_length) 
    elif smooth_type == 'jma':
        df[rw_deviation_peak_smooth_col]= \
                jma(df[rw_deviation_peak_col], 
                        length=smooth_length,
                        phase=smooth_gain) 
    elif smooth_type == 'ezls':
        df[rw_deviation_peak_smooth_col]= \
                ezls(price=df[rw_deviation_peak_col], 
                    length=smooth_length, 
                    gain_limit=smooth_gain)
    elif smooth_type == 'ema':
        s = ema(df[rw_deviation_peak_col], 
                        length=round(smooth_length / 2))
        df[rw_deviation_peak_smooth_col]= \
                ema(s, length=round(smooth_length / 2)) 
    return df 
   
def log_returns(df):
    '''
    Market price returns are assumed to be log normal. Thus, the volatility of  
    a security can be defined as the standard deviation of the logarithmic rate 
    of change (in price). 
    
    Input:
        df: the referred data frame
        price: df[price] should contain the price input
    Output:       
        df[log_returns] will contain math.log(p[i]/p[i-1]) for price
    '''

    log_returns_col = f"LOG_RETURNS"

    p = np.array(df['close'])
    returns = np.empty(len(df.index))
    returns[:] = np.NaN
    for i in range(1, len(df.index)):
        if not math.isnan(p[i-1]) and p[i-1] != 0:
            returns[i] = math.log(p[i]/p[i-1])
    df[log_returns_col] = pd.Series(returns, index=df.index)
    if len(returns) != len(p):
        raise AssertionError
      
def log_return_volatility(df, num_obs):
    '''
    log_return_volatility
        
    Description:
        Relying on the model that the price volatility is log-normally 
        distributed (as Black-Scholes formula suggests), this function 
        calculates what one standard deviation of ln(price volatility) is.
        
        Two parameters will be populated upon return of this function:
        - log_return_stddev will contain the value of stddev(ln(price-deltas)) 
        - log_return_mean will contain the mean of the log(price-delta) series.
        
        Loosely based on the article "The Best momentum Indicators" written by
        Cynthia Kase. See section Historical Volatility in this article.
        
    Input Parameters:
        df: referred data frame
        price: df[price] should contain the price series input (Close)
        num_obs: no of bars back to base the volatility calculation on

    Output:       
        df[log_returns] will contain math.log(p[j]/p[j-1]) for Price
        df[log_return_mean]  will contain the mean of df[oLogReturns]
        df[log_return_variance] will contain the variance of df[oLogReturns]
        df[log_return_stddev] the value of one stddev of df[log_returns]      
    '''       
    log_returns_col = f"LOG_RETURNS"
    log_return_stddev_col = f"__LOG_RETURN_STDDEV_{num_obs}"

    # if not done already, calculate math.log(p[j]/p[j-1]) for all df[Price]
    if log_returns_col not in df:
        log_returns(df)
        
    # convert the log returns to an np array for the calculations
    np_logreturns = np.array(df[log_returns_col])   
    
    # if not done already, calculate rolling window stddev of these np_logreturns
    if log_return_stddev_col not in df:
        np_rolling_stddev = \
            np.std(rolling_window(np_logreturns, num_obs), 
                   axis=1, ddof=1)   
        np_rolling_stddev = prependIncalcuables(df, np_rolling_stddev)       
        df[log_return_stddev_col] = pd.Series(np_rolling_stddev, index=df.index) 
        if df[log_return_stddev_col].size != df[log_returns_col].size:
            raise AssertionError

def prependIncalcuables(df, npArray):
        # prepend a NaN for the first incalculable values and return the result
        missing = len(df.index) - len(npArray)
        if missing > 0:
            incalculables = np.empty(missing)
            incalculables[:] = np.NaN
            npArray = np.concatenate((incalculables, npArray), axis=0)
        if len(df.index) != len(npArray):
            raise AssertionError
        return npArray

def rolling_window(array, window):
    '''
    Helper function to speed up applying a function iteratively over a rolling
    window.
        
     Input:
        array - the array of data to roll the window along
        window - the length of the window
    Output: 
        an array of all "windowed" subarrays to apply a function to
 
     Usage:
        vols = numpy.std(rolling_window(price_array, window_len),axis=1)   
    '''
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

def ezls(price: Series, length=3, gain_limit=6):
    '''
    Ehlers Zero Lag Smoother (EZLS)

    This function implements John Ehlers' Zero Lag Smoother, adapted for use 
    with pandas. It aims to reduce the lag in moving averages by dynamically 
    adjusting the smoothing process through an error-correcting feedback loop.

    Parameters:
    - price (pd.Series): The price series to apply the smoother on
    - length (int): The equivalent length for the smoothing calculation. 
      Represents the period of the Exponential Moving Average (EMA). 
    - gain_limit (int): Sets the range for adjusting the gain in the error 
      correction loop. The actual gain is adjusted in steps of 0.1 within this 
      limit. 

    The algorithm:
    - Calculates an initial EMA of the price series.
    - Iteratively adjusts the gain within the specified limit to find the 
      setting that minimizes the error between the actual price and the error-
      corrected (EC) price.
    - The gain resulting in the least error is then used to compute the final 
      EC value.

    Returns:
    - pd.Series: A pandas Series containing the Zero Lag Smoother values, with 
      reduced lag in comparison to a standard EMA.

    Note:
    Original source: https://www.mesasoftware.com/papers/ZeroLag.pdf 
    '''
    p = np.array(price)
    ema = np.empty(len(price.index))
    ec = np.empty(len(price.index)) 
    ema[:] = 0
    ec[:] = 0
    alpha = 2 / (length + 1)
    least_error = 1000000
    best_gain = 0
    
    for i in range(1, len(price.index)):        
        ema[i] = alpha * p[i] + (1 - alpha) * ema[i-1];
        for g in range(-gain_limit, gain_limit):
            gain = g / 10 
            ec[i] = (alpha * (ema[i] + gain * (p[i] - ec[i-1]))
                      + (1 - alpha) * ec[i-1])
            error = p[i] - ec[i]
            if abs(error) < least_error:
                least_error = abs(error)
                best_gain = gain
        ec[i] = (alpha * (ema[i] + best_gain * (p[i] - ec[i-1])) 
                 + (1 - alpha) * ec[i-1])
        if abs(ec[i]) < 0.001:
            ec[i] = 0.001
    return pd.Series(ec, index=price.index) 
    