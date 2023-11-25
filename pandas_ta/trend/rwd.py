import pandas as pd
import numpy as np
import math
from pandas import Series
from pandas_ta.overlap import jma, ema, zlma

def rwd(high: Series, low: Series, close: Series, 
        min_length=8, max_length=65, bars_per_year=252,
        smooth_type='ezls',
        smooth_length=3, 
        smooth_gain=6) -> pd.DataFrame: 
    """
        Random Walk Deviation (RWD)

        Description:
        Random Walk Deviation (RWD) is a technical analysis indicator inspired 
        by Cynthia Kase's"The Best Momentum Indicators" (1997) and E. Michael 
        Poulos's Random Walk Index (RWI). RWI quantifies the extent to which 
        price ranges over a set period deviate from a random walk, where a 
        larger-than-expected range indicates a trend. RWI employs Average True 
        Range (ATR) for volatility measurement. RWD, in contrast, aims to 
        closely replicate the proprietary Kase Peak Oscillator (KPO) but with 
        significant differences:
        
        - RWD uses the standard deviation of logarithmic price changes, unlike 
          RWI which relies on linear ATRs.
        - While RWI plots two lines (Max(RWIUp) and Min(RWIDn)), RWD calculates 
        the difference (Max(RWIUp) - Min(RWIDn)) and typically presents it as 
        a histogram.
        - Four different smoothing functions can be configured

        RWD calculates by iterating over all lengths within a specified 
        interval, determining the maximum upward and downward deviation from 
        the expected random walk in log price returns, and expressing these 
        deviations in standard deviations.

        Parameters:
        - open, high, low and close as pandas Series.
        - bars_per_year: Number of bars per year (default 252 for daily data).
        - min_length: Min length to calculate max deviation (default 8 days).
        - max_length: Max length for deviation calculation (default 65 days).

        Output Columns:
        - log_returns: Logarithmic returns calculated as math.log(p[i]/p[i-1]).
        - length_up: Cycle length associated with maximum upward deviation.
        - length_dn: Cycle length associated with maximum downward deviation.
        - accord_ratio_up_value: Ratio of lengths approximating the maximum 
          upward deviation.
        - accord_ratio_dn_value: Ratio of lengths approximating the maximum 
          downward deviation.
        - max_rw_deviation_up: Max random walk deviation in the upward 
          direction.
        - max_rw_deviation_dn: Max random walk deviation in the downward 
          direction.
        - rw_deviation_peak: Difference between maximum upward and downward 
          deviations.

        Returns:
        A DataFrame containing the maximum upward deviation, along with 
        additional metrics, all expressed in standard deviations. These values 
        indicate deviations from expected random log price returns.
        """
    
    log_return_stddev_col = f"LOG_RETURN_STDDEV"
    params_suffix = f"{min_length}-{max_length}_{bars_per_year}"
    smooth_suffix = f"{smooth_type}_{smooth_length}_{smooth_gain}"

    length_up_col=  \
        f"UP_LENGTH_{params_suffix}"
    accord_ratio_up_col = \
        f"ACCORD_RATIO_UP_{params_suffix}"
    max_rw_deviation_up_col = \
        f"MAX_RANDOM_WALK_DEVIATION_UP_{params_suffix}"
    length_dn_col = \
        f"DN_LENGTH_{params_suffix}"
    accord_ratio_dn_col = \
        f"ACCORD_RATIO_DN_{params_suffix}"
    max_rw_deviation_dn_col = \
        f"DN_LENGTH__{params_suffix}"
    rw_deviation_peak_col = \
        f"RANDOM_WALK_DEVIATION_PEAK_{params_suffix}"
    rw_deviation_peak_smooth_col = \
        f"{rw_deviation_peak_col}_{smooth_suffix}"
    
    # allocate a new empty df with enough space to avoid performance issues
    prefixes = ['LOG_RETURN_MEAN', 
                'LOG_RETURN_VARIANCE', 
                'LOG_RETURN_STDDEV']
    suffix_range = range(min_length, max_length)
 

    column_names = [f"{prefix}_{suffix}"
                for prefix in prefixes for suffix in suffix_range]
    column_names.extend([length_up_col, accord_ratio_up_col, 
                         max_rw_deviation_up_col, length_dn_col, 
                         accord_ratio_dn_col, max_rw_deviation_dn_col,
                         rw_deviation_peak_col, rw_deviation_peak_smooth_col])

    # Create an empty DataFrame with the new column names
    df = pd.DataFrame(columns=column_names)

    # create a numpy matrix with the necessary price time series lines
    p = df[high, low, close].values
    HIGH = 0
    LOW = 1
    
    # calculate the historical volatility for each length in the interval
    stddev = [None] * (max_length + 1) 
    for n in range(min_length, max_length + 1):
        col_iter_name = f"{log_return_stddev_col}_{n}"
        log_return_volatility(df, 'close', n)
        stddev[n] = np.array(df[col_iter_name].to_numpy())
    
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

    # populate df with the nine calculated RWI-related time series cols
    df[length_up_col] = \
        pd.Series(accord_length_up_value, index=df.index) 
    df[accord_ratio_up_col] = \
        pd.Series(accord_ratio_up_value, index=df.index)
    df[max_rw_deviation_up_col] = \
        pd.Series(max_rwd_up_value, index=df.index)        
    df[length_dn_col] = \
        pd.Series(accord_length_dn_value, index=df.index) 
    df[accord_ratio_dn_col] = \
        pd.Series(accord_ratio_dn_value, index=df.index)
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
        df = ezls(df[rw_deviation_peak_col], 
                        rw_deviation_peak_col, 
                        smooth_length, 
                        gain_limit=smooth_gain, 
                        output_col=rw_deviation_peak_smooth_col)
    elif smooth_type == 'ema':
        s = ema(df[rw_deviation_peak_col], 
                        length=round(smooth_length / 2))
        df[rw_deviation_peak_smooth_col]= \
                ema(s, length=round(smooth_length / 2)) 

    return df 

def ezls(price: Series, length=3, gain_limit=6):
    """
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
    """
    # Convert the input Series to a numpy array for computation
    p = price.to_numpy()
    ema = np.zeros_like(p)
    ec = np.zeros_like(p)
    
    alpha = 2 / (length + 1)
    least_error = 1000000
    best_gain = 0
    
    for i in range(1, len(p)):
        ema[i] = alpha * p[i] + (1 - alpha) * ema[i-1]
        for g in range(-gain_limit, gain_limit):
            gain = g / 10 
            ec_temp = alpha * (ema[i] + gain * (p[i] - ec[i-1])) + (1 - alpha) * ec[i-1]
            error = p[i] - ec_temp
            if abs(error) < least_error:
                least_error = abs(error)
                best_gain = gain
        ec[i] = alpha * (ema[i] + best_gain * (p[i] - ec[i-1])) + (1 - alpha) * ec[i-1]
        ec[i] = max(ec[i], 0.001)  # Ensure ec[i] does not go below 0.001

    # Name and Category
    ezls.name = f"EZLS_{length}_{gain_limit}"
    ezls.category = "overlap"

    # Return the result as a new pandas Series
    return pd.Series(ec, index=price.index)

def log_return_volatility(price: Series, num_obs):
    '''
    log_return_volatility

    Description:
        Calculates the volatility of logarithmic returns of a price series.
        The volatility is modeled as log-normally distributed, as suggested 
        by the Black-Scholes formula.

    Parameters:
        - price (pd.Series): Series containing price data.
        - num_obs (int): Number of observations to base the volatility 
          calculation on.

    Returns:
        pd.DataFrame: DataFrame containing the log returns, mean, variance, and 
                      standard deviation of log returns.
    '''     
    log_returns_col = "LOG_RETURNS"
    log_return_mean_col = \
        f"LOG_RETURN_MEAN_{num_obs}"
    log_return_variance_col = \
        f"LOG_RETURN_VARIANCE_{num_obs}"
    log_return_stddev_col = \
        f"LOG_RETURN_STDDEV_{num_obs}"

    # Calculate log returns
    log_returns = np.log(price / price.shift(1))

    # Calculate rolling mean, variance, and standard deviation of log returns
    log_return_mean = log_returns.rolling(window=num_obs).mean()
    log_return_variance = log_returns.rolling(window=num_obs).var(ddof=1)
    log_return_stddev = log_returns.rolling(window=num_obs).std(ddof=1)

    # Compile results into a DataFrame
    result = pd.DataFrame({
        log_returns_col: log_returns,
        log_return_mean_col: log_return_mean,
        log_return_variance_col: log_return_variance,
        log_return_stddev_col: log_return_stddev
    }, index=price.index)

    return result
