import pandas as pd
from pandas import DataFrame, Series, Timedelta
from pandas_ta.utils import v_datetime_ordered, v_series, v_offset
from typing import Optional, List

def anchored_vwap(df, start_date, end_date) -> float:
    """
    Calculate the Anchored VWAP from start_date to end_date in the given DataFrame.

    Args:
        df (DataFrame): DataFrame containing 'high', 'low', 'close', and 'volume'.
        start_date: Start date for the calculation.
        end_date: End date for the calculation.

    Returns:
        float: Anchored VWAP value.
    """
    # Ensure the DataFrame is sliced between the start and end dates
    anchored_data = df.loc[start_date:end_date]

    # Calculate Anchored VWAP
    vp_product = (anchored_data['volume'] * anchored_data['close']).cumsum()
    cumulative_volume = anchored_data['volume'].cumsum()
    avwap_value = vp_product / cumulative_volume

    return avwap_value.iloc[-1]  # Return the last calculated AVWAP value


def avwap(
    high: Series, low: Series, close: Series, volume: Series,
    left_strength: Optional[int] = 5, right_strength: Optional[int] = 5,
    bands: Optional[List] = None, offset: Optional[int] = 0,
    **kwargs
) -> DataFrame:
    """
    Anchored Volume Weighted Average Price (AVWAP)

    Description:
        Anchored VWAP is calculated from specific pivot points in the data, 
        providing insights into price action around significant market events.

    Args:
        high (Series): Series of 'high's.
        low (Series): Series of 'low's.
        close (Series): Series of 'close's.
        volume (Series): Series of 'volume's.
        left_strength (int): Number of bars back for pivot points. Default: 5.
        right_strength (int): Number of bars forward for pivot points. Default: 5.
        bands (List): List of standard deviations for Bollinger bands. Default: None.
        offset (int): How many periods to offset the result. Default: 0.

    Returns:
        DataFrame: A DataFrame with new features including segmented AVWAP values.
    """
    # Validate inputs
    high = v_series(high)
    low = v_series(low)
    close = v_series(close)
    volume = v_series(volume)

    if high is None or low is None or close is None or volume is None:
        return

    # Create DataFrame
    df = DataFrame({'high': high, 'low': low, 'close': close, 'volume': volume})
    
    # Calculate pivot points
    pivot_highs = pivot(high, left_strength, right_strength, pivot_type='high')
    pivot_lows = pivot(low, left_strength, right_strength, pivot_type='low')

     # Initialize columns for segmented AVWAP and pivot points
    df['AVWAP_HIGH'] = None
    df['AVWAP_LOW'] = None
    pivot_high_col = f"PIVOT_HIGH_{left_strength}_{right_strength}"
    pivot_low_col = f"PIVOT_LOW_{left_strength}_{right_strength}"
    df[pivot_high_col] = False
    df[pivot_low_col] = False

    # Calculate segmented AVWAP and set pivot points
    last_pivot_high = last_pivot_low = df.index[0]
    for index in df.index:
        if index in pivot_highs:
            last_pivot_high = index
            df.at[index, pivot_high_col] = True
        if index in pivot_lows:
            last_pivot_low = index
            df.at[index, pivot_low_col] = True

        df.at[index, 'AVWAP_HIGH'] = anchored_vwap(df, last_pivot_high, index)
        df.at[index, 'AVWAP_LOW'] = anchored_vwap(df, last_pivot_low, index)

    # Apply offset if needed and not None
    if offset is not None and offset != 0:
        df = df.shift(offset)

    # Handle missing values based on kwargs
    if "fillna" in kwargs:
        df.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        df.fillna(method=kwargs["fill_method"], inplace=True)

    return df


def pivot(data: Series, left_strength: int, right_strength: int, pivot_type: str) -> list:
    """
    Identify pivot points within a given data series, return their idx values.

    Pivot points are significant levels in the price of an asset where there is 
    a potential for a directional movement change. This function scans the data 
    series to find these pivot points, typically used in technical analysis to 
    mark areas of support (low pivots) and resistance (high pivots).

    Args:
        data (Series): A Pandas Series object containing the data in which to 
                       find pivot points, such as a series of stock prices.
        left_strength (int): The number of bars to the left of a potential 
                             pivot point to consider. A pivot point is valid if 
                             it represents an extremum within this range.
        right_strength (int): The number of bars to the right of a potential 
                              pivot point to consider. A pivot point is only 
                              confirmed if it remains extremum within the range.
        pivot_type (str): Type of pivot to find. This can be 'high' for pivot 
                          highs (local max) or 'low' for pivot lows (local min).

    Returns:
        list: A list of index values (e.g., dates) within the data series where 
              pivot points occur.

    Example:
        # Assuming df is a DataFrame with a 'high' column containing stock high prices
        pivot_highs = pivot(df['high'], left_strength=5, 
                            right_strength=5, pivot_type='high')
    """
    pivots = []
    data_values = data.values
    for i in range(left_strength, len(data) - right_strength):
        window = data_values[i - left_strength:i + right_strength + 1]
        if pivot_type == "high" and data_values[i] == max(window):
            pivots.append(data.index[i])
        elif pivot_type == "low" and data_values[i] == min(window):
            pivots.append(data.index[i])
    return pivots

