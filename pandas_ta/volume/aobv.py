# -*- coding: utf-8 -*-
from pandas import DataFrame, Series
from pandas_ta.ma import ma
from pandas_ta.trend import long_run, short_run
from pandas_ta.utils import get_offset, verify_series
from .obv import obv


def aobv(
        close: Series, volume: Series, fast: int = None, slow: int = None,
        max_lookback: int = None, min_lookback: int = None, mamode: str = None,
        offset: int = None, **kwargs
    ) -> DataFrame:
    """Archer On Balance Volume (AOBV)

    Archer On Balance Volume (AOBV) developed by Kevin Johnson provides
    additional indicator analysis on OBV. It calculates moving averages, default
    'ema', of OBV as well as the moving average Long and Short Run Trends, see
    ```help(ta.long_run)```. Lastly, the indicator also calculates the rolling
    Maximum and Minimum OBV.

    Sources:
        https://www.tradingview.com/script/Co1ksara-Trade-Archer-On-balance-Volume-Moving-Averages-v1/

    Args:
        close (pd.Series): Series of 'close's
        volume (pd.Series): Series of 'volume's
        fast (int): The period of the fast moving average. Default: 4
        slow (int): The period of the slow moving average. Default: 12
        max_lookback (int): Maximum OBV bars back. Default: 2
        min_lookback (int): Minimum OBV bars back. Default: 2
        mamode (str): See ```help(ta.ma)```. Default: 'ema'
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        run_length (int): Trend length for OBV long and short runs. Default: 2
        fillna (value, optional): pd.DataFrame.fillna(value)
        fill_method (value, optional): Type of fill method

    Returns:
        pd.DataFrame: OBV_MIN, OBV_MAX, OBV_FMA, OBV_SMA, OBV_LR, OBV_SR columns.
    """
    # Validate
    fast = int(fast) if fast and fast > 0 else 4
    slow = int(slow) if slow and slow > 0 else 12
    max_lookback = int(max_lookback) if max_lookback and max_lookback > 0 else 2
    min_lookback = int(min_lookback) if min_lookback and min_lookback > 0 else 2
    if slow < fast:
        fast, slow = slow, fast
    mamode = mamode if isinstance(mamode, str) else "ema"
    _length = max(fast, slow, max_lookback, min_lookback)
    close = verify_series(close, _length)
    volume = verify_series(volume, _length)
    offset = get_offset(offset)
    if "length" in kwargs: kwargs.pop("length")
    run_length = kwargs.pop("run_length", 2)

    if close is None or volume is None: return

    # Calculate
    obv_ = obv(close=close, volume=volume, **kwargs)
    maf = ma(mamode, obv_, length=fast, **kwargs)
    mas = ma(mamode, obv_, length=slow, **kwargs)

    obv_long = long_run(maf, mas, length=run_length)
    obv_short = short_run(maf, mas, length=run_length)

    # Offset
    if offset != 0:
        obv_ = obv_.shift(offset)
        maf = maf.shift(offset)
        mas = mas.shift(offset)
        obv_long = obv_long.shift(offset)
        obv_short = obv_short.shift(offset)

    # Fill
    if "fillna" in kwargs:
        obv_.fillna(kwargs["fillna"], inplace=True)
        maf.fillna(kwargs["fillna"], inplace=True)
        mas.fillna(kwargs["fillna"], inplace=True)
        obv_long.fillna(kwargs["fillna"], inplace=True)
        obv_short.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        obv_.fillna(method=kwargs["fill_method"], inplace=True)
        maf.fillna(method=kwargs["fill_method"], inplace=True)
        mas.fillna(method=kwargs["fill_method"], inplace=True)
        obv_long.fillna(method=kwargs["fill_method"], inplace=True)
        obv_short.fillna(method=kwargs["fill_method"], inplace=True)

    _mode = mamode.lower()[0] if len(mamode) else ""
    data = {
        obv_.name: obv_,
        f"OBV_min_{min_lookback}": obv_.rolling(min_lookback).min(),
        f"OBV_max_{max_lookback}": obv_.rolling(max_lookback).max(),
        f"OBV{_mode}_{fast}": maf,
        f"OBV{_mode}_{slow}": mas,
        f"AOBV_LR_{run_length}": obv_long,
        f"AOBV_SR_{run_length}": obv_short,
    }
    aobvdf = DataFrame(data)

    # Name and Category
    aobvdf.name = f"AOBV{_mode}_{fast}_{slow}_{min_lookback}_{max_lookback}_{run_length}"
    aobvdf.category = "volume"

    return aobvdf
