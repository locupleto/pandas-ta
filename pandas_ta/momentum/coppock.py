# -*- coding: utf-8 -*-
from pandas import Series
from pandas_ta.overlap import wma
from pandas_ta.utils import get_offset, verify_series
from .roc import roc


def coppock(
        close: Series, length: int = None, fast: int = None, slow: int = None,
        offset: int = None, **kwargs
    ) -> Series:
    """Coppock Curve (COPC)

    Coppock Curve (originally called the "Trendex Model") is a momentum indicator
    is designed for use on a monthly time scale.  Although designed for monthly
    use, a daily calculation over the same period can be made, converting the
    periods to 294-day and 231-day rate of changes, and a 210-day weighted
    moving average.

    Sources:
        https://en.wikipedia.org/wiki/Coppock_curve

    Args:
        close (pd.Series): Series of 'close's
        length (int): WMA period. Default: 10
        fast (int): Fast ROC period. Default: 11
        slow (int): Slow ROC period. Default: 14
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)
        fill_method (value, optional): Type of fill method

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    length = int(length) if length and length > 0 else 10
    fast = int(fast) if fast and fast > 0 else 11
    slow = int(slow) if slow and slow > 0 else 14
    close = verify_series(close, max(length, fast, slow))
    offset = get_offset(offset)

    if close is None: return

    # Calculate
    total_roc = roc(close, fast) + roc(close, slow)
    coppock = wma(total_roc, length)

    # Offset
    if offset != 0:
        coppock = coppock.shift(offset)

    # Fill
    if "fillna" in kwargs:
        coppock.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        coppock.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Category
    coppock.name = f"COPC_{fast}_{slow}_{length}"
    coppock.category = "momentum"

    return coppock
