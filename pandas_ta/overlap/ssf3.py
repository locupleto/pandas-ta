# -*- coding: utf-8 -*-
from numpy import copy, cos, exp, ndarray
from pandas import Series
from pandas_ta.utils import get_offset, verify_series

try:
    from numba import njit
except ImportError:
    njit = lambda _: _


@njit
def np_ssf3(x: ndarray, n: int, pi: float, sqrt3: float):
    """John F. Ehler's Super Smoother Filter by Everget (3 poles), Tradingview
    https://www.tradingview.com/script/VdJy0yBJ-Ehlers-Super-Smoother-Filter/"""
    m, result = x.size, copy(x)
    a = exp(-pi / n)
    b = 2 * a * cos(-pi * sqrt3 / n)
    c = a * a

    d4 = c * c
    d3 = -c * (1 + b)
    d2 = b + c
    d1 = 1 - d2 - d3 - d4

    for i in range(3, m):
        result[i] = d1 * x[i] + d2 * result[i - 1] \
            + d3 * result[i - 2] + d4 * result[i - 3]

    return result


def ssf3(
        close: Series, length: int = None,
        pi: float = None, sqrt3: float = None,
        offset=None, **kwargs
    ):
    """Ehler's 3 Pole Super Smoother Filter (SSF) © 2013

    John F. Ehlers's solution to reduce lag and remove aliasing noise with his
    research in aerospace analog filter design. This is implementation has three
    poles. Since SSF is a (Resursive) Digital Filter, the number of poles
    determine how many prior recursive SSF bars to include in the filter design.

    For Everget's calculation on TradingView, set arguments:
        pi = np.pi, sqrt3 = 1.738

    Sources:
        https://www.tradingview.com/script/VdJy0yBJ-Ehlers-Super-Smoother-Filter/
        https://www.mql5.com/en/code/589

    Args:
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 20
        pi (float): The value of PI to use. The default is Ehler's
            truncated value 3.14159. Adjust the value for more precision.
            Default: 3.14159
        sqrt3 (float): The value of sqrt(3) to use. The default is Ehler's
            truncated value 1.732. Adjust the value for more precision.
            Default: 1.732
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)
        fill_method (value, optional): Type of fill method

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    length = int(length) if isinstance(length, int) and length > 0 else 20
    pi = float(pi) if isinstance(pi, float) and pi > 0 else 3.14159
    sqrt3 = float(sqrt3) if isinstance(sqrt3, float) and sqrt3 > 0 else 1.732
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None: return

    # Calculate
    np_close = close.values
    ssf = np_ssf3(np_close, length, pi, sqrt3)
    ssf = Series(ssf, index=close.index)

    # Offset
    if offset != 0:
        ssf = ssf.shift(offset)

    # Fill
    if "fillna" in kwargs:
        ssf.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        ssf.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Category
    ssf.name = f"SSF3_{length}"
    ssf.category = "overlap"

    return ssf
