# ~*~ coding: utf8 ~*~
"""Utilities for correlations."""
from __future__ import print_function, division

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acovf, acf


def count_pairs(col):
    """Count the number of pairs for each lag.

    Parameters
    ----------
    col: pd.Series

    Returns
    -------
    n_pairs: np.ndarray
    """
    have_data = ~col.isnull()
    n_data = len(col)
    embedded = np.zeros(2 * n_data - 1)
    embedded[:n_data] = have_data
    spectrum = np.fft.fft(embedded)
    pair_count = np.round(
        np.fft.ifft(spectrum.conj() * spectrum, n_data).real
    ).astype(int)
    return pair_count


def get_autocorrelation_stats(column):
    """Get ACF and pair counts for column.

    Parameters
    ----------
    column: pd.Series

    Returns
    -------
    pd.DataFrame
    """
    # There's probably a better way to drop leading and trailing
    # missing values.  I should look into that.
    column = column.dropna().resample(column.index.freq).mean()
    time_index = column.index
    n_lags = len(column.index)
    timedelta_index = pd.timedelta_range(
        start=0, freq=time_index.freq, periods=n_lags
    )
    result = pd.DataFrame(
        index=timedelta_index, columns=["acf", "acovf", "pair_counts"]
    )
    result.loc[:, "acovf"] = acovf(
        column, missing="conservative", adjusted=True, fft=True,
    ).astype(np.float32)
    result.loc[:, "acf"] = acf(
        column, missing="conservative", nlags=n_lags, adjusted=True, fft=True
    ).astype(np.float32)
    result.loc[:, "pair_counts"] = count_pairs(column)
    return result
