#!/usr/bin/env python
# ~*~ coding: utf8 ~*~
"""Analyze the fit of the various functions to the correlations.

One tower at a time, still.
"""
import inspect

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize

import flux_correlation_functions
import flux_correlation_functions_py

print("Reading correlation data", flush=True)
corr_data1 = pd.read_csv(
    "ameriflux-minus-casa-half-hour-towers-autocorrelation-functions.csv",
    index_col=0
)
corr_data2 = pd.read_csv(
    "ameriflux-minus-casa-hour-towers-autocorrelation-functions.csv",
    index_col=0
)
corr_data = pd.concat([corr_data1, corr_data2], axis=1)
corr_data.index = pd.TimedeltaIndex(corr_data.index)
corr_data.index.name = "Time separation"
corr_data = corr_data.astype(np.float32)
print("Have correlation data", flush=True)

pair_counts1 = pd.read_csv(
    "ameriflux-minus-casa-half-hour-towers-pair-counts.csv",
    index_col=0
)
pair_counts2 = pd.read_csv(
    "ameriflux-minus-casa-hour-towers-pair-counts.csv",
    index_col=0
)
pair_counts = pd.concat([pair_counts1, pair_counts2], axis=1)
pair_counts.index = pd.TimedeltaIndex(pair_counts.index)
print("Have pair counts", flush=True)

HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
DAYS_PER_YEAR = 365.2425
HOURS_PER_YEAR = HOURS_PER_DAY * DAYS_PER_YEAR
N_YEARS_DATA = 3

CORRELATION_FUNCTION_NAMES = [
    corr_name
    for corr_name in dir(flux_correlation_functions_py)
    if corr_name.endswith("numexpr_fn")
]
CORRELATION_FUNCTIONS = [
    getattr(flux_correlation_functions_py, corr_name)
    for corr_name in CORRELATION_FUNCTION_NAMES
]

STARTING_PARAMS = dict(
    daily_coef = 0.2,
    daily_coef1 = .7,
    daily_coef2 = .3,
    daily_width = .5,
    Td = 60,  # weeks
    dm_width = .8,
    dm_coef1 = .3,
    dm_coef2 = +.1,
    ann_coef1 = +.4,
    ann_coef2 = +.3,
    ann_coef = 0.1,
    ann_width = .3,
    Ta = 3.,  # years
    resid_coef = 0.05,
    To = 2.,  # weeks
    ec_coef = 0.7,
    Tec = 2.,  # hours
)
PARAM_LOWER_BOUNDS = dict(
    daily_coef = -10,
    daily_coef1 = -10,
    daily_coef2 = -10,
    daily_width = 0,
    Td = 0,  # weeks
    dm_width = 0,
    dm_coef1 = -10,
    dm_coef2 = -10,
    ann_coef1 = -10,
    ann_coef2 = -10,
    ann_coef = -10,
    ann_width = 0,
    Ta = 0,  # years
    resid_coef = -10,
    To = 0.,  # weeks
    ec_coef = -10,
    Tec = 0.,  # hours
)
PARAM_UPPER_BOUNDS = dict(
    daily_coef = 10,
    daily_coef1 = 10,
    daily_coef2 = 10,
    daily_width = 10,
    Td = 500,  # fortnights
    dm_width = 10,
    dm_coef1 = 10,
    dm_coef2 = 10,
    ann_coef1 = 10,
    ann_coef2 = 10,
    ann_coef = 10,
    ann_width = 10,
    Ta = 4,  # decades
    resid_coef = 10,
    To = 500.,  # fortnights
    ec_coef = 10,
    Tec = 1000.,  # hours
)

for coef, val in STARTING_PARAMS.items():
    STARTING_PARAMS[coef] = np.float32(val)

COEFFICIENT_DATA = pd.DataFrame(
    columns=STARTING_PARAMS.keys(),
    index=pd.MultiIndex.from_product(
        [corr_data, CORRELATION_FUNCTION_NAMES],
        names=["Site", "Correlation Function"],
    )
)
COEFFICIENT_VAR_DATA = pd.DataFrame(
    columns=STARTING_PARAMS.keys(),
    index=pd.MultiIndex.from_product(
        [corr_data, CORRELATION_FUNCTION_NAMES],
        names=["Site", "Correlation Function"],
    )
)

for column in corr_data.iloc[:, :]:
    print(column, flush=True)
    tower_counts = pair_counts.loc[:, column].dropna()
    tower_correlations = corr_data.loc[tower_counts.index, column].loc[tower_counts != 0]
    tower_counts = tower_counts.loc[tower_correlations.index]
    tower_lags = tower_correlations.index.values.astype("m8[h]").astype("u8")
    tower_lags -= tower_lags[0]
    tower_lags = tower_lags.astype(np.float32) / HOURS_PER_DAY
    if (
            len(tower_lags) < 0.75 * N_YEARS_DATA or
            tower_lags[-1] / DAYS_PER_YEAR < N_YEARS_DATA
    ):
        print(
            "Skipping tower.",
            len(tower_lags) / HOURS_PER_YEAR,
            "years worth of data points over",
            tower_lags[-1] /DAYS_PER_YEAR,
            "years.",
        )
        continue
    tower_lag_weights = np.sqrt(1. / tower_counts).astype(np.float32)
    fig, axes = plt.subplots(len(CORRELATION_FUNCTIONS) + 1, 1,
                             figsize=(6.5, 9), sharex=True, sharey=True)
    emp_line, = axes[0].plot(tower_lags, tower_correlations)
    emp_title = axes[0].set_title("Empirical Correlogram")
    fig_title = fig.suptitle(column)
    for corr_name, corr_fun, ax in zip(
            CORRELATION_FUNCTION_NAMES,
            CORRELATION_FUNCTIONS,
            axes[1:]
    ):
        print(corr_name, flush=True)
        argspec = inspect.getfullargspec(corr_fun)
        param_names = argspec.args[1:]
        try:
            param_vals, param_cov = scipy.optimize.curve_fit(
                corr_fun, tower_lags, tower_correlations,
                [STARTING_PARAMS[param] for param in param_names],
                sigma=tower_lag_weights,
                bounds=(
                    np.array(
                        [PARAM_LOWER_BOUNDS[param] for param in param_names],
                        dtype=np.float32
                    ),
                    np.array(
                        [PARAM_UPPER_BOUNDS[param] for param in param_names],
                        dtype=np.float32
                    ),
                )
            )
        except RuntimeError:
            continue
        COEFFICIENT_DATA.loc[(column, corr_name), param_names] = param_vals
        COEFFICIENT_VAR_DATA.loc[(column, corr_name), param_names] = np.diag(
            param_cov
        )
        fit_line, = ax.plot(
            tower_lags, corr_fun(tower_lags, *param_vals), label="ACF fit"
        )
        fit_name = ax.set_title("Fit of {corr_name:s}".format(corr_name=corr_name))
    fig.tight_layout()
    fig.subplots_adjust(top=.9, hspace=1.1)
    old_xlim = ax.set_xlim(0, DAYS_PER_YEAR * N_YEARS_DATA)
    old_ylim = ax.set_ylim(-1, 1)
    xtick_locations = pd.timedelta_range(
        start=0, freq="365D", periods=N_YEARS_DATA + 1
    ).to_numpy().astype("m8[D]").astype("u8").astype(np.float32)
    for ax in axes:
        xticks = ax.set_xticks(xtick_locations)
    fig.savefig("{tower:s}-ameriflux-minus-casa-corr-fits-long.pdf".format(
        tower=column
    ))
    old_xlim = ax.set_xlim(0, DAYS_PER_WEEK * N_YEARS_DATA)
    xtick_locations = pd.timedelta_range(
        start=0, freq="7D", periods=N_YEARS_DATA + 1
    ).to_numpy().astype("m8[D]").astype("u8").astype(np.float32)
    xtick_locations_minor = pd.timedelta_range(
        start=0, freq="1D", periods=N_YEARS_DATA + 1
    ).to_numpy().astype("m8[D]").astype("u8").astype(np.float32)
    for ax in axes:
        xticks = ax.set_xticks(xtick_locations)
        xticks_minor = ax.set_xticks(xtick_locations_minor, minor=True)
    fig.savefig("{tower:s}-ameriflux-minus-casa-corr-fits-short.pdf".format(
        tower=column
    ))
    plt.close(fig)

COEFFICIENT_DATA.to_csv("ameriflux-minus-casa-all-towers-parameters.csv")
COEFFICIENT_VAR_DATA.to_csv("ameriflux-minus-casa-all-towers-parameter-variances.csv")
