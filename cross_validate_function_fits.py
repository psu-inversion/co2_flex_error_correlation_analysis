#!/usr/bin/env python
# ~*~ coding: utf8 ~*~
"""Find best correlation functions based on cross-validation on a single tower.

I will likely need to generalize this to include cross-validation
across towers at some point.  I may limit that to only a subset of the
correlation functions.
"""
from __future__ import print_function, division

import datetime
import itertools

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd
import xarray

import flux_correlation_function_fits
from correlation_utils import get_autocorrelation_stats

from correlation_function_fits import (
    GLOBAL_DICT, CorrelationPart, PartForm,
    is_valid_combination, get_full_expression,
    get_full_parameter_list,
    get_weighted_fit_expression,
)

# Time constants
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
DAYS_PER_FORTNIGHT = 14
DAYS_PER_YEAR = 365.2425
DAYS_PER_DECADE = 10 * DAYS_PER_YEAR
HOURS_PER_YEAR = HOURS_PER_DAY * DAYS_PER_YEAR

# Need to check this separately for each half of the data
N_YEARS_DATA = 2
REQUIRED_DATA_FRAC = 0.8


def has_enough_data(da):
    """Check whether there is enough data for a good analysis.

    Parameters
    ----------
    da: xarray.DataArray

    Returns
    -------
    bool
    """
    time_index_name = [
        name
        for name in da.dims
        if "time" in name.lower()
    ][0]
    time_index = da.indexes[time_index_name]
    if len(time_index) < 1:
        print("No data")
        return False
    if time_index[-1] - time_index[0] < datetime.timedelta(
            days=N_YEARS_DATA * DAYS_PER_YEAR
    ):
        print("< 2 years")
        return False
    if (
            da.count(time_index_name).values[()] <
            REQUIRED_DATA_FRAC * N_YEARS_DATA * HOURS_PER_YEAR
    ):
        print("Missing data")
        return False
    return True


def timedelta_index_to_floats(index):
    """Turn a TimedeltaIndex into an array of floats.

    Parameters
    ----------
    index: pd.TimedeltaIndex

    Returns
    -------
    np.ndarray
    """
    # Timedeltas can be negative, so the underlying type should be a
    # signed integer.
    lag_times = index.values.astype("m8[h]").astype("i8")
    lag_times -= lag_times[0]
    lag_times = lag_times.astype("f4") / 24
    return lag_times.astype(np.float32)


############################################################
# Set initial values and bounds for the parameters
STARTING_PARAMS = dict(
    daily_coef = 0.2,
    daily_coef1 = .7,
    daily_coef2 = .3,
    daily_width = .5,
    daily_timescale = 60,  # fortnights
    dm_width = .8,
    dm_coef1 = .3,
    dm_coef2 = +.1,
    ann_coef1 = +.4,
    ann_coef2 = +.3,
    ann_coef = 0.1,
    ann_width = .3,
    ann_timescale = 3.,  # decades
    resid_coef = 0.05,
    resid_timescale = 2.,  # fortnights
    ec_coef = 0.7,
    ec_timescale = 2.,  # hours
)
PARAM_LOWER_BOUNDS = dict(
    daily_coef = -10,
    daily_coef1 = -10,
    daily_coef2 = -10,
    daily_width = 0,
    daily_timescale = 0,  # fortnights
    dm_width = 0,
    dm_coef1 = -10,
    dm_coef2 = -10,
    ann_coef1 = -10,
    ann_coef2 = -10,
    ann_coef = -10,
    ann_width = 0,
    ann_timescale = 0,  # decades
    resid_coef = -10,
    resid_timescale = 0.,  # fortnights
    ec_coef = -10,
    ec_timescale = 0.,  # hours
)
PARAM_UPPER_BOUNDS = dict(
    daily_coef = 10,
    daily_coef1 = 10,
    daily_coef2 = 10,
    daily_width = 10,
    daily_timescale = 500,  # fortnights
    dm_width = 10,
    dm_coef1 = 10,
    dm_coef2 = 10,
    ann_coef1 = 10,
    ann_coef2 = 10,
    ann_coef = 10,
    ann_width = 10,
    ann_timescale = 4,  # decades
    resid_coef = 10,
    resid_timescale = 500.,  # fortnights
    ec_coef = 10,
    ec_timescale = 1000.,  # hours
)

# Convert initial values and bounds to float32
for coef, val in STARTING_PARAMS.items():
    STARTING_PARAMS[coef] = np.float32(val)

for coef, val in PARAM_LOWER_BOUNDS.items():
    PARAM_LOWER_BOUNDS[coef] = np.float32(val)

for coef, val in PARAM_UPPER_BOUNDS.items():
    PARAM_UPPER_BOUNDS[coef] = np.float32(val)

############################################################
# Read in data
AMERIFLUX_MINUS_CASA_DATA = xarray.open_dataset(
    "ameriflux-and-casa-matching-data-2.nc4"
)

############################################################
# Set up data frames for results
COEF_DATA = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [AMERIFLUX_MINUS_CASA_DATA.indexes["site"],
         [
             "_".join([
                 "{0:s}{1:s}".format(
                     part.get_short_name(),
                     form.get_short_name(),
                 )
                 for part, form in zip(CorrelationPart, forms)
             ])
             for forms in itertools.product(PartForm, PartForm, PartForm)
             if is_valid_combination(*forms)
         ]],
        names=["Site", "Correlation Function"],
    ),
    columns=[
        "daily_coef",
        "daily_timescale",
        "daily_coef1",
        "daily_coef2",
        "daily_width",
        "dm_coef1",
        "dm_coef2",
        "dm_width",
        "ann_coef",
        "ann_timescale",
        "ann_coef1",
        "ann_coef2",
        "ann_width",
        "resid_coef",
        "resid_timescale",
        "ec_coef",
        "ec_timescale",
    ],
    dtype=np.float32,
)
COEF_VAR_DATA = COEF_DATA.copy()
CORRELATION_FIT_ERROR = pd.DataFrame(
    index=COEF_DATA.index,
    columns=pd.MultiIndex.from_product(
        [
            ["function_optimized", "other_function"],
            ["weighted_error_in_sample", "weighted_error_out_of_sample"],
        ],
        names=["which_function", "which_data"],
    ),
    dtype=np.float32,
)
CORRELATION_FIT_ERROR.iloc[:, :] = np.inf

for site_name in AMERIFLUX_MINUS_CASA_DATA.indexes["site"]:
    print(site_name, flush=True)
    # Pull out non-missing site data, so I can get a decent idea of
    # what has enough data I can use.
    site_data = AMERIFLUX_MINUS_CASA_DATA[
        "flux_difference"
    ].sel(
        site=site_name
    ).dropna("time").sortby("time")

    # Split data in two, so we have separate training and validation
    # data sets.
    first_half  = site_data.isel(
        time=slice(None, len(site_data) // 2)
    )
    second_half = site_data.isel(
        time=slice(len(site_data) // 2, None)
    )
    if not (has_enough_data(first_half) and has_enough_data(second_half)):
        print("Not enough data.  Skipping:", site_name)
        continue
    
    # Resample to an hour, so acf/count_pairs can work (they assume
    # regularly spaced data, and I need the .freq attribute to make
    # that happen)
    first_half.resample(time="1H").first()
    second_half.resample(time="1H").first()

    for train_data, validation_data in itertools.permutations(
            [first_half, second_half]
    ):
        print("New train/val split")
        corr_data_train = get_autocorrelation_stats(
            train_data.to_dataframe()[
                "flux_difference"
            ].resample("1H").first()
        )
        corr_data_train = corr_data_train[
            corr_data_train["pair_counts"] > 0
        ]
        corr_data_validate = get_autocorrelation_stats(
            validation_data.to_dataframe()[
                "flux_difference"
            ].resample("1H").first()
        )
        corr_data_validate = corr_data_validate[
            corr_data_validate["pair_counts"] > 0
        ]
        acf_lags_train = timedelta_index_to_floats(corr_data_train.index)
        acf_lags_validate = timedelta_index_to_floats(corr_data_validate.index)
        acf_weights_train = 1. / np.sqrt(corr_data_train["pair_counts"])
        acf_weights_validate = 1. / np.sqrt(corr_data_validate["pair_counts"])

        for forms in itertools.product(PartForm, PartForm, PartForm):
            if not is_valid_combination(*forms):
                continue

            # Get function to optimize
            func_short_name = "_".join([
                 "{0:s}{1:s}".format(
                     part.get_short_name(),
                     form.get_short_name(),
                 )
                 for part, form in zip(CorrelationPart, forms)
             ])
            print(func_short_name, flush=True)
            name_to_optimize = "{fun_name}_fit_loop".format(fun_name=func_short_name)
            fun_to_optimize = getattr(
                flux_correlation_function_fits, name_to_optimize
            )
            fun_to_check = getattr(
                flux_correlation_function_fits,
                "{fun_name}_fit_ne".format(fun_name=func_short_name),
            )

            # Set up parameters
            parameter_list = get_full_parameter_list(*forms)
            starting_params = np.array(
                [STARTING_PARAMS[param] for param in parameter_list],
                dtype=np.float32,
            )
            lower_bounds = np.array(
                [PARAM_LOWER_BOUNDS[param] for param in parameter_list],
                dtype=np.float32,
            )
            upper_bounds = np.array(
                [PARAM_UPPER_BOUNDS[param] for param in parameter_list],
                dtype=np.float32,
            )

            # Try the optimization
            # Use curve_fit to fine-tune
            curve_and_deriv = getattr(
                flux_correlation_function_fits,
                "{fun_name:s}_curve_loop".format(fun_name=func_short_name),
            )
            def curve_deriv(tdata, *params):
                return curve_and_deriv(tdata, *params)[1]
            try:
                opt_params, param_cov = scipy.optimize.curve_fit(
                    getattr(
                        flux_correlation_function_fits,
                        "{fun_name:s}_curve_ne".format(fun_name=func_short_name),
                    ),
                    acf_lags_train.astype(np.float32),
                    corr_data_train["acf"].astype(np.float32).values,
                    starting_params.astype(np.float32),
                    acf_weights_train.astype(np.float32),
                    bounds=(
                        lower_bounds.astype(np.float32),
                        upper_bounds.astype(np.float32),
                    ),
                    jac=curve_deriv,
                )
            except (RuntimeError, ValueError) as err:
                print(err, "Curve fit failed, next function", sep="\n")
                continue
            opt_res = scipy.optimize.minimize(
                fun_to_optimize,
                opt_params,
                (
                    acf_lags_train,
                    corr_data_train["acf"].astype(np.float32).values,
                    corr_data_train["pair_counts"].astype(np.float32).values,
                ),
                jac="loop" in name_to_optimize,
                bounds=scipy.optimize.Bounds(lower_bounds, upper_bounds),
                method="L-BFGS-B",
                options={
                    "maxcor": 30,  # "iprint": 101
                },
            )
            # # Go to next function if it fails
            # if not opt_res.success:
            #     print(opt_res.message)
            #     print("No convergence, next function")
            #     continue
            print(opt_res.x)
            print(opt_params)

            # If this fit's cross-validation score is worse than the
            # currently-stored one, don't bother recording.
            if (
                    fun_to_optimize(
                        opt_res.x,
                        acf_lags_validate,
                        corr_data_validate["acf"].astype(np.float32).values,
                        corr_data_validate["pair_counts"].astype(np.float32).values,
                    )[0] >
                    CORRELATION_FIT_ERROR.loc[
                        (site_name, func_short_name),
                        ("function_optimized", "weighted_error_out_of_sample")
                    ]
            ):
                continue

            # Otherwise, save the results
            COEF_DATA.loc[(site_name, func_short_name), parameter_list] = (
                opt_res.x
            )
            COEF_VAR_DATA.loc[(site_name, func_short_name), parameter_list] = (
                np.diag(opt_res.hess_inv.todense())
            )
            CORRELATION_FIT_ERROR.loc[
                (site_name, func_short_name),
                ("function_optimized", "weighted_error_in_sample"),
            ] = opt_res.fun
            CORRELATION_FIT_ERROR.loc[
                (site_name, func_short_name),
                ("function_optimized", "weighted_error_out_of_sample"),
            ] = fun_to_optimize(
                opt_res.x,
                acf_lags_validate,
                corr_data_validate["acf"].astype(np.float32).values,
                corr_data_validate["pair_counts"].astype(np.float32).values,
            )[0]
            CORRELATION_FIT_ERROR.loc[
                (site_name, func_short_name),
                ("other_function", "weighted_error_in_sample"),
            ] = fun_to_check(
                opt_res.x,
                acf_lags_train,
                corr_data_train["acf"].astype(np.float32).values,
                corr_data_train["pair_counts"].astype(np.float32).values,
            )
            CORRELATION_FIT_ERROR.loc[
                (site_name, func_short_name),
                ("other_function", "weighted_error_out_of_sample"),
            ] = fun_to_check(
                opt_res.x,
                acf_lags_validate,
                corr_data_validate["acf"].astype(np.float32).values,
                corr_data_validate["pair_counts"].astype(np.float32).values,
            )
        break
    # Done fits, make plots.
    fig, axes = plt.subplots(4, 2, sharey=True, sharex=True, figsize=(6.5, 5))
    fig.suptitle("Correlation fit for {site:s}".format(site=site_name))
    axes[0, 0].set_title("Training data")
    axes[0, 1].set_title("Validation data")
    axes[0, 0].set_ylabel("Empirical\nCorrelogram")
    for ax in axes[1:, 0]:
        ax.set_ylabel("Fitted\nCorrelogram")
    axes[0, 0].plot(acf_lags_train, corr_data_train["acf"])
    axes[0, 1].plot(acf_lags_validate, corr_data_validate["acf"])
    axes[0, 0].set_ylim(-1, 1)
    max_lag = max(acf_lags_train[-1], acf_lags_validate[-1])
    axes[0, 0].set_xlim(0, max_lag)
    xticks = np.arange(0, max_lag, 365)
    for ax in axes.flat:
        ax.set_xticks(xticks)
    sorted_fits = CORRELATION_FIT_ERROR.loc[
        (site_name, slice(None)), ("function_optimized", slice(None))
    ].sort_values(
        ("function_optimized", "weighted_error_out_of_sample")
    ).dropna(how="all")
    print(sorted_fits.iloc[:3, :])
    for i, fun_name in enumerate(sorted_fits.iloc[:3, :].index.get_level_values(1), 1):
        print(fun_name)
        curve_fun = getattr(
            flux_correlation_function_fits,
            "{fun_name:s}_curve_ne".format(fun_name=fun_name),
        )
        axes[i, 0].plot(
            acf_lags_train,
            curve_fun(
                acf_lags_train,
                **COEF_DATA.loc[(site_name, fun_name)].dropna(),
            ),
        )
        axes[i, 1].plot(
            acf_lags_validate,
            curve_fun(
                acf_lags_validate,
                **COEF_DATA.loc[(site_name, fun_name)].dropna(),
            ),
        )
        axes[i, 0].set_ylabel(fun_name)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(
        "{site_name:s}-cross-validation-function-fits.png"
        .format(site_name=site_name)
    )
    plt.close(fig)

COEF_DATA.to_csv("coefficient-data-loop.csv")
COEF_VAR_DATA.to_csv("coefficient-variance-data-loop.csv")
CORRELATION_FIT_ERROR.to_csv("correlation-fit-error-loop.csv")
