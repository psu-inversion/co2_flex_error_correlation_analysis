#!/usr/bin/env python
# ~*~ coding: utf8 ~*~
"""Cross-validation fits to tower correlation data

Uses correlations from one tower to fit the correlation functions,
then measures the fit against the other towers' correlation data.

"""
from __future__ import division, print_function, unicode_literals

import datetime
import itertools
import logging
import os.path
import random

import flux_correlation_function_fits
import numpy as np
import pandas as pd
import pint
import scipy.optimize
import xarray

from correlation_function_fits import (CorrelationPart, PartForm,
                                       get_full_parameter_list,
                                       is_valid_combination)
from correlation_utils import get_autocorrelation_stats

print(datetime.datetime.now())

CORRELATION_PARTS_LIST = [
    (day_part, dm_part, ann_part)
    for day_part, dm_part, ann_part in itertools.product(PartForm, PartForm, PartForm)
    if is_valid_combination(day_part, dm_part, ann_part)
]

# Time constants
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
DAYS_PER_FORTNIGHT = 14
DAYS_PER_YEAR = 365.2425
DAYS_PER_DECADE = 10 * DAYS_PER_YEAR
HOURS_PER_YEAR = HOURS_PER_DAY * DAYS_PER_YEAR

UREG = pint.UnitRegistry()

############################################################
# Configuration
N_YEARS_DATA = 4
REQUIRED_DATA_FRAC = 0.8

N_SPLITS = 350
# There are 75 towers that fit my criteria
N_TRAINING = 45
N_HYPER_TRAIN = 30
N_CROSS_VAL = 0  # or whatever's left

CALCULATE_AUTOCORRELATIONS = False

# Configure logging
logging.basicConfig(
    format=(
        "%(asctime)s:%(levelname)7s:%(name)8s:"
        "%(module)20s:%(funcName)15s:%(lineno)03s: %(message)s"
    ),
    level=logging.DEBUG,
)
_LOGGER = logging.getLogger(__name__)


def has_enough_data(da):
    """Check whether there is enough data for a good analysis.

    Parameters
    ----------
    da: xarray.DataArray

    Returns
    -------
    bool
    """
    time_index_name = [name for name in da.dims if "time" in name.lower()][0]
    time_index = da.indexes[time_index_name]
    if len(time_index) < 1:
        _LOGGER.debug("No data")
        return False
    if time_index[-1] - time_index[0] < datetime.timedelta(
        days=N_YEARS_DATA * DAYS_PER_YEAR
    ):
        _LOGGER.debug("< 2 years")
        return False
    if (
        da.count(time_index_name).values[()]
        < REQUIRED_DATA_FRAC * N_YEARS_DATA * HOURS_PER_YEAR
    ):
        _LOGGER.debug("Missing data")
        return False
    return True


def timedelta_index_to_floats(index):
    """Turn a TimedeltaIndex into an array of floats.

    Parameters
    ----------
    index: pd.TimedeltaIndex

    Returns
    -------
    np.ndarray: Units are days
    """
    # Timedeltas can be negative, so the underlying type should be a
    # signed integer.
    lag_times = index.values.astype("m8[h]").astype("i8")
    lag_times -= lag_times[0]
    lag_times = lag_times.astype("f4") / 24
    return lag_times.astype(np.float32)


def select_tower_subset(corr_data, towers):
    """Combine tower autocorr data to make single series

    Parameters
    ----------
    corr_data: xarray.Dataset
    towers: List[str]

    Returns
    -------
    corr_data: xarray.Dataset
    acf_lags: np.ndarray
    acf_weights: np.ndarray
    """
    corr_data_towers = corr_data.sel(site=towers)

    n_pairs = corr_data_towers["flux_error_n_pairs"]
    # # Make sure elements with no data are zero
    # # All data should already be zero.
    # corr_data_towers = corr_data_towers.where(n_pairs >= 1, 0)
    n_pairs_tot = n_pairs.sum("site")
    # 0.15.1 introduces a weighted() method that does the same thing,
    # but it's only three and a half months old so far.
    corr_data = (corr_data_towers * n_pairs).sum(dim="site") / n_pairs_tot
    corr_data["flux_error_n_pairs"] = n_pairs_tot
    corr_data = corr_data.sel(
        time_lag=n_pairs_tot >= 1,
    ).dropna("time_lag")

    acf_lags = timedelta_index_to_floats(
        pd.TimedeltaIndex(corr_data.coords["time_lag"])
    )
    acf_weights = 1.0 / np.sqrt(corr_data["flux_error_n_pairs"])
    assert np.isfinite(acf_weights).all()

    return corr_data, acf_lags, acf_weights


############################################################
# Set initial values and bounds for the parameters
STARTING_PARAMS = dict(
    daily_coef=0.5,
    daily_coef1=0.7,
    daily_coef2=0.3,
    daily_width=0.5,
    daily_timescale=50,  # fortnights
    dm_width=0.8,
    dm_coef1=0.3,
    dm_coef2=+0.1,
    ann_coef1=+0.8,
    ann_coef2=+0.4,
    ann_coef=0.03,
    ann_width=0.6,
    ann_timescale=3.0,  # decades
    resid_coef=0.03,
    resid_timescale=2.0,  # fortnights
    ec_coef=0.5,
)
PARAM_LOWER_BOUNDS = dict(
    daily_coef=-10,
    daily_coef1=-10,
    daily_coef2=-10,
    daily_width=0,
    daily_timescale=0,  # fortnights
    dm_width=0,
    dm_coef1=-10,
    dm_coef2=-10,
    ann_coef1=-10,
    ann_coef2=-10,
    ann_coef=-10,
    ann_width=0,
    ann_timescale=0,  # decades
    resid_coef=-10,
    resid_timescale=0.0,  # fortnights
    ec_coef=-10,
)
PARAM_UPPER_BOUNDS = dict(
    daily_coef=10,
    daily_coef1=10,
    daily_coef2=10,
    daily_width=10,
    daily_timescale=5000,  # fortnights
    dm_width=10,
    dm_coef1=10,
    dm_coef2=10,
    ann_coef1=10,
    ann_coef2=10,
    ann_coef=10,
    ann_width=10,
    ann_timescale=40,  # decades
    resid_coef=10,
    resid_timescale=5000.0,  # fortnights
    ec_coef=10,
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
    "ameriflux-and-casa-matching-data.nc4",
    chunks={"site": 30},
)

DATA_COUNTS = AMERIFLUX_MINUS_CASA_DATA["flux_difference"].count("time").load()
SITES_TO_KEEP = [
    site
    for site in AMERIFLUX_MINUS_CASA_DATA.indexes["site"]
    if (DATA_COUNTS.sel(site=site) > HOURS_PER_YEAR * N_YEARS_DATA * REQUIRED_DATA_FRAC)
]
AMERIFLUX_MINUS_CASA_DATA = AMERIFLUX_MINUS_CASA_DATA.sel(site=SITES_TO_KEEP).persist()

TIME_LAG_INDEX = pd.timedelta_range(
    start=0, freq="1H", periods=AMERIFLUX_MINUS_CASA_DATA.dims["time"]
)

############################################################
# Calculate the autocorrelation for each tower
if CALCULATE_AUTOCORRELATIONS:
    AUTOCORRELATION_DATA = xarray.Dataset(
        {
            "flux_error_autocorrelation": (
                ("site", "time_lag"),
                np.empty(
                    (
                        AMERIFLUX_MINUS_CASA_DATA.dims["site"],
                        AMERIFLUX_MINUS_CASA_DATA.dims["time"],
                    ),
                    dtype=np.float32,
                ),
                {
                    "long_name": "ameriflux_minus_casa_surface_upward_carbon_dioxide_"
                    "flux_difference_autocorrelation",
                    "units": "1",
                },
            ),
            "flux_error_autocovariance": (
                ("site", "time_lag"),
                np.empty(
                    (
                        AMERIFLUX_MINUS_CASA_DATA.dims["site"],
                        AMERIFLUX_MINUS_CASA_DATA.dims["time"],
                    ),
                    dtype=np.float32,
                ),
                {
                    "long_name": "ameriflux_minus_casa_surface_upward_carbon_dioxide_"
                    "flux_difference_autocovariance",
                    "units": str(
                        UREG(
                            AMERIFLUX_MINUS_CASA_DATA["flux_difference"].attrs["units"]
                        )
                        ** 2
                    ),
                },
            ),
            "flux_error_n_pairs": (
                ("site", "time_lag"),
                np.empty(
                    (
                        AMERIFLUX_MINUS_CASA_DATA.dims["site"],
                        AMERIFLUX_MINUS_CASA_DATA.dims["time"],
                    ),
                    dtype=np.float32,
                ),
                {
                    "long_name": "ameriflux_minus_casa_surface_upward_carbon_dioxide_"
                    "flux_difference_number_of_pairs_at_lag",
                    "units": "1",
                },
            ),
        },
        {
            "site": (
                ("site",),
                AMERIFLUX_MINUS_CASA_DATA.coords["site"].values,
            ),
            "time_lag": (
                ("time_lag",),
                TIME_LAG_INDEX,
                {
                    "long_name": "time_difference",
                },
            ),
        },
    )

    for site in AMERIFLUX_MINUS_CASA_DATA.indexes["site"]:
        correlation_data = get_autocorrelation_stats(
            AMERIFLUX_MINUS_CASA_DATA["flux_difference"]
            .sel(site=site)
            .to_series()
            .dropna()
            .resample("1H")
            .first()
        )
        AUTOCORRELATION_DATA["flux_error_autocorrelation"].sel(site=site).isel(
            time_lag=slice(None, len(correlation_data))
        ).values[:] = correlation_data["acf"]
        AUTOCORRELATION_DATA["flux_error_autocovariance"].sel(site=site).isel(
            time_lag=slice(None, len(correlation_data))
        ).values[:] = correlation_data["acovf"]
        AUTOCORRELATION_DATA["flux_error_n_pairs"].sel(site=site).isel(
            time_lag=slice(None, len(correlation_data))
        ).values[:] = correlation_data["pair_counts"]

    # Save the new dataset
    encoding = {
        var: {"_FillValue": -9999, "zlib": True}
        for var in AUTOCORRELATION_DATA.data_vars
    }
    encoding.update({var: {"_FillValue": None} for var in AUTOCORRELATION_DATA.coords})

    AUTOCORRELATION_DATA.to_netcdf(
        "ameriflux-minus-casa-autocorrelation-data-all-towers.nc4",
        format="NETCDF4_CLASSIC",
        encoding=encoding,
    )
else:
    AUTOCORRELATION_DATA = xarray.open_dataset(
        "ameriflux-minus-casa-autocorrelation-data-all-towers.nc4",
    )

############################################################
# Trim to just the useful bits for each tower
AUTOCORRELATION_FOR_CURVE_FIT = dict()

for tower in SITES_TO_KEEP:
    corr_data = AUTOCORRELATION_DATA.sel(site=tower).dropna("time_lag")
    corr_data = corr_data.where(
        corr_data["flux_error_n_pairs"] > 0,
        drop=True,
    )
    if not has_enough_data(corr_data["flux_error_n_pairs"]):
        continue
    AUTOCORRELATION_FOR_CURVE_FIT[tower] = corr_data

LIST_OF_SITES = list(AUTOCORRELATION_FOR_CURVE_FIT)
random.shuffle(LIST_OF_SITES)
SITES_TO_FIT = LIST_OF_SITES[: N_TRAINING + N_HYPER_TRAIN]

############################################################
# Set up dataset for results
CROSS_TOWER_FIT_ERROR_DS = xarray.Dataset(
    {
        "cross_validation_error": (
            ("correlation_function", "splits"),
            np.full(
                (
                    len(CORRELATION_PARTS_LIST),
                    N_SPLITS,
                ),
                np.nan,
                # I could probably get away with storing float16, but I
                # don't think netcdf can handle that.
                dtype=np.float32,
            ),
            {
                "long_name": "flux_error_correlation_function_cross_validation_error",
                "comment": "lower is better",
                "units": "1",
                "valid_min": 0,
            },
        ),
        "optimized_parameters": (
            ("correlation_function", "splits", "parameter_name"),
            np.full(
                (
                    len(CORRELATION_PARTS_LIST),
                    N_SPLITS,
                    len(STARTING_PARAMS),
                ),
                np.nan,
                dtype=np.float32,
            ),
            {
                "long_name": "flux_error_correlation_function_fitted_parameters",
                "units": [
                    "1",  #    daily_coef = 0.2,
                    "1",  #    daily_coef1 = .7,
                    "1",  #    daily_coef2 = .3,
                    "1",  #    daily_width = .5,
                    "fortnights",  #    daily_timescale = 60,
                    "1",  #    dm_width = .8,
                    "1",  #    dm_coef1 = .3,
                    "1",  #    dm_coef2 = +.1,
                    "1",  #    ann_coef1 = +.4,
                    "1",  #    ann_coef2 = +.3,
                    "1",  #    ann_coef = 0.1,
                    "1",  #    ann_width = .3,
                    "decades",  #    ann_timescale = 3.,
                    "1",  #    resid_coef = 0.05,
                    "fortnights",  #    resid_timescale = 2.,
                    "1",  #    ec_coef = 0.7,
                ],
            },
        ),
        "optimized_parameters_estimated_covariance_matrix": (
            (
                "correlation_function",
                "splits",
                "parameter_name_adjoint",
                "parameter_name",
            ),
            np.full(
                (
                    len(CORRELATION_PARTS_LIST),
                    N_SPLITS,
                    len(STARTING_PARAMS),
                    len(STARTING_PARAMS),
                ),
                np.nan,
                dtype=np.float32,
            ),
            {
                "long_name": "flux_error_correlation_function_fitted_parameters"
                " covariance_matrix",
            },
        ),
    },
    {
        "splits": (
            ("splits",),
            np.arange(N_SPLITS, dtype=np.float32),
        ),
        "training_towers": (
            ("splits", "n_training"),
            np.full(
                (N_SPLITS, N_TRAINING),
                "\0",
                dtype="U6",
            ),
        ),
        "validation_towers": (
            ("splits", "n_validation"),
            np.full(
                (N_SPLITS, N_HYPER_TRAIN),
                "\0",
                dtype="U6",
            ),
        ),
        "correlation_function": (
            ("correlation_function",),
            [
                "daily_{0.value:s}_daily_modulation_{1.value:s}_"
                "annual_{2.value:s}".format(*parts)
                for parts in CORRELATION_PARTS_LIST
            ],
        ),
        "correlation_function_short_name": (
            ("correlation_function",),
            [
                "_".join(
                    [
                        "{0:s}{1:s}".format(
                            part.get_short_name(),
                            form.get_short_name(),
                        )
                        for part, form in zip(CorrelationPart, forms)
                    ]
                )
                for forms in CORRELATION_PARTS_LIST
            ],
        ),
        "has_daily_cycle": (
            ("correlation_function",),
            np.array(
                [forms[0] != PartForm.NONE for forms in CORRELATION_PARTS_LIST],
                dtype=bool,
            ),
            {
                "long_name": "has_daily_cycle",
                "description": "whether the associated correlation function attempts to "
                "fit a daily cycle in the autocorrelations",
            },
        ),
        "daily_cycle_has_modulation": (
            ("correlation_function",),
            np.array(
                [forms[1] != PartForm.NONE for forms in CORRELATION_PARTS_LIST],
                dtype=bool,
            ),
            {
                "long_name": "daily_cycle_has_modulation",
                "description": "whether the associated correlation function attempts to "
                "fit an annual modulation of the daily cycle in the "
                "autocorrelations",
            },
        ),
        "has_annual_cycle": (
            ("correlation_function",),
            np.array(
                [forms[2] != PartForm.NONE for forms in CORRELATION_PARTS_LIST],
                dtype=bool,
            ),
            {
                "long_name": "has_annual_cycle",
                "description": "whether the associated correlation function attempts to "
                "fit an annual cycle in the autocorrelations",
            },
        ),
        "daily_cycle_has_parameters": (
            ("correlation_function",),
            np.array(
                [
                    forms[0] != PartForm.NONE and forms[0] != PartForm.GEOSTAT
                    for forms in CORRELATION_PARTS_LIST
                ],
                dtype=bool,
            ),
            {
                "long_name": "daily_cycle_has_parameters",
                "description": "whether the associated correlation function attempts to "
                "fit parameters for a daily cycle in the autocorrelations",
            },
        ),
        "daily_cycle_modulation_has_parameters": (
            ("correlation_function",),
            np.array(
                [
                    forms[1] != PartForm.NONE and forms[1] != PartForm.GEOSTAT
                    for forms in CORRELATION_PARTS_LIST
                ],
                dtype=bool,
            ),
            {
                "long_name": "daily_cycle_modulation_has_parameters",
                "description": "whether the associated correlation function attempts to "
                "fit parameters for an annual modulation of the daily "
                "cycle in the autocorrelations",
            },
        ),
        "annual_cycle_has_parameters": (
            ("correlation_function",),
            np.array(
                [
                    forms[2] != PartForm.NONE and forms[2] != PartForm.GEOSTAT
                    for forms in CORRELATION_PARTS_LIST
                ],
                dtype=bool,
            ),
            {
                "long_name": "annual_cycle_has_parameters",
                "description": "whether the associated correlation function attempts to "
                "fit parameters for an annual cycle in the "
                "autocorrelations",
            },
        ),
        "daily_cycle": (
            ("correlation_function",),
            np.array([forms[0].value for forms in CORRELATION_PARTS_LIST]),
        ),
        "annual_modulation_of_daily_cycle": (
            ("correlation_function",),
            np.array([forms[1].value for forms in CORRELATION_PARTS_LIST]),
        ),
        "annual_cycle": (
            ("correlation_function",),
            np.array([forms[2].value for forms in CORRELATION_PARTS_LIST]),
        ),
        "parameter_name": (
            ("parameter_name",),
            np.array(list(STARTING_PARAMS.keys())),
        ),
        "parameter_name_adjoint": (
            ("parameter_name_adjoint",),
            np.array(list(STARTING_PARAMS.keys())),
        ),
    },
)

############################################################
# Actually do the cross-validation
FUNCTION_PARAMS_AND_COV = []
_LOGGER.info("Starting cross-validation")

for i in range(N_SPLITS):
    random.shuffle(SITES_TO_FIT)
    training_towers = np.array(sorted(SITES_TO_FIT[:N_TRAINING]))
    validation_towers = np.array(sorted(SITES_TO_FIT[N_TRAINING:]))
    CROSS_TOWER_FIT_ERROR_DS["training_towers"].sel(splits=i).values[
        :
    ] = training_towers
    CROSS_TOWER_FIT_ERROR_DS["validation_towers"].sel(
        splits=i,
        n_validation=slice(None, len(validation_towers)),
    ).values[:] = validation_towers

    _LOGGER.info("Split %3d: Training towers:\n%s", i, training_towers)
    corr_data_train, acf_lags_train, acf_weights_train = select_tower_subset(
        AUTOCORRELATION_DATA, training_towers
    )

    # Set up validation data as well
    corr_data_validate, acf_lags_validate, acf_weights_validate = select_tower_subset(
        AUTOCORRELATION_DATA, validation_towers
    )

    FUNCTION_PARAMS_AND_COV.append([])

    for combination in CORRELATION_PARTS_LIST:
        _LOGGER.info("Fitting function: %s", combination)
        # Get function to optimize
        func_short_name = "_".join(
            [
                "{0:s}{1:s}".format(
                    part.get_short_name(),
                    form.get_short_name(),
                )
                for part, form in zip(CorrelationPart, combination)
            ]
        )
        curve_function = getattr(
            flux_correlation_function_fits,
            "{fun_name:s}_curve_ne".format(fun_name=func_short_name),
        )
        curve_and_derivative_function = getattr(
            flux_correlation_function_fits,
            "{fun_name:s}_curve_loop".format(fun_name=func_short_name),
        )
        mismatch_function = getattr(
            flux_correlation_function_fits,
            "{fun_name}_fit_ne".format(fun_name=func_short_name),
        )

        def curve_deriv(tdata, *params):
            """Find the derivative of the curve wrt. params.

            Parameters
            ----------
            tdata: np.ndarray[N]
            params: np.ndarray[M]

            Returns
            -------
            deriv: np.ndarray[N, M]
            """
            return curve_and_derivative_function(tdata, *params)[1]

        # Set up parameters
        parameter_list = get_full_parameter_list(*combination)
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
        correlation_function_long_name = (
            "daily_{0.value:s}_daily_modulation_{1.value:s}_"
            "annual_{2.value:s}".format(*combination)
        )

        try:
            opt_params, param_cov = scipy.optimize.curve_fit(
                curve_function,
                acf_lags_train.astype(np.float32),
                corr_data_train["flux_error_autocorrelation"].astype(np.float32).values,
                starting_params.astype(np.float32),
                acf_weights_train.astype(np.float32),
                bounds=(
                    lower_bounds.astype(np.float32),
                    upper_bounds.astype(np.float32),
                ),
                # jac=curve_deriv,
            )
        except (RuntimeError, ValueError) as err:
            _LOGGER.error("Curve fit failed, next split")
            _LOGGER.exception(err)
            _LOGGER.debug("Lower bounds:\n%s", lower_bounds.astype(np.float32))
            _LOGGER.debug("Upper bounds:\n%s", upper_bounds.astype(np.float32))
            _LOGGER.debug("Starting params:\n%s", starting_params.astype(np.float32))
            _LOGGER.debug("ACF weights:\n%s", acf_weights_train.astype(np.float32))
            _LOGGER.debug("ACF lags:\n%s", acf_lags_train.astype(np.float32))
            _LOGGER.debug(
                "Corr data:\n%s",
                corr_data_train["flux_error_autocorrelation"].astype(np.float32).values,
            )
            if err.message == "array must not contain infs or NaNs":
                raise
            continue

        FUNCTION_PARAMS_AND_COV[-1].append(
            xarray.Dataset(
                {
                    "optimized_parameters": (
                        ("parameter_name",),
                        opt_params,
                    ),
                    "optimized_parameters_estimated_covariance_matrix": (
                        ("parameter_name_adjoint", "parameter_name"),
                        param_cov,
                    ),
                },
                {
                    "parameter_name": (("parameter_name",), parameter_list),
                    "parameter_name_adjoint": (
                        ("parameter_name_adjoint",),
                        parameter_list,
                    ),
                    "training_towers": (
                        (
                            "splits",
                            "n_training",
                        ),
                        training_towers.reshape(1, -1),
                    ),
                    "correlation_function": ((), correlation_function_long_name),
                    "splits": (
                        ("splits",),
                        np.array([i], dtype="i2"),
                    ),
                },
            )
        )

        CROSS_TOWER_FIT_ERROR_DS["cross_validation_error"].sel(
            correlation_function=(correlation_function_long_name),
            splits=i,
        ).values[()] = mismatch_function(
            opt_params,
            acf_lags_validate,
            corr_data_validate["flux_error_autocorrelation"].astype(np.float32).values,
            corr_data_validate["flux_error_n_pairs"].astype(np.float32).values,
        )
        _LOGGER.info("Done fit and cross-validation")
    _LOGGER.info("Done cross-validation loop %d", i)

FUNCTION_PARAMS_AND_COV_DS = xarray.concat(
    [
        xarray.concat(ds_list, dim="correlation_function")
        for ds_list in FUNCTION_PARAMS_AND_COV
        if len(ds_list) > 0
    ],
    dim="splits",
)
CROSS_TOWER_FIT_ERROR_DS = CROSS_TOWER_FIT_ERROR_DS.update(FUNCTION_PARAMS_AND_COV_DS)

encoding = {
    name: {"_FillValue": -9.999e9, "zlib": True}
    for name in CROSS_TOWER_FIT_ERROR_DS.data_vars
}
encoding.update(
    {name: {"_FillValue": None} for name in CROSS_TOWER_FIT_ERROR_DS.coords}
)
rerun_number = 1
while True:
    print(rerun_number)
    save_name = (
        "ameriflux-minus-casa-autocorrelation-function-multi-tower-fits"
        "-{:03d}splits-run{:01d}.nc4".format(N_SPLITS, rerun_number)
    )
    if not os.path.exists(save_name):
        break
    rerun_number += 1



CROSS_TOWER_FIT_ERROR_DS.to_netcdf(
    save_name,
    format="NETCDF4",
    encoding=encoding,
)
_LOGGER.info("Saved output")
