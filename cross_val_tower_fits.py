#!/usr/bin/env python
# ~*~ coding: utf8 ~*~
"""Cross-validation fits to tower correlation data

Uses correlations from one tower to fit the correlation functions,
then measures the fit against the other towers' correlation data.

"""
from __future__ import division, print_function, unicode_literals

import datetime
import itertools

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd
import pint
import xarray

import flux_correlation_function_fits
from correlation_utils import get_autocorrelation_stats

from correlation_function_fits import (
    CorrelationPart, PartForm,
    is_valid_combination,
    get_full_parameter_list,
)

CORRELATION_PARTS_LIST = [
    (day_part, dm_part, ann_part)
    for day_part, dm_part, ann_part in itertools.product(
        PartForm, PartForm, PartForm
    )
    if is_valid_combination(day_part, dm_part, ann_part)
]

# Time constants
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
DAYS_PER_FORTNIGHT = 14
DAYS_PER_YEAR = 365.2425
DAYS_PER_DECADE = 10 * DAYS_PER_YEAR
HOURS_PER_YEAR = HOURS_PER_DAY * DAYS_PER_YEAR

N_YEARS_DATA = 4
REQUIRED_DATA_FRAC = 0.8

UREG = pint.UnitRegistry()


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
    np.ndarray: Units are days
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
    "ameriflux-and-casa-matching-data-2.nc4",
    chunks={"site": 30},
)

DATA_COUNTS = AMERIFLUX_MINUS_CASA_DATA["flux_difference"].count("time").load()
SITES_TO_KEEP = [
    site
    for site in AMERIFLUX_MINUS_CASA_DATA.indexes["site"]
    if (
        DATA_COUNTS.sel(site=site) >
        HOURS_PER_YEAR * N_YEARS_DATA * REQUIRED_DATA_FRAC
    )
]
AMERIFLUX_MINUS_CASA_DATA = AMERIFLUX_MINUS_CASA_DATA.sel(
    site=SITES_TO_KEEP
).persist()

############################################################
# Set up dataset for results
CROSS_TOWER_FIT_ERROR_DS = xarray.Dataset(
    {
        "cross_validation_error": (
            ("correlation_function", "training_tower", "validation_tower"),
            np.full(
                (
                    len(CORRELATION_PARTS_LIST),
                    AMERIFLUX_MINUS_CASA_DATA.dims["site"],
                    AMERIFLUX_MINUS_CASA_DATA.dims["site"],
                ),
                np.nan,
                # I could probably get away with storing float16, but I
                # don't think netcdf can handle that.
                dtype=np.float32
            ),
            {
                "long_name":
                "flux_error_correlation_function_cross_validation_error",
                "comment": "lower is better",
                "units": "1",
                "valid_min": 0,
            },
        ),
        "optimized_parameters": (
            ("correlation_function", "training_tower", "parameter_name"),
            np.full(
                (
                    len(CORRELATION_PARTS_LIST),
                    len(SITES_TO_KEEP),
                    len(STARTING_PARAMS),
                ),
                np.nan,
                dtype=np.float32,
            ),
            {
                "long_name":
                "flux_error_correlation_function_fitted_parameters",
                "units": [
                    "1",  #    daily_coef = 0.2,
                    "1",  #    daily_coef1 = .7,
                    "1",  #    daily_coef2 = .3,
                    "1",  #    daily_width = .5,
                    "fortnights",  #    daily_timescale = 60,  # fortnights
                    "1",  #    dm_width = .8,
                    "1",  #    dm_coef1 = .3,
                    "1",  #    dm_coef2 = +.1,
                    "1",  #    ann_coef1 = +.4,
                    "1",  #    ann_coef2 = +.3,
                    "1",  #    ann_coef = 0.1,
                    "1",  #    ann_width = .3,
                    "decades",  #    ann_timescale = 3.,  # decades
                    "1",  #    resid_coef = 0.05,
                    "fortnights",  #    resid_timescale = 2.,  # fortnights
                    "1",  #    ec_coef = 0.7,
                    "hours",  #    ec_timescale = 2.,  # hours
                ],
            },
        ),
        "optimized_parameters_estimated_covariance_matrix": (
            ("correlation_function", "training_tower",
             "parameter_name_adjoint", "parameter_name"),
            np.full(
                (
                    len(CORRELATION_PARTS_LIST),
                    len(SITES_TO_KEEP),
                    len(STARTING_PARAMS),
                    len(STARTING_PARAMS),
                ),
                np.nan,
                dtype=np.float32,
            ),
            {
                "long_name":
                "flux_error_correlation_function_fitted_parameters"
                " covariance_matrix",
            },
        ),
    },

    {
        "training_tower": (
            ("training_tower",),
            AMERIFLUX_MINUS_CASA_DATA.coords["site"],
        ),
        "validation_tower": (
            ("validation_tower",),
            AMERIFLUX_MINUS_CASA_DATA.coords["site"],
        ),
        "correlation_function": (
            ("correlation_function",),
            [
                "daily_{0.name:s}_daily_modulation_{1.name:s}_"
                "annual_{2.name:s}".format(*parts)
                for parts in CORRELATION_PARTS_LIST
            ]
        ),
        "correlation_function_short_name": (
            ("correlation_function",),
            [
                "_".join([
                    "{0:s}{1:s}".format(
                        part.get_short_name(),
                        form.get_short_name(),
                    )
                    for part, form in zip(CorrelationPart, forms)
                ])
                for forms in CORRELATION_PARTS_LIST
            ],
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

TIME_LAG_INDEX = pd.timedelta_range(
    start=0, freq="1H", periods=AMERIFLUX_MINUS_CASA_DATA.dims["time"]
)

# ############################################################
# # Calculate the autocorrelation for each tower
### Commented out because I already have this calculated and can just
### read in the saved file.
# AUTOCORRELATION_DATA = xarray.Dataset(
#     {
#         "flux_error_autocorrelation": (
#             ("site", "time_lag"),
#             np.empty(
#                 (
#                     AMERIFLUX_MINUS_CASA_DATA.dims["site"],
#                     AMERIFLUX_MINUS_CASA_DATA.dims["time"],
#                 ),
#                 dtype=np.float32
#             ),
#             {
#                 "long_name":
#                 "ameriflux_minus_casa_surface_upward_carbon_dioxide_"
#                 "flux_difference_autocorrelation",
#                 "units": "1",
#             },
#         ),
#         "flux_error_autocovariance": (
#             ("site", "time_lag"),
#             np.empty(
#                 (
#                     AMERIFLUX_MINUS_CASA_DATA.dims["site"],
#                     AMERIFLUX_MINUS_CASA_DATA.dims["time"],
#                 ),
#                 dtype=np.float32
#             ),
#             {
#                 "long_name":
#                 "ameriflux_minus_casa_surface_upward_carbon_dioxide_"
#                 "flux_difference_autocovariance",
#                 "units": str(
#                     UREG(
#                         AMERIFLUX_MINUS_CASA_DATA[
#                             "flux_difference"
#                         ].attrs["units"]
#                     ) ** 2
#                 ),
#             },
#         ),
#         "flux_error_n_pairs": (
#             ("site", "time_lag"),
#             np.empty(
#                 (
#                     AMERIFLUX_MINUS_CASA_DATA.dims["site"],
#                     AMERIFLUX_MINUS_CASA_DATA.dims["time"],
#                 ),
#                 dtype=np.float32
#             ),
#             {
#                 "long_name":
#                 "ameriflux_minus_casa_surface_upward_carbon_dioxide_"
#                 "flux_difference_number_of_pairs_at_lag",
#                 "units": "1",
#             },
#         ),
#     },
#     {
#         "site": (
#             ("site",),
#             AMERIFLUX_MINUS_CASA_DATA.coords["site"],
#         ),
#         "time_lag": (
#             ("time_lag",),
#             TIME_LAG_INDEX,
#             {
#                 "long_name": "time_difference",
#             },
#         )
#     },
# )

# for site in AMERIFLUX_MINUS_CASA_DATA.indexes["site"]:
#     correlation_data = get_autocorrelation_stats(
#         AMERIFLUX_MINUS_CASA_DATA["flux_difference"]
#         .sel(site=site)
#         .to_series()
#         .dropna()
#         .resample("1H").first()
#     )
#     AUTOCORRELATION_DATA["flux_error_autocorrelation"].sel(
#         site=site
#     ).isel(
#         time_lag=slice(None, len(correlation_data))
#     ).values[:] = correlation_data["acf"]
#     AUTOCORRELATION_DATA["flux_error_autocovariance"].sel(
#         site=site
#     ).isel(
#         time_lag=slice(None, len(correlation_data))
#     ).values[:] = correlation_data["acovf"]
#     AUTOCORRELATION_DATA["flux_error_n_pairs"].sel(
#         site=site
#     ).isel(
#         time_lag=slice(None, len(correlation_data))
#     ).values[:] = correlation_data["pair_counts"]

# encoding = {var: {"_FillValue": -9999, "zlib": True}
#             for var in AUTOCORRELATION_DATA.data_vars}
# encoding.update({var: {"_FillValue": None}
#                  for var in AUTOCORRELATION_DATA.coords})

# AUTOCORRELATION_DATA.to_netcdf(
#     "ameriflux-minus-casa-autocorrelation-data-all-towers.nc4",
#     format="NETCDF4_CLASSIC", encoding=encoding
# )

AUTOCORRELATION_DATA = xarray.open_dataset(
    "ameriflux-minus-casa-autocorrelation-data-all-towers.nc4",
)

############################################################
# Trim to just the useful bits for each tower
AUTOCORRELATION_FOR_CURVE_FIT = dict()

for tower in SITES_TO_KEEP:
    corr_data = AUTOCORRELATION_DATA.sel(
        site=tower
    ).dropna("time_lag")
    corr_data = corr_data.where(
        corr_data["flux_error_n_pairs"] > 0,
        drop=True,
    )
    if not has_enough_data(corr_data["flux_error_n_pairs"]):
        continue
    AUTOCORRELATION_FOR_CURVE_FIT[tower] = corr_data

############################################################
# Actually do the cross-validation
FUNCTION_PARAMS_AND_COV = []

for combination in CORRELATION_PARTS_LIST:
    print(combination)
    # Get function to optimize
    func_short_name = "_".join([
        "{0:s}{1:s}".format(
            part.get_short_name(),
            form.get_short_name(),
        )
        for part, form in zip(CorrelationPart, combination)
    ])
    curve_function = getattr(
        flux_correlation_function_fits,
        "{fun_name:s}_curve_ne".format(
            fun_name=func_short_name
        )
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
    FUNCTION_PARAMS_AND_COV.append([])
    correlation_function_long_name = (
        "daily_{0.name:s}_daily_modulation_{1.name:s}_"
        "annual_{2.name:s}".format(*combination)
    )
    for training_tower in AUTOCORRELATION_FOR_CURVE_FIT:
        print("Now training on:", training_tower)
        corr_data_train = AUTOCORRELATION_FOR_CURVE_FIT[training_tower]
        acf_lags_train = timedelta_index_to_floats(
            corr_data_train.indexes["time_lag"]
        )
        acf_weights_train = 1. / np.sqrt(corr_data_train["flux_error_n_pairs"])
        try:
            opt_params, param_cov = scipy.optimize.curve_fit(
                curve_function,
                acf_lags_train.astype(np.float32),
                corr_data_train[
                    "flux_error_autocorrelation"
                ].astype(np.float32).values,
                starting_params.astype(np.float32),
                acf_weights_train.astype(np.float32),
                bounds=(
                    lower_bounds.astype(np.float32),
                    upper_bounds.astype(np.float32),
                ),
                jac=curve_deriv,
            )
        except (RuntimeError, ValueError) as err:
            print(err, "Curve fit failed, next tower", sep="\n")
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
                        param_cov
                    ),
                },
                {
                    "parameter_name": (("parameter_name",), parameter_list),
                    "parameter_name_adjoint": (
                        ("parameter_name_adjoint",), parameter_list
                    ),
                    "training_tower": ((), training_tower),
                    "correlation_function": (
                        (), correlation_function_long_name
                    ),
                },
            )
        )
        for validation_tower in AUTOCORRELATION_FOR_CURVE_FIT:

            corr_data_validate = AUTOCORRELATION_FOR_CURVE_FIT[
                validation_tower
            ]
            acf_lags_validate = timedelta_index_to_floats(
                corr_data_validate.indexes["time_lag"]
            )
            acf_weights_validate = 1. / np.sqrt(
                corr_data_validate["flux_error_n_pairs"]
            )
            CROSS_TOWER_FIT_ERROR_DS["cross_validation_error"].sel(
                correlation_function=(
                    correlation_function_long_name
                ),
                training_tower=training_tower,
                validation_tower=validation_tower,
            ).values[()] = mismatch_function(
                opt_params,
                acf_lags_validate,
                corr_data_validate["flux_error_autocorrelation"].astype(
                    np.float32
                ).values,
                corr_data_validate["flux_error_n_pairs"].astype(
                    np.float32
                ).values
            )

FUNCTION_PARAMS_AND_COV_DS = xarray.concat(
    [xarray.concat(ds_list, dim="training_tower")
     for ds_list in FUNCTION_PARAMS_AND_COV],
    dim="correlation_function"
)
CROSS_TOWER_FIT_ERROR_DS = CROSS_TOWER_FIT_ERROR_DS.update(
    FUNCTION_PARAMS_AND_COV_DS
)

encoding = {name: {"_FillValue": -9.999e9, "zlib": True}
            for name in CROSS_TOWER_FIT_ERROR_DS.data_vars}
encoding.update({name: {"_FillValue": None}
                 for name in CROSS_TOWER_FIT_ERROR_DS.coords})
CROSS_TOWER_FIT_ERROR_DS.to_netcdf(
    "ameriflux-minus-casa-autocorrelation-function-fits.nc4",
    format="NETCDF4", encoding=encoding
)

############################################################
# Define sort orders for plots
SORT_KEYS = {
    "alphabetical": lambda site: site,
    "vegetation": lambda site: (
        AMERIFLUX_MINUS_CASA_DATA["Vegetation"].sel(site=site).values[()],
        site
    ),
    "climate class": lambda site: (
        AMERIFLUX_MINUS_CASA_DATA["Climate_Cl"].sel(site=site).values[()],
        site
    ),
    "latitude": lambda site: (
        AMERIFLUX_MINUS_CASA_DATA["Latitude"].sel(site=site).values[()],
        site
    ),
    "longitude": lambda site: (
        AMERIFLUX_MINUS_CASA_DATA["Longitude"].sel(site=site).values[()],
        site
    ),
    "mean temp": lambda site: (
        AMERIFLUX_MINUS_CASA_DATA["Mean_Temp"].sel(site=site).values[()],
        site
    ),
    "mean precip": lambda site: (
        AMERIFLUX_MINUS_CASA_DATA["Mean_Preci"].sel(site=site).values[()],
        site
    ),
}


############################################################
# Unnormalized plots, just the raw cross-validation error
min_err = max(
    CROSS_TOWER_FIT_ERROR_DS["cross_validation_error"].where(
        CROSS_TOWER_FIT_ERROR_DS.coords["training_tower"] !=
        CROSS_TOWER_FIT_ERROR_DS.coords["validation_tower"]
    ).min(),
    0
)
max_err = CROSS_TOWER_FIT_ERROR_DS["cross_validation_error"].where(
    CROSS_TOWER_FIT_ERROR_DS.coords["training_tower"] !=
    CROSS_TOWER_FIT_ERROR_DS.coords["validation_tower"]
).quantile(0.95)

for sort_name, sort_key in SORT_KEYS.items():
    sorted_towers = sorted(AUTOCORRELATION_FOR_CURVE_FIT.keys(), key=sort_key)
    fig, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(6.5, 7))
    for ax in axes.flat:
        ax.set_visible(False)
    for corr_fun, ax in zip(
            CROSS_TOWER_FIT_ERROR_DS.coords["correlation_function"], axes.flat
    ):
        ax.set_visible(True)
        image = ax.imshow(
            CROSS_TOWER_FIT_ERROR_DS["cross_validation_error"].sel(
                correlation_function=corr_fun,
                training_tower=sorted_towers,
                validation_tower=sorted_towers,
            ),
            vmin=min_err, vmax=max_err
        )
        ax.set_title(corr_fun.coords["correlation_function_short_name"].values)
    fig.subplots_adjust(hspace=0.4)
    cbar = fig.colorbar(
        image, ax=axes[-1, 1:], orientation="horizontal", extend="both",
        fraction=1
    )
    cbar.set_label(
        "Cross-Validation Error\n"
        "Sort order: {name:s}".format(name=sort_name)
    )
    fig.savefig(
        "tower-cross-validation-raw-mismatch-sort-{name:s}.png"
        .format(name=sort_name.replace(" ", "-"))
    )
    plt.close(fig)

############################################################
# Plots normalized by the number of pairs going into the autocorrelations
TOTAL_N_PAIRS = (
    AUTOCORRELATION_DATA["flux_error_n_pairs"]
    .sum("time_lag")
    .rename(site="validation_tower")
)
PAIRS_NORMALIZED_CV_ERR = (
    CROSS_TOWER_FIT_ERROR_DS["cross_validation_error"] /
    TOTAL_N_PAIRS
)

pairs_min_err = max(
    PAIRS_NORMALIZED_CV_ERR.where(
        PAIRS_NORMALIZED_CV_ERR.coords["training_tower"] !=
        PAIRS_NORMALIZED_CV_ERR.coords["validation_tower"]
    ).min(),
    0
)
pairs_max_err = PAIRS_NORMALIZED_CV_ERR.where(
    PAIRS_NORMALIZED_CV_ERR.coords["training_tower"] !=
    PAIRS_NORMALIZED_CV_ERR.coords["validation_tower"]
).quantile(0.95)


for sort_name, sort_key in SORT_KEYS.items():
    sorted_towers = sorted(AUTOCORRELATION_FOR_CURVE_FIT.keys(), key=sort_key)
    fig, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(6.5, 7))
    for ax in axes.flat:
        ax.set_visible(False)
    for corr_fun, ax in zip(
            PAIRS_NORMALIZED_CV_ERR.coords["correlation_function"], axes.flat
    ):
        ax.set_visible(True)
        image = ax.imshow(
            PAIRS_NORMALIZED_CV_ERR.sel(
                correlation_function=corr_fun,
                training_tower=sorted_towers,
                validation_tower=sorted_towers,
            ),
            vmin=pairs_min_err, vmax=pairs_max_err
        )
        ax.set_title(corr_fun.coords["correlation_function_short_name"].values)
    fig.subplots_adjust(hspace=0.4)
    cbar = fig.colorbar(
        image, ax=axes[-1, 1:], orientation="horizontal", extend="both",
        fraction=1
    )
    cbar.set_label(
        "Cross-validation error normalized by number of pairs\n"
        "Sort order: {name:s}".format(name=sort_name)
    )
    fig.savefig(
        "tower-cross-validation-pairs-normalized-mismatch-sort-{name:s}.png"
        .format(name=sort_name.replace(" ", "-"))
    )
    plt.close(fig)

############################################################
# Plots normalized by the number of differences going into the autocorrelations
N_DIFFERENCES = AMERIFLUX_MINUS_CASA_DATA["flux_difference"].count(
    "time"
).rename(site="validation_tower").load()

DIFFERENCES_NORMALIZED_CV_ERR = (
    CROSS_TOWER_FIT_ERROR_DS["cross_validation_error"] /
    N_DIFFERENCES
)

differences_min_err = max(
    DIFFERENCES_NORMALIZED_CV_ERR.where(
        DIFFERENCES_NORMALIZED_CV_ERR.coords["training_tower"] !=
        DIFFERENCES_NORMALIZED_CV_ERR.coords["validation_tower"]
    ).min(),
    0
)
differences_max_err = DIFFERENCES_NORMALIZED_CV_ERR.where(
    DIFFERENCES_NORMALIZED_CV_ERR.coords["training_tower"] !=
    DIFFERENCES_NORMALIZED_CV_ERR.coords["validation_tower"]
).quantile(0.95)

for sort_name, sort_key in SORT_KEYS.items():
    sorted_towers = sorted(AUTOCORRELATION_FOR_CURVE_FIT.keys(), key=sort_key)
    fig, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(6.5, 7))
    for ax in axes.flat:
        ax.set_visible(False)
    for corr_fun, ax in zip(
            DIFFERENCES_NORMALIZED_CV_ERR.coords["correlation_function"],
            axes.flat
    ):
        ax.set_visible(True)
        image = ax.imshow(
            DIFFERENCES_NORMALIZED_CV_ERR.sel(
                correlation_function=corr_fun,
                training_tower=sorted_towers,
                validation_tower=sorted_towers,
            ),
            vmin=differences_min_err, vmax=differences_max_err
        )
        ax.set_title(corr_fun.coords["correlation_function_short_name"].values)
    fig.subplots_adjust(hspace=0.4)
    cbar = fig.colorbar(
        image, ax=axes[-1, 1:], orientation="horizontal", extend="both",
        fraction=1
    )
    cbar.set_label(
        "Cross-validation error over number of differences\n"
        "Sort order: {name:s}".format(name=sort_name)
    )
    fig.savefig(
        "tower-cross-validation-differences-normalized-mismatch-"
        "sort-{name:s}.png"
        .format(name=sort_name.replace(" ", "-"))
    )
    plt.close(fig)

############################################################
# Plots normalized by the mean CV error for each validation tower
MEAN_CV_ERR = CROSS_TOWER_FIT_ERROR_DS["cross_validation_error"].mean(
    ("correlation_function", "training_tower")
)

MEAN_NORMALIZED_CV_ERR = (
    CROSS_TOWER_FIT_ERROR_DS["cross_validation_error"] /
    MEAN_CV_ERR
)

mean_min_err = max(
    MEAN_NORMALIZED_CV_ERR.where(
        MEAN_NORMALIZED_CV_ERR.coords["training_tower"] !=
        MEAN_NORMALIZED_CV_ERR.coords["validation_tower"]
    ).min(),
    0
)
mean_max_err = MEAN_NORMALIZED_CV_ERR.where(
    MEAN_NORMALIZED_CV_ERR.coords["training_tower"] !=
    MEAN_NORMALIZED_CV_ERR.coords["validation_tower"]
).quantile(0.95)

for sort_name, sort_key in SORT_KEYS.items():
    sorted_towers = sorted(AUTOCORRELATION_FOR_CURVE_FIT.keys(), key=sort_key)
    fig, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(6.5, 7))
    for ax in axes.flat:
        ax.set_visible(False)
    for corr_fun, ax in zip(
            MEAN_NORMALIZED_CV_ERR.coords["correlation_function"], axes.flat
    ):
        ax.set_visible(True)
        image = ax.imshow(
            MEAN_NORMALIZED_CV_ERR.sel(
                correlation_function=corr_fun,
                training_tower=sorted_towers,
                validation_tower=sorted_towers,
            ),
            vmin=mean_min_err, vmax=mean_max_err
        )
        ax.set_title(corr_fun.coords["correlation_function_short_name"].values)
    fig.subplots_adjust(hspace=0.4)
    cbar = fig.colorbar(
        image, ax=axes[-1, 1:], orientation="horizontal", extend="both",
        fraction=1
    )
    cbar.set_label(
        "Cross-validation error over validation-tower mean\n"
        "Sort order: {name:s}".format(name=sort_name)
    )
    fig.savefig(
        "tower-cross-validation-mean-normalized-mismatch-sort-{name:s}.png"
        .format(name=sort_name.replace(" ", "-"))
    )
    plt.close(fig)
