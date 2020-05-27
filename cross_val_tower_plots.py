#!/usr/bin/env python
# ~*~ coding: utf8 ~*~
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

AMERIFLUX_MINUS_CASA_DATA = xarray.open_dataset(
    "ameriflux-and-casa-matching-data-2.nc4",
    chunks={"site": 30},
)

DATA_COUNTS = AMERIFLUX_MINUS_CASA_DATA["flux_difference"].count("time").load()

AUTOCORRELATION_DATA = xarray.open_dataset(
    "ameriflux-minus-casa-autocorrelation-data-all-towers.nc4",
)

CROSS_TOWER_FIT_ERROR_DS = xarray.open_dataset(
    "ameriflux-minus-casa-autocorrelation-function-fits.nc4",
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
# define normalizations for the plots
NORMALIZATIONS = {
    "one": 1,
    "num pairs": (
        AUTOCORRELATION_DATA["flux_error_n_pairs"]
        .sum("time_lag")
        .rename(site="validation_tower")
    ),
    "num differences": DATA_COUNTS.rename(site="validation_tower").load(),
    "mean": CROSS_TOWER_FIT_ERROR_DS["cross_validation_error"].mean(
        ("correlation_function", "training_tower")
    ),
}



############################################################
# Create the plots for each sort order and normalization
for norm_name, norm_val in NORMALIZATIONS.items():
    normalized = CROSS_TOWER_FIT_ERROR_DS["cross_validation_error"] / norm_val
    min_err = max(
        normalized.where(
            normalized.coords["training_tower"] !=
            normalized.coords["validation_tower"]
        ).min(),
        0
    )
    max_err = normalized.where(
        normalized.coords["training_tower"] !=
        normalized.coords["validation_tower"]
    ).quantile(0.95)

    for sort_name, sort_key in SORT_KEYS.items():
        sorted_towers = sorted(normalized.coords["training_tower"].values, key=sort_key)
        fig, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(6.5, 7))
        for ax in axes.flat:
            ax.set_visible(False)
        for corr_fun, ax in zip(
                normalized.coords["correlation_function"], axes.flat
        ):
            ax.set_visible(True)
            image = ax.imshow(
                normalized.sel(
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
            "Cross-Validation Error{normalization:s}\n"
            "Sort order: {name:s}".format(
                name=sort_name,
                normalization=(
                    " over {0:s}".format(norm_name)
                    if norm_name != "one" else ""
                )
            )
        )
        fig.savefig(
            "tower-cross-validation-mismatch-over-{normalization:s}-sort-{name:s}.png"
            .format(
                name=sort_name.replace(" ", "-"),
                normalization=norm_name.replace(" ", "-"),
            )
        )
        plt.close(fig)
