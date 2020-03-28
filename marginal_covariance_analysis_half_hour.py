#!/usr/bin/env python
# ~*~ coding: utf8 ~*~
# pylint: disable=invalid-name
"""Find marginal covariances in space and time."""
from __future__ import print_function, division
import itertools
import datetime
import inspect

import numpy as np
from numpy import exp, sin, cos, square, newaxis
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize

import pyproj
import cartopy.crs as ccrs
import xarray
from statsmodels.tsa.stattools import acovf, acf
from statsmodels.tools.eval_measures import aic, aicc, bic, hqic
from bottleneck import nansum

MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
MINUTES_PER_DAY = MINUTES_PER_HOUR * HOURS_PER_DAY
DAYS_PER_DAY = 1
DAYS_PER_YEAR = 365.2425
DAYS_PER_WEEK = 7

GLOBE = ccrs.Globe(semimajor_axis=6370000, semiminor_axis=6370000)
PROJECTION = ccrs.LambertConformal(
    standard_parallels=(30, 60),
    central_latitude=40,
    central_longitude=-96,
    globe=GLOBE,
)
GEOD = pyproj.Geod(sphere=True, a=6370000)

PI_OVER_DAY = np.pi / DAYS_PER_DAY
TWO_PI_OVER_DAY = 2 * PI_OVER_DAY
PI_OVER_YEAR = np.pi / DAYS_PER_YEAR
TWO_PI_OVER_YEAR = 2 * PI_OVER_YEAR
FOUR_PI_OVER_YEAR = 4 * PI_OVER_YEAR

amf_ds = xarray.open_dataset(
    "/abl/s0/Continent/dfw5129/ameriflux_netcdf/"
    "AmeriFlux_single_value_per_tower_half_hour_data.nc4"
)
casa_ds = xarray.open_mfdataset(
    ("/mc1s2/s4/dfw5129/casa_downscaling/"
     "200?-0?_downscaled_CASA_L2_Ensemble_Mean_Biogenic_NEE_Ameriflux.nc4"),
    combine="by_coords"
)

sites_in_both = sorted(set(casa_ds.coords["Site_Id"].values) &
                       set(amf_ds.coords["site"].values))
times_in_both = sorted(set(casa_ds.coords["time"].values) &
                       set(amf_ds.coords["TIMESTAMP_START"].values))
amf_data = amf_ds["ameriflux_carbon_dioxide_flux_estimate"].sel(
    site=sites_in_both, TIMESTAMP_START=times_in_both
).stack(
    data_point=("site", "TIMESTAMP_START")
).dropna("data_point")

casa_data = casa_ds["NEE"].set_index(
    ameriflux_tower_location="Site_Id"
).sel(
    ameriflux_tower_location=amf_data.coords["site"], time=amf_data.coords["TIMESTAMP_START"]
).dropna("data_point")
difference = (amf_data - casa_data).dropna("data_point")

coords = np.empty((difference.shape[0], 3), dtype=np.float32)
values = np.array(difference.values, dtype=np.float32)

data_times_int = difference.coords["TIMESTAMP_START"].values.astype("M8[m]").astype("i8")
data_times_int -= data_times_int[0]
coords[:, 0] = data_times_int.astype(np.float32)
coords[:, 0] /= MINUTES_PER_DAY
# Set to map coordinates in meters
coords[:, 1:] = PROJECTION.transform_points(
    PROJECTION.as_geodetic(),
    difference.coords["Longitude"].values,
    difference.coords["Latitude"].values
)[:, :2]
# Convert to kilometers
coords[:, 1:] /= 1e3

to_delete = [
    coord_name for coord_name in difference.coords
    if (coord_name not in difference.dims and
        "LOCATION" not in coord_name and
        "lat" not in coord_name.lower()
        and "lon" not in coord_name.lower())
]
for coord_name in to_delete:
    del difference.coords[coord_name]

hour_data = np.column_stack([coords, values.astype(np.float32)])
# assert amf_data.attrs["units"] == "umol/m2/s"
# assert casa_data.attrs["units"] == "umol/m2/s"
hour_df = pd.DataFrame(hour_data, columns=["time_days", "x_km", "y_km", "flux_diff_umol_m2_s"])
hour_df.to_csv("ameriflux_minus_casa_hour_towers.csv")

difference_df_rect = difference.to_dataframe(
    name="ameriflux_minus_casa_hour_towers_umol_m2_s"
)["ameriflux_minus_casa_hour_towers_umol_m2_s"].unstack(0)

# Will be in meters
distance_matrix = pd.DataFrame(
    index=difference_df_rect.columns,
    columns=difference_df_rect.columns,
    dtype=np.float64
)
site_coords = amf_ds.coords["site"].sel(
    site=difference_df_rect.columns
)
for site1, site2 in itertools.product(site_coords, site_coords):
    distance_matrix.loc[site1.values[()], site2.values[()]] = GEOD.line_length(
        [site1.coords["LOCATION_LONG"], site2.coords["LOCATION_LONG"]],
        [site1.coords["LOCATION_LAT"], site2.coords["LOCATION_LAT"]]
    )

# Convert distance to kilometers
# Will improve conditioning of later problems
distance_matrix /= 1000

length_opt =  scipy.optimize.minimize_scalar(
    fun=lambda length, corr, dist: nansum(square(corr - np.exp(-dist / length))),
    args=(difference_df_rect.corr().values, distance_matrix.values),
    bounds=(1, 1e3), method="bounded"
)
print("Optimizing length alone:\n", length_opt)

length_with_nugget_opt = scipy.optimize.minimize(
    fun=lambda params, corr, dist: nansum(square(
        corr - (
            params[0] * exp(-dist / params[1]) +
            (1 - params[0])
        )
    )),
    # Nondimensional, meters
    x0=[.8, 200],
    args=(difference_df_rect.corr().values, distance_matrix.values),
)
print("Optimizing length with nugget effect:",
      "\nWeight on correlated part:", length_with_nugget_opt.x[0],
      "\nCorrelation length:", length_with_nugget_opt.x[1],
      "\nConvergence:", length_with_nugget_opt.success, length_with_nugget_opt.message,
      "\nInverse Hessian:\n", length_with_nugget_opt.hess_inv)

acovf_index = pd.timedelta_range(start=0, freq="1H", periods=24 * 365 * 7)
acovf_data = pd.DataFrame(index=acovf_index)
acf_data = pd.DataFrame(index=acovf_index)

for column in difference_df_rect.columns:
    col_data = difference_df_rect.loc[:, column].dropna()
    if col_data.shape[0] == 0:
        continue
    col_data = col_data.resample("1H").mean()
    acovf_col = acovf(col_data, missing="conservative")
    nlags = len(acovf_col)
    acovf_data.loc[acovf_index[:nlags], column] = acovf_col
    acf_col = acf(col_data, missing="conservative", nlags=nlags, unbiased=True)
    acf_data.loc[acovf_index[:nlags], column] = acf_col

to_fit = acf_data.loc[~acf_data.isna().all(axis=1), :]
time_in_days = to_fit.index.values.astype("m8[h]").astype(np.int64) / 24

# corr_to_fit, time_in_days = np.broadcast_arrays(to_fit.values, time_in_days[:, newaxis])
# not_nan = np.isfinite(corr_to_fit)
# corr_to_fit = corr_to_fit[not_nan]
# time_in_days = time_in_days[not_nan]

single_time_opt = scipy.optimize.minimize_scalar(
    lambda length, acorr, times: nansum(square(
            acorr - exp(-times / length)[:, newaxis]
    )), args=(to_fit.values, time_in_days),
    method="bounded", bounds=(0, 365*5)
)
# Returns just over the lower bound if I make the lower bound more
# than one.
print("Parameters for exp(-dt/T):")
print(single_time_opt.x)

two_time_opt = scipy.optimize.minimize(
    fun=lambda params, acorr, times: nansum(square(
            acorr - 
            (
                params[0] * exp(-times / (params[1] * DAYS_PER_WEEK)) +
                (1 - params[0]) * exp(-times / (params[2] / HOURS_PER_DAY))
            )[:, np.newaxis]
    )),
    x0=(.8, 3., 3.),
    args=(to_fit.values, time_in_days),
    method="L-BFGS-B"
)
# Around a twenty-day correlation period, with most weight on EC error
print("Parameters for a exp(-dt/T) + (1 - a) exp(-dt/Tec): [a, T, Tec]")
print(two_time_opt)
print("Units: unitless, weeks, hours")
# print("EC time in hours:", two_time_opt.x[2] * HOURS_PER_DAY)

cos_opt = scipy.optimize.minimize(
    fun=lambda params, acorr, times: nansum(square(
            # [a, b0, b1, b2, c, d, Td, Ta, To, Tec]
            acorr - 
            (
                params[0] * cos(TWO_PI_OVER_DAY * times) * exp(-times / (params[6] * DAYS_PER_WEEK)) +
                (
                    params[1] / 10 +
                    params[2] / 10 * cos(TWO_PI_OVER_YEAR * times) +
                    params[3] / 10 * cos(2 * TWO_PI_OVER_YEAR * times)
                ) * exp(-times / (params[7] * DAYS_PER_YEAR)) +
                params[4] * exp(-times / (params[8] * DAYS_PER_WEEK)) +
                params[5] * exp(-times / (params[9] / HOURS_PER_DAY))
            )#[:, np.newaxis]
    )),
    x0=(.2, .2, .2, .2, .2, .2, 2, 3, 5, 3.),
    args=(to_fit.mean(axis=1).values, time_in_days),
    method="L-BFGS-B",
    # options=dict(maxiter=100, disp=True),
)

print("Parameters for cosine:",
      "\nDaily coefficient and timescale (weeks):", cos_opt.x[[0, 6]],
      "\nAnnual coefficients * 10 and timescale (years):", cos_opt.x[[1, 2, 3, 7]],
      # "\nAnnual timescale in years:", cos_opt.x[7] / DAYS_PER_YEAR,
      "\nResidual coefficient and timescale (weeks):", cos_opt.x[[4, 8]],
      "\nEddy Covariance coefficient and timescale (hours):", cos_opt.x[[5, 9]],
      # "\nEC timescale in hours:", cos_opt.x[9] * HOURS_PER_DAY,
      "\nConverged:", cos_opt.success, cos_opt.message)

exp_sin2_opt = scipy.optimize.minimize(
    fun=lambda params, acorr, times: nansum(square(
            # [a, b, c, d, ld, la, Td, Ta, To, Tec]
            acorr - 
            (
                params[0] * exp(-(sin(TWO_PI_OVER_DAY * times) / params[4]) ** 2) *
                  exp(-times / (params[6] * DAYS_PER_WEEK)) +
                params[1] * exp(-(sin(TWO_PI_OVER_YEAR * times) / params[5]) ** 2) *
                  exp(-times / (params[7] * DAYS_PER_YEAR)) +
                params[2] * exp(-times / (params[8] * DAYS_PER_WEEK)) +
                params[3] * exp(-times / (params[9] / HOURS_PER_DAY))
            )[:, np.newaxis]
    )),
    x0=(.2, .2, .2, .2, 10, 10, 2, 3, 5, 3.),
    args=(to_fit.values, time_in_days),
    method="L-BFGS-B"
)

print("Parameters for exponential sin-squared:",
      "\nDaily coefficient, dieoff, and timescale:", exp_sin2_opt.x[[0, 4, 6]],
      "\nAnnual coefficient, dieoff, and timescale:", exp_sin2_opt.x[[1, 5, 7]],
      "\nAnnual timescale in years:", exp_sin2_opt.x[7] / DAYS_PER_YEAR,
      "\nResidual coefficient and timescale:", exp_sin2_opt.x[[2, 8]],
      "\nEddy Covariance coefficient and timescale:", exp_sin2_opt.x[[3, 9]],
      "\nEC timescale in hours:", exp_sin2_opt.x[9] * HOURS_PER_DAY,
      "\nConverged:", exp_sin2_opt.success, exp_sin2_opt.message)

acf_data.plot(
    subplots=True, sharex=True, sharey=True,
    xlim=(0, 1e9 * 3600 * 24 * 365.2425 * 5), ylim=(-.5, 1),
    xticks=pd.timedelta_range(start=0, freq="365D", periods=6).to_numpy().astype(float)
)
plt.savefig("ameriflux_minus_casa_half_hour_tower_data_long.pdf")

plt.xlim(0, 1e9 * 3600 * 24 * 60)
plt.xticks(pd.timedelta_range(start=0, freq="7D", periods=8).to_numpy().astype(float))
plt.savefig("ameriflux_minus_casa_half_hour_tower_data_short.pdf")


def exp_only(tdata, resid_coef, To, Tec):
    Tec /= HOURS_PER_DAY
    To *= DAYS_PER_WEEK
    result = resid_coef * np.exp(-tdata / To)
    result += (1 - resid_coef) * np.exp(-tdata / Tec)
    return result


def exp_cos_daily(tdata, daily_coef, Td, resid_coef, To, ec_coef, Tec):
    Tec /= HOURS_PER_DAY
    To *= DAYS_PER_WEEK
    Td *= DAYS_PER_WEEK
    result = daily_coef * np.cos(TWO_PI_OVER_DAY * tdata) * np.exp(-tdata / Td)
    result += resid_coef * np.exp(-tdata / To)
    result += ec_coef * np.exp(-tdata / Tec)
    return result


def exp_cos_daily_annual(tdata, daily_coef, Td, ann_coef0, ann_coef1, ann_coef2, Ta, resid_coef, To, ec_coef, Tec):
    Tec /= HOURS_PER_DAY
    Td *= DAYS_PER_WEEK
    Ta *= DAYS_PER_YEAR
    To *= DAYS_PER_WEEK
    result = daily_coef * np.cos(TWO_PI_OVER_DAY * tdata) * np.exp(-tdata / Td)
    result += (ann_coef0 + ann_coef1 * np.cos(TWO_PI_OVER_YEAR * tdata) +
               ann_coef2 * np.cos(FOUR_PI_OVER_YEAR * tdata)) * np.exp(-tdata / Ta)
    result += resid_coef * np.exp(-tdata / To)
    result += ec_coef * np.exp(-tdata / Tec)
    return result


def exp_cos_daily_expsin2_annual(tdata, daily_coef, Td, ann_coef, ann_width, Ta, resid_coef, To, ec_coef, Tec):
    Tec /= HOURS_PER_DAY
    Td *= DAYS_PER_WEEK
    Ta *= DAYS_PER_YEAR
    To *= DAYS_PER_WEEK
    result = daily_coef * np.cos(TWO_PI_OVER_DAY * tdata) * np.exp(-tdata / Td)
    result += ann_coef * np.exp(-(np.sin(PI_OVER_YEAR * tdata) / ann_width) ** 2 - tdata / Ta)
    result += resid_coef * np.exp(-tdata / To)
    result += ec_coef * np.exp(-tdata / Tec)
    return result


def exp_cos_daily_times_annual(tdata, ann_coef0, ann_coef1, ann_coef2, Ta, resid_coef, To, ec_coef, Tec):
    Tec /= HOURS_PER_DAY
    Ta *= DAYS_PER_YEAR
    To *= DAYS_PER_WEEK
    result = (
        np.cos(TWO_PI_OVER_DAY * tdata) *
        (ann_coef0 + ann_coef1 * np.cos(TWO_PI_OVER_YEAR * tdata) +
         ann_coef2 * np.cos(FOUR_PI_OVER_YEAR * tdata)) * np.exp(-tdata / Ta)
    )
    result += resid_coef * np.exp(-tdata / To)
    result += ec_coef * np.exp(-tdata / Tec)
    return result


def exp_expsin2_daily_times_cos_annual(tdata, daily_width, ann_coef0, ann_coef1, ann_coef2, Ta, resid_coef, To, ec_coef, Tec):
    Tec /= HOURS_PER_DAY
    Ta *= DAYS_PER_YEAR
    To *= DAYS_PER_WEEK
    result = (
        np.exp(-(np.sin(PI_OVER_DAY * tdata) / daily_width) ** 2) *
        (ann_coef0 + ann_coef1 * np.cos(TWO_PI_OVER_YEAR * tdata) +
         ann_coef2 * np.cos(FOUR_PI_OVER_YEAR * tdata)) * np.exp(-tdata / Ta)
    )
    result += resid_coef * np.exp(-tdata / To)
    result += ec_coef * np.exp(-tdata / Tec)
    return result


def exp_cos_daily_times_expsin2_annual(tdata, ann_coef, ann_width, Ta, resid_coef, To, ec_coef, Tec):
    Tec /= HOURS_PER_DAY
    Ta *= DAYS_PER_YEAR
    To *= DAYS_PER_WEEK
    result = (
        np.cos(TWO_PI_OVER_DAY * tdata) *
        ann_coef *
        np.exp(-(np.sin(PI_OVER_YEAR * tdata) / ann_width) ** 2)
    )
    result += resid_coef * np.exp(-tdata / To)
    result += ec_coef * np.exp(-tdata / Tec)
    return result


CORR_FUNS = (
    exp_only,
    exp_cos_daily,
    exp_cos_daily_annual,
    exp_cos_daily_expsin2_annual,
    exp_cos_daily_times_annual,
    exp_expsin2_daily_times_cos_annual,
    exp_cos_daily_times_expsin2_annual,
)

STARTING_PARAMS = dict(
    daily_coef = 0.4,
    daily_width = .3,
    Td = 3.,
    ann_coef0 = -1e-3,
    ann_coef1 = +1e-2,
    ann_coef2 = +1e-2,
    ann_coef = 0.04,
    ann_width = .3,
    Ta = 3.,
    resid_coef = 0.2,
    To = 3.,
    ec_coef = 0.4,
    Tec = 3.,
)

COEF_DATA = pd.DataFrame(
    columns=STARTING_PARAMS.keys(),
    index=pd.MultiIndex.from_product(
        [acf_data,
         [fun.__name__ for fun in CORR_FUNS]],
        names=["Site", "Correlation function"]
    )
)
COEF_VAR_DATA = pd.DataFrame(
    columns=STARTING_PARAMS.keys(),
    index=pd.MultiIndex.from_product(
        [acf_data,
         [fun.__name__ for fun in CORR_FUNS]],
        names=["Site", "Correlation function"]
    )
)
INF_CRIT = (aic, aicc, bic, hqic)
IC_DATA = pd.DataFrame(
    columns=[ic.__name__ for ic in INF_CRIT],
    index=pd.MultiIndex.from_product(
        [acf_data,
         [fun.__name__ for fun in CORR_FUNS]],
        names=["Site", "Correlation function"]
    )
)
SAMPLE_SIZE = 2000
MVN_LOGPDF = scipy.stats.multivariate_normal.logpdf
MIN_LAGS_FOR_FIT = pd.Timedelta(50, "W")

for column in acf_data.iloc[:, :]:
    print(column, flush=True)
    acf_col = acf_data[column].dropna()
    if acf_col.index[-1] < MIN_LAGS_FOR_FIT:
        continue
    acf_times = acf_col.index.values.astype("m8[h]").astype("u8")
    acf_times -= acf_times[0]
    acf_times = acf_times.astype("f4") / 24
    fig, axes = plt.subplots(len(CORR_FUNS) + 1, 1, figsize=(6.5, 5),
                             sharex=True, sharey=True)
    tmp = axes[0].plot(acf_times, acf_col)
    tmp = axes[0].set_title("Empirical Correlogram")
    tmp = fig.suptitle(
        "{site:s} - {climate:s} - {veg:s}".format(
            site=column,
            climate=amf_ds.coords["CLIMATE_KOEPPEN"].sel(site=column).values,
            veg=amf_ds.coords["IGBP"].sel(site=column).values,
        )
    )
    col_data = difference_df_rect.loc[:, column].dropna()
    sample = col_data.sample(min(SAMPLE_SIZE, col_data.shape[0]))
    sample /= sample.std()
    sample_times = acf_col.index.values.astype("M8[h]").astype("u8")
    sample_times -= sample_times[0]
    sample_times = sample_times.astype("f4") / 24
    dt_sample = np.abs(sample_times[:, np.newaxis] - sample_times[np.newaxis, :])
    for corr_fun, ax in zip(CORR_FUNS, axes[1:]):
        argspec = inspect.getfullargspec(corr_fun)
        param_names = inspect.getfullargspec(corr_fun).args[1:]
        try:
            param_vals, param_cov = scipy.optimize.curve_fit(
                corr_fun, acf_times, acf_col,
                [STARTING_PARAMS[param] for param in param_names]
            )
        except RuntimeError:
            continue
        COEF_DATA.loc[(column, corr_fun.__name__), param_names] = param_vals
        COEF_VAR_DATA.loc[(column, corr_fun.__name__), param_names] = np.diag(param_cov)
        tmp = ax.plot(acf_times, corr_fun(acf_times, *param_vals), label="ACF fit")
        # def loglik_fn(params):
        #     corr_mat = corr_fun(dt_sample, *params)
        #     return MVN_LOGPDF(sample.values, cov=corr_mat)
        # # res = scipy.optimize.minimize(lambda params: -loglik_fn(params), param_vals)
        # # ax.plot(acf_times, corr_fun(acf_times, *res.x), label="ML fit")
        # # ax.legend()
        # loglik = loglik_fn(param_vals)
        # for ic_fun in INF_CRIT:
        #     IC_DATA.loc[(column, corr_fun.__name__), ic_fun.__name__] = ic_fun(loglik, 1, len(param_names))
        # ic_str = "AIC: {aic:3.2e} AICC: {aicc:3.2e} BIC: {3.2e} HQIC: {3.2e}".format(
        #     IC_DATA.loc[(column, corr_fun.__name__), :]
        # )
        # ax.set_title(
        #     "Fit of {corr_fun:s}  (ic_str:s}".format(
        #         corr_fun=corr_fun.__name__, ic_str=ic_str
        #     )
        # )
        tmp = ax.set_title("Fit of {corr_fun:s}".format(corr_fun=corr_fun.__name__))
    tmp = fig.tight_layout()
    tmp = fig.subplots_adjust(top=.9, hspace=1.1)
    tmp = ax.set_xlim(0, 365.2425 * 3)
    tmp = ax.set_ylim(-.5, 1)
    for ax in axes:
        tmp = ax.set_xticks(pd.timedelta_range(start=0, freq="365D", periods=6).to_numpy()
                            .astype("m8[D]").astype("u8").astype(float))
    tmp = fig.savefig("{site:s}-ameriflux-minus-casa-corr-fits-long.pdf".format(site=column))
    tmp = ax.set_xlim(0, 56)
    for ax in axes:
        tmp = ax.set_xticks(pd.timedelta_range(start=0, freq="7D", periods=9).to_numpy()
                            .astype("m8[D]").astype("u8").astype(float))
    tmp = fig.savefig("{site:s}-ameriflux-minus-casa-corr-fits-short.pdf".format(site=column))
    tmp = plt.close(fig)

COEF_DATA.to_csv("ameriflux-minus-casa-half-hour-towers-parameters.csv")
COEF_VAR_DATA.to_csv("ameriflux-minus-casa-half-hour-towers-parameter-variances.csv")
