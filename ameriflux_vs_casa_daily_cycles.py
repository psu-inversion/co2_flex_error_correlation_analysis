#!/usr/bin/env python
# ~*~ coding: utf8 ~*~
"""Read in and plot the AmeriFlux data.
"""
import argparse
import calendar
import glob
import os.path
import re
import zipfile

import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import seaborn as sns
import xarray

# mpl.interactive(True)
# mpl.use("TkAgg")


MINUTES_PER_HOUR = 60
SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24
SECONDS_PER_DAY = SECONDS_PER_HOUR * HOURS_PER_DAY
MINUTES_PER_DAY = MINUTES_PER_HOUR * HOURS_PER_DAY
MONTHS_PER_YEAR = 12
OBJECT_DTYPE = np.dtype("O")
MONTH_NAMES = calendar.month_name

PARSER = argparse.ArgumentParser(
    description=__doc__,
)

PARSER.add_argument(
    "ameriflux_root",
    help="Directory containing site directories with data.",
)
PARSER.add_argument(
    "casa_path",
    help="Directory containing downscaled CASA data.",
)

CENTRAL_TIME = pytz.timezone("US/Central")


def parse_file(ameriflux_file):
    """Pull NEE-related data from AmeriFlux file into DataFrame.

    Parameters
    ----------
    ameriflux_file : str

    Returns
    -------
    pd.DataFrame
    """
    site_name = os.path.basename(os.path.dirname(ameriflux_file))
    site_id = os.path.basename(ameriflux_file)[:5]
    if "-" not in site_id:
        site_id = "{country:2s}-{site:3s}".format(country=site_id[:2], site=site_id[2:])
    year_match = re.search(r"\d{4}_", ameriflux_file)
    year = ameriflux_file[year_match.start() : year_match.end() - 1]
    year_start = np.datetime64("{year:s}-01-01T00:00-06:00".format(year=year))
    ds = pd.read_csv(
        ameriflux_file,
        index_col=["time"],
        parse_dates=dict(time=["DoY"]),
        date_parser=lambda doy: (
            year_start + np.array(np.round(float(doy) * MINUTES_PER_DAY), dtype="m8[m]")
        ),
        na_values=[
            "-{nines:s}{dot:s}".format(nines="9" * n_nines, dot=dot)
            for n_nines in (3, 4, 5, 6)
            for dot in (".", "")
        ],
    )
    nee_ds = ds[[col for col in ds.columns if "NEE" in col]]
    nee_ds.columns = pd.MultiIndex.from_product([[site_id], nee_ds.columns])
    return nee_ds


if __name__ == "__main__":
    ARGS = PARSER.parse_args()
    # TOWER_DATA = [
    #     pd.concat(
    #         [
    #             parse_file(name)
    #             for name in glob.glob(
    #                 os.path.join(ARGS.ameriflux_root, site_dir, "*_h.txt")
    #             )
    #         ],
    #         axis=0,
    #     )
    #     for site_dir in os.listdir(ARGS.ameriflux_root)
    #     if os.path.isdir(os.path.join(ARGS.ameriflux_root, site_dir))
    #     and glob.glob(os.path.join(ARGS.ameriflux_root, site_dir, "*_h.txt"))
    # ]

    # TOWER_DF = pd.concat(TOWER_DATA, axis=1)
    # HOURLY_DATA = TOWER_DF.resample("1H").mean()

    HALF_HOUR_TOWERS = (
        xarray.open_dataset(
            os.path.join(
                ARGS.ameriflux_root,
                "AmeriFlux_all_CO2_fluxes_with_single_estimate_per_tower_half_hour_data.nc4",
            )
        )
        .resample(TIMESTAMP_START="1H")
        .mean()
    )
    HOUR_TOWERS = xarray.open_dataset(
        os.path.join(
            ARGS.ameriflux_root,
            "AmeriFlux_all_CO2_fluxes_with_single_estimate_per_tower_hour_data.nc4",
        )
    )
    HOURLY_DATA = xarray.concat(
        [
            HALF_HOUR_TOWERS["ameriflux_carbon_dioxide_flux_estimate"],
            HOUR_TOWERS["ameriflux_carbon_dioxide_flux_estimate"],
        ],
        dim="site",
    ).to_series()

    CASA_DATA = xarray.open_mfdataset(
        [
            os.path.join(
                ARGS.casa_path,
                "{year:04d}-{month:02d}_downscaled_CASA_L2_Ensemble_Mean_Biogenic_NEE_Ameriflux.nc4",
            ).format(year=year, month=month)
            for year in range(2003, 2019)
            for month in range(1, 13)
        ],
        compat="override",
        combine="nested",
        concat_dim="time",
        coords="minimal",
    )
    for coord_name, coord_val in CASA_DATA.coords.items():
        if coord_val.dtype != OBJECT_DTYPE:
            continue
        coord_val = coord_val.persist()
        max_len = max(map(len, coord_val.values))
        CASA_DATA.coords[coord_name] = coord_val.astype(
            "U{max_len:d}".format(max_len=max_len)
        )

    HOURLY_DATA["month"] = HOURLY_DATA.index.month
    HOURLY_DATA["hour"] = HOURLY_DATA.index.hour

    DAILY_CYCLE_BY_MONTH_GROUPS = HOURLY_DATA.groupby(["month", "hour"])
    DAILY_CYCLE_BY_MONTH = DAILY_CYCLE_BY_MONTH_GROUPS.mean()
    DAILY_CYCLE_BY_MONTH_STD = DAILY_CYCLE_BY_MONTH_GROUPS.std()

    TOWER_NAMES = DAILY_CYCLE_BY_MONTH.columns.get_level_values(0).unique()
    NEE_VAR_NAMES = DAILY_CYCLE_BY_MONTH.columns.get_level_values(1).unique()

    XR_DAILY_CYCLE_BY_MONTH = (
        xarray.Dataset.from_dataframe(DAILY_CYCLE_BY_MONTH.stack(0))
        .isel(level_2=slice(0, -2))
        .rename(level_2="site")
    )
    XR_MISSING_DAILY_CYCLE = XR_DAILY_CYCLE_BY_MONTH.isnull().all(("month", "hour"))

    # CASA_DF = CASA_DATA["NEE"].to_series().unstack(1)

    CASA_DATA.coords["month"] = CASA_DATA.indexes["time"].month
    CASA_DATA.coords["hour"] = CASA_DATA.indexes["time"].hour
    CASA_DATA.coords["month_hour"] = (
        CASA_DATA.coords["month"] * 1000 + CASA_DATA.coords["hour"]
    )

    CASA_DAILY_CYCLE_BY_MONTH_GROUPS = CASA_DATA.groupby("month_hour")
    CASA_DAILY_CYCLE_BY_MONTH = CASA_DAILY_CYCLE_BY_MONTH_GROUPS.mean()

    CASA_DAILY_CYCLE_BY_MONTH.coords["month_hour"] = pd.MultiIndex.from_arrays(
        # Undo the operation five lines back
        divmod(CASA_DAILY_CYCLE_BY_MONTH["month_hour"].values, 1000),
        # [CASA_DAILY_CYCLE_BY_MONTH["month"].values, CASA_DAILY_CYCLE_BY_MONTH["hour"].values],
        names=["month", "hour"],
    )
    CASA_DAILY_CYCLE_BY_MONTH = CASA_DAILY_CYCLE_BY_MONTH.unstack(
        "month_hour"
    ).set_index(ameriflux_tower_location="Site_Id")

    COMBINED_TOWER_LIST = sorted(
        set(CASA_DATA.coords["Site_Id"].values) & set(TOWER_NAMES.values)
    )

    for tower in COMBINED_TOWER_LIST:
        casa_daily_cycles = CASA_DAILY_CYCLE_BY_MONTH["NEE"].sel(
            ameriflux_tower_location=tower
        )
        ameriflux_daily_cycles = XR_DAILY_CYCLE_BY_MONTH["NEE_or_fMDS"].sel(site=tower)
        if ameriflux_daily_cycles.isnull().all():
            continue
        fig, axes = plt.subplots(3, 4, sharex=True, sharey=True)
        for month in range(1, MONTHS_PER_YEAR + 1):
            ax = axes.flat[month - 1]
            casa_month = casa_daily_cycles.sel(month=month)
            casa_line = ax.plot(
                casa_month.coords["hour"].values,
                casa_month.values,
                color="r",
                label="CASA",
            )
            ameriflux_month = ameriflux_daily_cycles.sel(month=month)
            amf_line = ax.plot(
                ameriflux_month.coords["hour"].values,
                ameriflux_month.values,
                color="b",
                label="AmeriFlux",
            )
            ax.set_title(MONTH_NAMES[month])
        for ax in axes[-1, :]:
            ax.set_xlabel("Hour")
            ax.set_xticks([0, 12, 24])
            ax.set_xlim(0, 24)
        for ax in axes[:, 0]:
            ax.set_ylabel(
                "CO\N{SUBSCRIPT TWO} flux\n"
                "(\N{MICRO SIGN}mol/m\N{SUPERSCRIPT TWO}/s)"
            )
        fig.suptitle(
            "CASA and AmeriFlux NEE at {site:s}".format(
                site=casa_daily_cycles.coords["Name"].values
            )
        )
        fig.subplots_adjust(bottom=0.16, hspace=0.3)
        fig.legend(
            [casa_line[0], amf_line[0]],
            ["CASA", "AmeriFlux"],
            loc="lower center",
            ncol=2,
        )
        fig.savefig("{site:s}-nee-daily-cycle-by-month.pdf".format(site=tower))
        plt.close(fig)
