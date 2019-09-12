#!/usr/bin/env python
# ~*~ coding: utf8 ~*~
"""Read in and plot the AmeriFlux data.
"""
import argparse
import os.path
import glob
import re

import numpy as np
import cycler
import matplotlib as mpl
mpl.interactive(True)
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray

MINUTES_PER_HOUR = 60
SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24
SECONDS_PER_DAY = SECONDS_PER_HOUR * HOURS_PER_DAY
MINUTES_PER_DAY = MINUTES_PER_HOUR * HOURS_PER_DAY
MONTHS_PER_YEAR = 12

PARSER = argparse.ArgumentParser(
    description=__doc__,
)

PARSER.add_argument(
    "ameriflux_root",
    help="Directory containing site directories with data.",
)


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
    year_match = re.search(r"\d{4}_", ameriflux_file)
    year = ameriflux_file[year_match.start():year_match.end() - 1]
    year_start = np.datetime64("{year:s}-01-01T00:00".format(year=year))
    ds = pd.read_csv(
        ameriflux_file, index_col=["time"],
        parse_dates=dict(time=["DoY"]),
        date_parser=lambda doy: (
            year_start + np.array(
                np.round(float(doy) * MINUTES_PER_DAY),
                dtype="m8[m]"
            )
        ),
        na_values=[
            "-{nines:s}{dot:s}".format(nines="9" * n_nines, dot=dot)
            for n_nines in (3, 4, 5, 6)
            for dot in (".", "")
        ]
    )
    nee_ds = ds[[col for col in ds.columns if "NEE" in col]]
    nee_ds.columns = pd.MultiIndex.from_product([[site_name], nee_ds.columns])
    return nee_ds


if __name__ == "__main__":
    ARGS = PARSER.parse_args()
    TOWER_DATA = [
        pd.concat(
            [parse_file(name)
             for name in glob.glob(
                os.path.join(ARGS.ameriflux_root, site_dir,
                             "*_h.txt"))],
            axis=0
        )
        for site_dir in os.listdir(ARGS.ameriflux_root)
        if os.path.isdir(os.path.join(ARGS.ameriflux_root, site_dir)) and
        glob.glob(os.path.join(ARGS.ameriflux_root, site_dir, "*_h.txt"))
    ]

    TOWER_DF = pd.concat(TOWER_DATA, axis=1)  # .loc["2005-01-01":, :]
    HOURLY_DATA = TOWER_DF.resample("1H").mean()

    HOURLY_DATA["month"] = HOURLY_DATA.index.month
    HOURLY_DATA["hour"] = HOURLY_DATA.index.hour

    DAILY_CYCLE_BY_MONTH_GROUPS = HOURLY_DATA.groupby(["month", "hour"])
    DAILY_CYCLE_BY_MONTH = DAILY_CYCLE_BY_MONTH_GROUPS.mean()
    DAILY_CYCLE_BY_MONTH_STD = DAILY_CYCLE_BY_MONTH_GROUPS.std()

    TOWER_NAMES = DAILY_CYCLE_BY_MONTH.columns.get_level_values(0).unique()
    NEE_VAR_NAMES = DAILY_CYCLE_BY_MONTH.columns.get_level_values(1).unique()
    assert len(TOWER_NAMES) == 7 * 11
    month_colors = sns.husl_palette(MONTHS_PER_YEAR)

    XR_DAILY_CYCLE_BY_MONTH = xarray.Dataset.from_dataframe(
        DAILY_CYCLE_BY_MONTH.stack(0)
    ).isel(level_2=slice(0, -2)).rename(level_2="site")
    XR_MISSING_DAILY_CYCLE = XR_DAILY_CYCLE_BY_MONTH.isnull().all(("month", "hour"))

    DEFAULT_PROP_CYCLE = mpl.rcParams["axes.prop_cycle"]
    mpl.rcParams["axes.prop_cycle"] = cycler.cycler(color=month_colors)

    for var_name in NEE_VAR_NAMES:
        # fig, axes = plt.subplots(7, 11, sharex=True, sharey=True)
        # for ax, tower_name in zip(axes.flat, TOWER_NAMES):
        #     data = DAILY_CYCLE_BY_MONTH.loc[:, (tower_name, var_name)].unstack(0)
        #     data.plot(ax=ax, subplots=False, colors=month_colors)
        #     ax.set_title(tower_name)
        grid = XR_DAILY_CYCLE_BY_MONTH[var_name].sel(
            site=~XR_MISSING_DAILY_CYCLE[var_name]
        ).plot.line(
            x="hour", hue="month", col="site", col_wrap=11
        )
        plt.pause(.1)
        grid.fig.savefig(
            "{var_name:s}_daily_cycles_by_season.png".format(
                var_name=var_name)
        )
