#!/usr/bin/env python
# ~*~ coding: utf8 ~*~
"""Make all figures for ACT presentation."""

from __future__ import print_function, division
import calendar

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeat

import correlation_utils
import flux_correlation_function_fits
import correlation_function_fits

############################################################
# Define constants for script
DAYS_PER_YEAR = 365.2425
HOURS_PER_DAY = 24
HOURS_PER_YEAR = HOURS_PER_DAY * DAYS_PER_YEAR
MONTHS_PER_YEAR = 12

MIN_YEARS_DATA = 4
MIN_DATA_FRAC = 0.75

############################################################
# Define representative sites
REPRESENTATIVE_DATA_SITES = {
    # "seasonal-positive": ["US-Bkg", "US-Ha2", "US-Blk"],
    "seasonal-both": ["US-PFa", "US-Syv", "US-Los"],
    "phase-shift": ["US-Ne1", "US-Ne3", "US-NC2", "US-Ne2"],
    # "small": ["US-Ha1", "US-MMS", "US-Sta", "US-Dk3"],
}

############################################################
# Read in data
MATCHED_DATA_DS = xarray.open_dataset(
    "ameriflux-and-casa-matching-data-2.nc4",
    # There will be decent chunks of this where I need all times but
    # only one site.
    chunks={"site": 1},
).load()
MATCHED_DATA_MONTH_HOUR_DS = xarray.open_dataset(
    "ameriflux-and-casa-all-towers-daily-cycle-by-month.nc4"
).load()
MATCHED_DATA_MONTH_DS = xarray.open_dataset(
    "ameriflux-and-casa-all-towers-seasonal-cycle.nc4"
).load()

############################################################
# Produce the plots
for category, site_list in REPRESENTATIVE_DATA_SITES.items():
    print(category)
    all_site_data = MATCHED_DATA_DS.sel(
        site=site_list
    ).load().dropna(
        "site", how="all"
    )
    ############################################################
    # Make a time series
    fig, axes = plt.subplots(3, 1, sharex=True)
    for ax, site_name in zip(axes, site_list):
        site_data = all_site_data.sel(site=site_name)
        ax.set_title(site_name)
        ameriflux_line, = ax.plot(
            site_data.coords["time"],
            site_data["ameriflux_fluxes"],
            label="AmeriFlux"
        )
        casa_line, = ax.plot(
            site_data.coords["time"],
            site_data["casa_fluxes"],
            label="CASA"
        )
    fig.legend(
        [ameriflux_line, casa_line], ["AmeriFlux", "CASA"], ncol=2
    )
    fig.savefig("{category:s}-flux-time-series.pdf".format(category=category))
    plt.close(fig)
    ############################################################
    # Plot seasonal cycle for the sites
    fig, axes = plt.subplots(3, 1, sharex=True)
    month_numbers = np.arange(12) + 1
    for ax, site_name in zip(axes, site_list):
        site_data = MATCHED_DATA_MONTH_DS.sel(
            site=site_name
        ).load()
        ax.set_title(site_name)
        ameriflux_line, = ax.plot(
            month_numbers, site_data["ameriflux_fluxes"], label="AmeriFlux"
        )
        casa_line, = ax.plot(
            month_numbers, site_data["casa_fluxes"], label="CASA"
        )
    for ax in axes:
        ax.set_ylabel("NEE (\N{MICRO SIGN}mol/m\N{SUPERSCRIPT TWO}/s)")
    axes[2].set_xlabel("Month")
    axes[2].set_xlim(1, 12)
    fig.legend(
        [ameriflux_line, casa_line], ["AmeriFlux", "CASA"], ncol=2
    )
    fig.tight_layout()
    fig.savefig(
        "{category:s}-flux-monthly-climatology.pdf".format(category=category)
    )
    plt.close(fig)
    ############################################################
    # Plot daily cycle as a function of month
    hour_numbers = np.arange(HOURS_PER_DAY)
    for site_name in site_list:
        print(site_name)
        fig, axes = plt.subplots(3, 4, sharex=True, sharey=True)
        axes_order = axes.reshape(-1, order="F")
        site_month_hour_data = MATCHED_DATA_MONTH_HOUR_DS.sel(
            site=site_name
        ).load()
        for month_num in month_numbers:
            ax = axes_order[month_num % MONTHS_PER_YEAR]
            ax.set_title(calendar.month_name[month_num])
            ameriflux_line, = ax.plot(
                hour_numbers,
                site_month_hour_data["ameriflux_fluxes"].isel(month=month_num - 1),
                label="AmeriFlux"
            )
            casa_line, = ax.plot(
                hour_numbers,
                site_month_hour_data["casa_fluxes"].isel(month=month_num - 1),
                label="CASA"
            )
        for ax in axes[:, 0]:
            ax.set_ylabel("NEE (\N{MICRO SIGN}mol/m\N{SUPERSCRIPT TWO}/s)")
        for ax in axes[-1, :]:
            ax.set_xlabel("Hour")
            ax.set_xlim(0, HOURS_PER_DAY - 1)
            ax.set_ylim(-40, 10)
        fig.legend(
            [ameriflux_line, casa_line], ["AmeriFlux", "CASA"], ncol=2
        )
        fig.suptitle(site_name)
        fig.tight_layout()
        fig.subplots_adjust(top=.9)
        fig.savefig(
            "{category:s}-{site:s}-daily-cycle-climatology.pdf".format(
                category=category, site=site_name
            )
        )
        plt.close(fig)

############################################################
# Plot time series, smoothed time series, and autocovariance
# Long and short versions of this plot to show different scales
for category, site_list in REPRESENTATIVE_DATA_SITES.items():
    print(category)
    all_site_data = MATCHED_DATA_DS.sel(
        site=site_list
    ).load().dropna(
        "site", how="all"
    )
    for site_name in site_list:
        print(site_name)
        site_data = all_site_data.sel(
            site=site_name
        ).load().dropna(
            "time", how="any"
        )
        site_df = site_data.to_dataframe(
        ).loc[:, site_data.data_vars].resample("1H").mean()
        correlation_data = correlation_utils.get_autocorrelation_stats(
            site_df["flux_difference"]
        )
        fig, axes = plt.subplots(3, 1)
        site_df["ameriflux_fluxes"].plot.line(
            ax=axes[0]
        )
        site_df["casa_fluxes"].plot.line(
            ax=axes[0]
        )
        site_df["ameriflux_fluxes"].resample("2W").mean().plot.line(
            ax=axes[1]
        )
        site_df["casa_fluxes"].resample("2W").mean().plot.line(
            ax=axes[1]
        )
        correlation_data["acovf"].plot.line(
            ax=axes[2]
        )
        time_bounds = site_df.index.astype("M8[ns]")[[0, -1]]
        axes[0].set_xlim(time_bounds)
        axes[0].set_ylabel("NEE (\N{MICRO SIGN}mol/m\N{SUPERSCRIPT TWO}/s)")
        axes[1].set_xlim(time_bounds)
        axes[1].set_ylabel("Two-week average\nNEE\n"
                           "(\N{MICRO SIGN}mol/m\N{SUPERSCRIPT TWO}/s)")
        axes[1].set_xlabel("Year")
        axes[0].set_xlabel("")
        xtick_index = pd.timedelta_range(
            start=correlation_data.index[0],
            end=correlation_data.index[-1],
            freq="365D",
        )
        axes[2].set_xticks(
            xtick_index.astype("i8").astype("f4")
        )
        axes[2].set_xticklabels(
            np.arange(0, len(xtick_index))
        )
        axes[2].set_xlabel("Time difference (years)")
        axes[2].set_ylabel(
            "Empirical\nAutocovariance\n"
            "(\N{MICRO SIGN}mol\N{SUPERSCRIPT TWO}/"
            "m\N{SUPERSCRIPT FOUR}/s\N{SUPERSCRIPT TWO})"
        )
        fig.subplots_adjust(left=.2)
        fig.suptitle(site_name)
        fig.savefig(
            "{category:s}-{site:s}-time-series-correlations.pdf".format(
                category=category, site=site_name
            )
        )
        for ax in axes[:2]:
            ax.set_xlim("2007-06-01", "2007-07-31")
        xtick_index = pd.timedelta_range(
            start=correlation_data.index[0],
            end=correlation_data.index[60 * HOURS_PER_DAY],
            freq="7D",
        )
        axes[-1].set_xlim(
            xtick_index[[0, -1]].astype("i8").astype("f4")
        )
        axes[-1].set_xticks(xtick_index.astype("i8").astype("f4"))
        axes[-1].set_xticklabels(np.arange(len(xtick_index)))
        axes[-1].set_xlabel("Time difference (weeks)")
        fig.subplots_adjust(top=.95, hspace=.8)
        fig.savefig(
            "{category:s}-{site:s}-time-series-correlations-short.pdf".format(
                category=category, site=site_name
            )
        )
        plt.close(fig)
        break

print("Done climatology plots")
LONG_DATA_SITES = []

############################################################
# Find and plot towers with lots of data
print("Finding sites with long data")
for site_name in MATCHED_DATA_DS.indexes["site"]:
    site_data = MATCHED_DATA_DS.sel(site=site_name).load().dropna(
        "time", how="all"
    )
    data_time_period = site_data.indexes["time"][-1] - site_data.indexes["time"][0]
    n_data_points = site_data.dims["time"]
    if data_time_period < pd.Timedelta(days=MIN_YEARS_DATA * DAYS_PER_YEAR):
        # Short data record
        continue
    if n_data_points < MIN_DATA_FRAC * MIN_YEARS_DATA * HOURS_PER_YEAR:
        # Lots of missing data
        continue
    LONG_DATA_SITES.append(site_name)

LONG_DATA_LONGITUDES = MATCHED_DATA_DS.coords["Longitude"].sel(
    site=LONG_DATA_SITES
)
LONG_DATA_LATITUDES = MATCHED_DATA_DS.coords["Latitude"].sel(
    site=LONG_DATA_SITES
)

fig, ax = plt.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})
ax.plot(LONG_DATA_LONGITUDES, LONG_DATA_LATITUDES, ".")
ax.coastlines()
ax.add_feature(cfeat.STATES, edgecolor="gray")
fig.savefig("ameriflux-towers-at-least-{n_years:d}-years-data.pdf"
            .format(n_years=MIN_YEARS_DATA))

for site_name in LONG_DATA_SITES:
    pass
