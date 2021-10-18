#!/usr/bin/env python
# ~*~ coding: utf8 ~*~
"""Make all figures for ACT presentation."""

from __future__ import division, print_function

import calendar
import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeat
import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray
from pandas.plotting import register_matplotlib_converters

import correlation_function_fits
import correlation_utils
import flux_correlation_function_fits

register_matplotlib_converters()

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
# Set plotting defaults
sns.set_context("paper")
sns.set(style="ticks")
sns.set_palette("colorblind")

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

seasonal_fig, seasonal_axes = plt.subplots(
    len(REPRESENTATIVE_DATA_SITES),
    1,
    sharex=True,
    sharey=True,
)

############################################################
# Produce the plots
for cat_i, (category, site_list) in enumerate(REPRESENTATIVE_DATA_SITES.items()):
    print(category)
    all_site_data = MATCHED_DATA_DS.sel(site=site_list).load().dropna("site", how="all")
    ############################################################
    # Make a time series
    fig, axes = plt.subplots(3, 1, sharex=True)
    for ax, site_name in zip(axes, site_list):
        site_data = all_site_data.sel(site=site_name)
        ax.set_title(site_name)
        (ameriflux_line,) = ax.plot(
            site_data.coords["time"], site_data["ameriflux_fluxes"], label="AmeriFlux"
        )
        (casa_line,) = ax.plot(
            site_data.coords["time"], site_data["casa_fluxes"], label="CASA"
        )
    fig.legend([ameriflux_line, casa_line], ["AmeriFlux", "CASA"], ncol=2)
    fig.savefig("{category:s}-flux-time-series.pdf".format(category=category))
    plt.close(fig)
    ############################################################
    # Plot seasonal cycle for the sites
    fig, axes = plt.subplots(3, 1, sharex=True)
    month_numbers = np.arange(12) + 1
    for site_i, (ax, site_name) in enumerate(zip(axes, site_list)):
        site_data = MATCHED_DATA_MONTH_DS.sel(site=site_name).load()
        ax.set_title(site_name)
        (ameriflux_line,) = ax.plot(
            month_numbers, site_data["ameriflux_fluxes"], label="AmeriFlux"
        )
        (casa_line,) = ax.plot(month_numbers, site_data["casa_fluxes"], label="CASA")
        if site_i == 0:
            seasonal_axes[cat_i].set_title(site_name)
            (ameriflux_line,) = seasonal_axes[cat_i].plot(
                month_numbers, site_data["ameriflux_fluxes"], label="AmeriFlux"
            )
            (casa_line,) = seasonal_axes[cat_i].plot(
                month_numbers, site_data["casa_fluxes"], label="CASA"
            )
            seasonal_axes[cat_i].set_ylabel(
                "NEE (\N{MICRO SIGN}mol/m\N{SUPERSCRIPT TWO}/s)"
            )
            seasonal_fig.legend(
                [ameriflux_line, casa_line], ["AmeriFlux", "CASA"], ncol=2
            )
    for ax in axes:
        ax.set_ylabel("NEE (\N{MICRO SIGN}mol/m\N{SUPERSCRIPT TWO}/s)")
    axes[2].set_xlabel("Month")
    axes[2].set_xlim(1, 12)
    fig.legend([ameriflux_line, casa_line], ["AmeriFlux", "CASA"], ncol=2)
    fig.tight_layout()
    fig.savefig("{category:s}-flux-monthly-climatology.pdf".format(category=category))
    plt.close(fig)
    ############################################################
    # Plot daily cycle as a function of month
    hour_numbers = np.arange(HOURS_PER_DAY)
    for site_name in site_list:
        print(site_name)
        fig, axes = plt.subplots(3, 4, sharex=True, sharey=True)
        axes_order = axes.reshape(-1, order="F")
        site_month_hour_data = MATCHED_DATA_MONTH_HOUR_DS.sel(site=site_name).load()
        for month_num in month_numbers:
            ax = axes_order[month_num % MONTHS_PER_YEAR]
            ax.set_title(calendar.month_name[month_num])
            (ameriflux_line,) = ax.plot(
                hour_numbers,
                site_month_hour_data["ameriflux_fluxes"].isel(month=month_num - 1),
                label="AmeriFlux",
            )
            (casa_line,) = ax.plot(
                hour_numbers,
                site_month_hour_data["casa_fluxes"].isel(month=month_num - 1),
                label="CASA",
            )
            ax.set_xlim(0, HOURS_PER_DAY - 1)
            ax.set_ylim(-40, 10)
            ax.set_xticks([0, 12, 24])
            ax.set_xticks([6, 18], minor=True)
        for ax in axes[:, 0]:
            ax.set_ylabel("NEE (\N{MICRO SIGN}mol/m\N{SUPERSCRIPT TWO}/s)")
        for ax in axes[-1, :]:
            ax.set_xlabel("Hour")
        fig.legend([ameriflux_line, casa_line], ["AmeriFlux", "CASA"], ncol=2)
        fig.suptitle(site_name)
        fig.tight_layout()
        fig.subplots_adjust(top=0.87)
        fig.savefig(
            "{category:s}-{site:s}-daily-cycle-climatology.pdf".format(
                category=category, site=site_name
            )
        )
        plt.close(fig)

seasonal_fig.savefig("combined-categories-flux-monthly-climatology.pdf")
plt.close(seasonal_fig)

multi_corr_fig, multi_corr_axes = plt.subplots(
    len(
        [
            site_list[0]
            for site_list in REPRESENTATIVE_DATA_SITES.values()
            # for site in site_list
        ]
    ),
    1,
    sharex=True,
    sharey=True,
    figsize=(6.5, 4.5),
)
multi_corr_ax_i = 0

multi_spectrum_day_fig, multi_spectrum_day_axes = plt.subplots(
    len(
        [
            site_list[0]
            for site_list in REPRESENTATIVE_DATA_SITES.values()
            # for site in site_list
        ]
    ),
    1,
    sharex=True,
    sharey=True,
    figsize=(6.5, 4.5),
)
multi_spectrum_ax_i = 0

multi_spectrum_year_fig, multi_spectrum_year_axes = plt.subplots(
    len(
        [
            site_list[0]
            for site_list in REPRESENTATIVE_DATA_SITES.values()
            # for site in site_list
        ]
    ),
    1,
    sharex=True,
    sharey=True,
    figsize=(6.5, 4.5),
)
multi_spectrum_ax_i = 0


############################################################
# Plot time series, smoothed time series, and autocovariance
# Long and short versions of this plot to show different scales
for category, site_list in REPRESENTATIVE_DATA_SITES.items():
    print(category)
    all_site_data = MATCHED_DATA_DS.sel(site=site_list).load()
    # Dropping all-na data shouldn't be relevant here.
    # Change site_list if it is
    # .dropna(
    #     "site", how="all"
    # )
    for site_name in site_list:
        print(site_name)
        site_data = all_site_data.sel(site=site_name).load().dropna("time", how="any")
        print(datetime.datetime.now(), "Starting resample")
        site_df = (
            site_data.to_dataframe().loc[:, site_data.data_vars].resample("1H").mean()
        )
        print(datetime.datetime.now(), "Finding autocorrelation")
        correlation_data = correlation_utils.get_autocorrelation_stats(
            site_df["flux_difference"]
        )
        fig, axes = plt.subplots(3, 1, figsize=(6.5, 5))
        print(datetime.datetime.now(), "Starting AmeriFlux line")
        site_df["ameriflux_fluxes"].plot.line(ax=axes[0])
        print(datetime.datetime.now(), "Starting CASA line")
        site_df["casa_fluxes"].plot.line(ax=axes[0])
        print(datetime.datetime.now(), "Starting smoothed lines")
        site_df["ameriflux_fluxes"].resample("2W").mean().plot.line(ax=axes[1])
        site_df["casa_fluxes"].resample("2W").mean().plot.line(ax=axes[1])
        print(datetime.datetime.now(), "Starting Autocovariance")
        correlation_data["acovf"].plot.line(ax=axes[2])
        print(datetime.datetime.now(), "Done plots, starting ticks and labels")
        time_bounds = site_df.index.astype("M8[ns]")[[0, -1]]
        axes[0].set_xlim(time_bounds)
        axes[0].set_ylabel("NEE (\N{MICRO SIGN}mol/m\N{SUPERSCRIPT TWO}/s)")
        axes[1].set_xlim(time_bounds)
        axes[1].set_ylabel(
            "Two-week average\nNEE\n" "(\N{MICRO SIGN}mol/m\N{SUPERSCRIPT TWO}/s)"
        )
        axes[1].set_xlabel("Year")
        axes[0].set_xlabel("")
        xtick_index = pd.timedelta_range(
            start=correlation_data.index[0],
            end=correlation_data.index[-1],
            freq="365D",
        )
        axes[2].set_xticks(xtick_index.astype("i8").astype("f4"))
        axes[2].set_xticklabels(np.arange(0, len(xtick_index)))
        axes[2].set_xlabel("Time difference (years)")
        axes[2].set_ylabel(
            "Empirical\nAutocovariance\n"
            "(\N{MICRO SIGN}mol\N{SUPERSCRIPT TWO}/"
            "m\N{SUPERSCRIPT FOUR}/s\N{SUPERSCRIPT TWO})"
        )
        fig.subplots_adjust(left=0.2, hspace=0.3, top=0.9)
        fig.suptitle(site_name)
        print(datetime.datetime.now(), "Writing pdf")
        fig.savefig(
            "{category:s}-{site:s}-time-series-correlations.pdf".format(
                category=category, site=site_name
            )
        )
        print(datetime.datetime.now(), "Wrote pdf, writing png")
        fig.savefig(
            "{category:s}-{site:s}-time-series-correlations.png".format(
                category=category, site=site_name
            )
        )
        print(datetime.datetime.now(), "Wrote png")
        for ax in axes[:2]:
            ax.set_xlim("2007-06-01", "2007-07-31")
        xtick_index = pd.timedelta_range(
            start=correlation_data.index[0],
            end=correlation_data.index[60 * HOURS_PER_DAY],
            freq="7D",
        )
        axes[-1].set_xlim(xtick_index[[0, -1]].astype("i8").astype("f4"))
        axes[-1].set_xticks(xtick_index.astype("i8").astype("f4"))
        axes[-1].set_xticklabels(np.arange(len(xtick_index)))
        axes[-1].set_xlabel("Time difference (weeks)")
        fig.subplots_adjust(top=0.95, hspace=0.8)
        print(datetime.datetime.now(), "Writing short pdf")
        fig.savefig(
            "{category:s}-{site:s}-time-series-correlations-short.pdf".format(
                category=category, site=site_name
            )
        )
        print(datetime.datetime.now(), "Wrote short pdf, writing short png")
        fig.savefig(
            "{category:s}-{site:s}-time-series-correlations-short.png".format(
                category=category, site=site_name
            )
        )
        print(datetime.datetime.now(), "Wrote short png")
        plt.close(fig)
        ax = multi_corr_axes[multi_corr_ax_i]
        correlation_data["acf"].rename("Empirical").plot.line(ax=ax)
        ax.set_title(site_name)
        ax.set_ylim(-1, 1)
        ax.set_xlim(
            correlation_data.index[[0, int(np.ceil(10 * HOURS_PER_YEAR))]]
            .astype("i8")
            .astype("f4"),
        )
        multi_corr_ax_i += 1
        ax = multi_spectrum_day_axes[multi_spectrum_ax_i]
        ax.plot(
            np.fft.rfftfreq(len(correlation_data), 1.0 / HOURS_PER_DAY),
            abs(np.fft.ihfft(correlation_data["acf"])) ** 2,
        )
        ax.set_ylabel("Spectrum\n(unitless)")
        ax.set_title(site_name)
        ax.set_xlim(0, 6)
        ax.set_xlabel("Frequency (1/day)")
        ax.set_yscale("log")
        ax = multi_spectrum_year_axes[multi_spectrum_ax_i]
        ax.plot(
            np.fft.rfftfreq(len(correlation_data), 1.0 / HOURS_PER_YEAR),
            abs(np.fft.ihfft(correlation_data["acf"])) ** 2,
        )
        ax.set_ylabel("Spectrum\n(unitless)")
        ax.set_title(site_name)
        ax.set_xlim(0, 12)
        ax.set_xlabel("Frequency (1/year)")
        ax.set_yscale("log")
        multi_spectrum_ax_i += 1
        break

xtick_index = pd.timedelta_range(
    start=correlation_data.index[0],
    end=correlation_data.index[int(np.ceil(10 * HOURS_PER_YEAR))],
    freq="365D",
)
for ax in multi_corr_axes.flat:
    ax.set_xticks(xtick_index.astype("i8").astype("f4"), minor=True)
    ax.set_xticks(xtick_index.astype("i8").astype("f4"))
    ax.set_ylabel("Empirical Autocorrelation\n" "(unitless)")

ax.set_xticklabels(np.arange(0, len(xtick_index)))
ax.set_xlabel("Time difference (years)")
multi_corr_fig.tight_layout()
multi_corr_fig.savefig("shared-axis-acf-plots-long.pdf")
multi_corr_fig.savefig("shared-axis-acf-plots-long.png", dpi=300)

# read the coefficients from the fit summary file
COEF_DATA = xarray.open_dataset(
    "multi-tower-cross-validation-error-data-1050-splits.nc4"
)
MEAN_COEFFICIENTS = (
    COEF_DATA.data_vars["optimized_parameters"]
    .set_index(correlation_function="correlation_function_short_name")
    .sel(correlation_function=["dc_dmc_ap", "dp_dmc_a0"])
    .mean("splits")
)
# Correlation functions expect time deltas in units of days
CORRELATION_TIMES = (
    correlation_data.index.values.astype("m8[h]").astype("i8").astype("f4") / 24
)
CORRELATIONS = {
    "Best overall function": flux_correlation_function_fits.dc_dmc_ap_curve_ne(
        CORRELATION_TIMES,
        **MEAN_COEFFICIENTS.sel(correlation_function="dc_dmc_ap")
        .to_series()
        .dropna()
        .to_dict(),
    ),
    "Best function w/o ann. cycle": flux_correlation_function_fits.dp_dmc_a0_curve_ne(
        CORRELATION_TIMES,
        **MEAN_COEFFICIENTS.sel(correlation_function="dp_dmc_a0")
        .to_series()
        .dropna()
        .to_dict(),
    ),
}

with mpl.rc_context(
    {
        "axes.prop_cycle": mpl.rcParams["axes.prop_cycle"][:4]
        + cycler.cycler("linestyle", ["-", "--", ":", "-."])
    }
):
    # Plot the modeled correlations
    MODELED_CORRELATION_LINES = []
    for ax in multi_corr_axes.flat:
        for label, modeled_corr in CORRELATIONS.items():
            MODELED_CORRELATION_LINES.extend(
                ax.plot(
                    correlation_data.index.values, modeled_corr, alpha=0.6, label=label
                )
            )

modeled_acf_legend = multi_corr_axes.flat[0].legend(ncol=2)

# Save the new plots
multi_corr_fig.savefig("shared-axis-modeled-acf-plots-long.pdf")
multi_corr_fig.savefig("shared-axis-modeled-acf-plots-long.png", dpi=300)

multi_spectrum_day_axes[0].set_xlabel("")
multi_spectrum_day_fig.tight_layout()
multi_spectrum_day_fig.savefig("shared-axis-spectrum-plots-day-full.pdf")
multi_spectrum_day_fig.savefig("shared-axis-spectrum-plots-day-full.png", dpi=300)

for ax in multi_spectrum_day_axes:
    ax.set_xlim(1 - 5.0 / DAYS_PER_YEAR, 1 + 5.0 / DAYS_PER_YEAR)

multi_spectrum_day_fig.savefig("shared-axis-spectrum-plots-day-zoom.pdf")
multi_spectrum_day_fig.savefig("shared-axis-spectrum-plots-day-zoom.png", dpi=300)

multi_spectrum_year_axes[0].set_xlabel("")
multi_spectrum_year_fig.tight_layout()
multi_spectrum_year_fig.savefig("shared-axis-spectrum-plots-year-full.pdf")
multi_spectrum_year_fig.savefig("shared-axis-spectrum-plots-year-full.png", dpi=300)

for ax in multi_spectrum_year_axes:
    ax.set_xlim(1 - 0.5, 1 + 0.5)

multi_spectrum_year_fig.savefig("shared-axis-spectrum-plots-year-zoom.pdf")
multi_spectrum_year_fig.savefig("shared-axis-spectrum-plots-year-zoom.png", dpi=300)

xtick_index = pd.timedelta_range(
    start=correlation_data.index[0],
    end=correlation_data.index[40 * HOURS_PER_DAY],
    freq="7D",
)
xtick_index_minor = pd.timedelta_range(
    start=correlation_data.index[0],
    end=correlation_data.index[40 * HOURS_PER_DAY],
    freq="1D",
)
for ax in multi_corr_axes.flat:
    ax.set_xlim(xtick_index_minor[[0, -1]].astype("i8").astype("f4"))
    ax.set_xticks(xtick_index_minor.astype("i8").astype("f4"), minor=True)
    ax.set_xticks(xtick_index.astype("i8").astype("f4"))

ax.set_xticklabels(np.arange(0, len(xtick_index)))
ax.set_xlabel("Time difference (weeks)")

# Save version with modeled ACFs
multi_corr_fig.savefig("shared-axis-modeled-acf-plots-short.pdf")
multi_corr_fig.savefig("shared-axis-modeled-acf-plots-short.png", dpi=300)

# Remove the modeled correlations
for line in MODELED_CORRELATION_LINES:
    line.set_visible(False)

modeled_acf_legend.set_visible(False)

# And plot just the empirical autocorrelations again
multi_corr_fig.savefig("shared-axis-acf-plots-short.pdf")
multi_corr_fig.savefig("shared-axis-acf-plots-short.png", dpi=300)

print("Done climatology plots")
LONG_DATA_SITES = []

############################################################
# Find and plot towers with lots of data
print("Finding sites with long data")
for site_name in MATCHED_DATA_DS.indexes["site"]:
    site_data = MATCHED_DATA_DS.sel(site=site_name).load().dropna("time", how="all")
    data_time_period = site_data.indexes["time"][-1] - site_data.indexes["time"][0]
    n_data_points = site_data.dims["time"]
    if data_time_period < pd.Timedelta(days=MIN_YEARS_DATA * DAYS_PER_YEAR):
        # Short data record
        continue
    if n_data_points < MIN_DATA_FRAC * MIN_YEARS_DATA * HOURS_PER_YEAR:
        # Lots of missing data
        continue
    LONG_DATA_SITES.append(site_name)

LONG_DATA_LONGITUDES = MATCHED_DATA_DS.coords["Longitude"].sel(site=LONG_DATA_SITES)
LONG_DATA_LATITUDES = MATCHED_DATA_DS.coords["Latitude"].sel(site=LONG_DATA_SITES)

print(len(LONG_DATA_SITES), "towers with long data")
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})
ax.plot(LONG_DATA_LONGITUDES, LONG_DATA_LATITUDES, ".")
ax.coastlines()
ax.add_feature(cfeat.STATES, edgecolor="gray")
fig.savefig(
    "ameriflux-towers-at-least-{n_years:d}-years-data.pdf".format(
        n_years=MIN_YEARS_DATA
    )
)

for site_name in LONG_DATA_SITES:
    pass
