#!/usr/bin/env python
# ~*~ coding: utf8 ~*~
"""Spectral analysis of ameriflux - CASA differences."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.multivariate.pca import PCA

# This is probably what my screen is
mpl.rcParams["figure.dpi"] = 144
mpl.rcParams["savefig.dpi"] = 300

corr_data1 = pd.read_csv(
    "ameriflux-minus-casa-half-hour-towers-autocorrelation-functions.csv", index_col=0
)
corr_data2 = pd.read_csv(
    "ameriflux-minus-casa-hour-towers-autocorrelation-functions.csv", index_col=0
)
corr_data = pd.concat([corr_data1, corr_data2], axis=1)
# corr_data = corr_data2
corr_data.index = pd.TimedeltaIndex(corr_data.index)
corr_data.index.name = "Time separation"
corr_data = corr_data.astype(np.float32)

pair_counts1 = pd.read_csv(
    "ameriflux-minus-casa-half-hour-towers-pair-counts.csv", index_col=0
)
pair_counts2 = pd.read_csv(
    "ameriflux-minus-casa-hour-towers-pair-counts.csv", index_col=0
)
pair_counts = pd.concat([pair_counts1, pair_counts2], axis=1)
# pair_counts = pair_counts2
pair_counts.index = pd.TimedeltaIndex(pair_counts.index)

HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365.2425
HOURS_PER_YEAR = HOURS_PER_DAY * DAYS_PER_YEAR
N_YEARS_DATA = 5

TOWERS_LONG_DATA = [
    name
    for name in corr_data
    if corr_data[name].count() >= HOURS_PER_YEAR * N_YEARS_DATA
    and pair_counts.loc[
        slice(None, "{0:d} days".format(int(DAYS_PER_YEAR * N_YEARS_DATA + 1))), name
    ].min()
    >= HOURS_PER_YEAR * 0.5
]

uniform_corr_data = corr_data.loc[
    slice(None, "{0:d} days".format(int(DAYS_PER_YEAR * N_YEARS_DATA))),
    TOWERS_LONG_DATA,
]
uniform_corr_data.plot(
    subplots=True,
    sharey=True,
    figsize=(12.5, 6.5),
    title="AmeriFlux - CASA autocorrelations",
    xticks=pd.timedelta_range(start=0, freq="365D", periods=N_YEARS_DATA + 1)
    .to_numpy()
    .astype(float),
)
plt.subplots_adjust(bottom=0.07, top=0.95, left=0.05, right=0.95)
fig = plt.gcf()
for ax in fig.axes:
    ax.set_xticks(
        pd.timedelta_range(start=0, freq="365D", periods=N_YEARS_DATA + 1)
        .to_numpy()
        .astype(float),
        minor=False,
    )
    ax.set_xticklabels(
        ["{0:d} years".format(year) for year in range(N_YEARS_DATA + 1)],
        rotation=0,
        minor=False,
    )

fig.savefig("ameriflux-minus-casa-long-correlation-data.png")

uniform_spectrum = pd.DataFrame(
    np.fft.rfft(uniform_corr_data, axis=0),
    columns=uniform_corr_data.columns,
    index=np.fft.rfftfreq(uniform_corr_data.shape[0], 1.0 / HOURS_PER_DAY),
)
uniform_spectrum.index.name = "Freq. (1/day)"
abs(uniform_spectrum).plot(
    subplots=True,
    sharey=True,
    logy=True,
    figsize=(12.5, 6.5),
    xlim=(0, 6),
    title="AmeriFlux - CASA CO$_2$ Flux difference spectra (log scale)",
)
plt.subplots_adjust(bottom=0.07, top=0.95, left=0.05, right=0.95)
plt.savefig("ameriflux-minus-casa-long-correlation-spectrum-days.png")

uniform_spectrum_year = pd.DataFrame(
    np.fft.rfft(uniform_corr_data, axis=0),
    columns=uniform_corr_data.columns,
    index=np.fft.rfftfreq(uniform_corr_data.shape[0], 1.0 / HOURS_PER_YEAR),
)
uniform_spectrum_year.index.name = "Freq. (1/year)"
abs(uniform_spectrum_year).plot(
    subplots=True,
    sharey=True,
    logy=True,
    figsize=(12.5, 6.5),
    xlim=(0, 8),
    title="AmeriFlux - CASA CO$_2$ Flux difference spectra (log scale)",
)
plt.subplots_adjust(bottom=0.07, top=0.95, left=0.05, right=0.95)
plt.savefig("ameriflux-minus-casa-long-correlation-spectrum-years-log.pdf")

abs(uniform_spectrum_year).plot(
    subplots=True,
    sharey=True,
    logy=False,
    figsize=(12.5, 6.5),
    xlim=(0, 8),
    ylim=(0, 1000),
    title="AmeriFlux - CASA CO$_2$ Flux difference spectra (linear scale)",
)
plt.subplots_adjust(bottom=0.07, top=0.95, left=0.05, right=0.95)
plt.savefig("ameriflux-minus-casa-long-correlation-spectrum-years-linear.pdf")

plt.pause(1)

pca_results = PCA(
    uniform_corr_data.iloc[::4, :],
    # This should keep correlation components centered on zero
    demean=False,
    # Data is already standardized
    standardize=False,
    missing="drop-min",
    ncomp=10
)
# Boxplots of R^2 for tower ~ comp0, tower ~ comp0 + comp1, ...
pca_results.plot_rsquare()
plt.savefig("ameriflux-minus-casa-long-correlation-pca-rsquared.pdf")
# Essentially differencing the plot above
pca_results.plot_scree()
plt.savefig("ameriflux-minus-casa-long-correlation-pca-scree.pdf")
# In both those plots, look for elbow.

pca_results.scores.plot(
    subplots=True,
    sharey=True,
    ylim=(-0.03, 0.03),
    title="Principal compontents of correlations",
)
plt.savefig("ameriflux-minus-casa-long-correlation-pca-principal-components.png")

spectral_pca_results = PCA(
    uniform_spectrum.iloc[: uniform_spectrum.shape[0] // 4, :],
    # This should keep correlation components centered on zero
    demean=False,
    # Data is already standardized
    standardize=False,
    missing="drop-min",
    ncomp=10
)
# Boxplots of R^2 for tower ~ comp0, tower ~ comp0 + comp1, ...
spectral_pca_results.plot_rsquare()
plt.savefig("ameriflux-minus-casa-long-correlation-spectral-pca-rsquared.pdf")
# Essentially differencing the plot above
spectral_pca_results.plot_scree()
plt.savefig("ameriflux-minus-casa-long-correlation-spectral-pca-scree.pdf")
# In both those plots, look for elbow.

spectral_pca_results.scores.plot(
    subplots=True,
    sharey=True,
    ylim=(-0.03, 0.03),
    title="Principal compontents of correlations",
)
plt.savefig(
    "ameriflux-minus-casa-long-correlation-spectral-pca-principal-components.png"
)
plt.pause(1)
