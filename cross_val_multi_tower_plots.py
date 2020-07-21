#!/usr/bin/env python
from __future__ import print_function
import collections

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import seaborn as sns
import xarray
import statsmodels.api as sm
import statsmodels.formula.api as smf

sns.set_context("paper")
sns.set(style="whitegrid")
sns.set_palette("colorblind")


############################################################
# Define description function
def long_description(df, ci_width=0.95):
    """Print longer description of df.

    Parameters
    ----------
    df: pd.DataFrame
    ci_width: float
         Width of confidence intervals.
         Must between 0 and 1.

    Returns
    -------
    pd.DataFrame
    """
    df_stats = df.describe()
    df_stats_loc = df_stats.loc
    # Robust measures of scale
    df_stats_loc["IQR", :] = df_stats_loc["75%", :] - df_stats_loc["25%", :]
    df_stats_loc["mean abs. dev.", :] = df.mad()
    deviation_from_median = df - df_stats_loc["50%", :]
    df_stats_loc["med. abs. dev.", :] = deviation_from_median.abs().median()
    # Higher-order moments
    df_stats_loc["Fisher skewness", :] = df.skew()
    df_stats_loc["Y-K skewness", :] = (
        (df_stats_loc["75%", :] + df_stats_loc["25%", :] -
         2 * df_stats_loc["50%", :]) /
        (df_stats_loc["75%", :] - df_stats_loc["25%", :])
    )
    df_stats_loc["Fisher kurtosis", :] = df.kurt()
    # Confidence intervals
    for col_name in df:
        # I'm already dropping NAs for the rest of these.
        mean, var, std = scipy.stats.bayes_mvs(
            df[col_name].dropna(),
            alpha=ci_width
        )
        # Record mean
        df_stats_loc["Mean point est", col_name] = mean[0]
        df_stats_loc[
            "Mean {width:2d}%CI low".format(width=round(ci_width * 100)),
            col_name
        ] = mean[1][0]
        df_stats_loc[
            "Mean {width:2d}%CI high".format(width=round(ci_width * 100)),
            col_name
        ] = mean[1][1]
        # Record var
        df_stats_loc["Var. point est", col_name] = var[0]
        df_stats_loc[
            "Var. {width:2d}%CI low".format(width=round(ci_width * 100)),
            col_name
        ] = var[1][0]
        df_stats_loc[
            "Var. {width:2d}%CI high".format(width=round(ci_width * 100)),
            col_name
        ] = var[1][1]
        # Record Std Dev
        df_stats_loc["std point est", col_name] = std[0]
        df_stats_loc[
            "std {width:2d}%CI low".format(width=round(ci_width * 100)),
            col_name
        ] = std[1][0]
        df_stats_loc[
            "std {width:2d}%CI high".format(width=round(ci_width * 100)),
            col_name
        ] = std[1][1]
    return df_stats


############################################################
# Read in and merge datasets
# ds1 = xarray.open_dataset(
#     "ameriflux-minus-casa-autocorrelation-function-multi-tower-fits-200splits-run1.nc4"
# )
# ds2 = xarray.open_dataset(
#     "ameriflux-minus-casa-autocorrelation-function-multi-tower-fits-250splits-run1.nc4"
# )
# ds = xarray.concat(
#     [
#         ds1.assign_coords(
#             splits=pd.RangeIndex(0, ds1.dims["splits"])
#         ),
#         ds2.assign_coords(
#             splits=pd.RangeIndex(ds1.dims["splits"], ds1.dims["splits"] + ds2.dims["splits"])
#         ),
#     ],
#     dim="splits",
# )

# # Fill in the training towers
# ALL_TOWERS = np.unique(ds["validation_towers"].values.astype("U6").flat)
# ds["training_towers"] = (
#     ("splits", "n_training"),
#     np.array([
#         np.setdiff1d(ALL_TOWERS, val_towers)
#         for val_towers in ds["validation_towers"].values.astype("U6")
#     ])
# )
# del ds1, ds2

# ds3 = xarray.open_dataset(
#     "ameriflux-minus-casa-autocorrelation-function-multi-tower-fits-300splits-run1.nc4"
# )
# ds4 = xarray.open_dataset(
#     "ameriflux-minus-casa-autocorrelation-function-multi-tower-fits-300splits-run2.nc4"
# )
# ds = xarray.concat(
#     [
#         ds,
#         ds3.assign_coords(
#             splits=pd.RangeIndex(
#                 ds.dims["splits"],
#                 ds.dims["splits"] + ds3.dims["splits"]
#             )
#         ),
#         ds4.assign_coords(
#             splits=pd.RangeIndex(
#                 ds.dims["splits"] + ds3.dims["splits"],
#                 ds.dims["splits"] + ds3.dims["splits"] + ds4.dims["splits"],
#             )
#         )
#     ],
#     dim="splits",
# )
# del ds3, ds4

# ds.coords["n_parameters"] = ds["optimized_parameters"].isel(splits=0).count(
#     "parameter_name"
# ).drop_vars("splits").astype("i1")

# encoding = {
#     var_name: {"zlib": True, "_FillValue": -9.999e9}
#     for var_name in ds.data_vars
# }
# encoding.update({coord_name: {"_FillValue": None} for coord_name in ds.coords})
# ds.to_netcdf("multi-tower-cross-validation-error-data-1050-splits.nc4",
#              encoding=encoding, format="NETCDF4_CLASSIC")
ds = xarray.open_dataset(
    "multi-tower-cross-validation-error-data-1050-splits.nc4"
)

############################################################
# Turn dataset into dataframe
df = ds["cross_validation_error"].to_dataframe().replace(
    "Geostatistical", "Geostat."
).replace(
    "3-term cosine series", "Cosines"
).replace(
    "Exponential sine-squared", "Exp. sin\N{SUPERSCRIPT TWO}"
)
for slot_var in ("daily_cycle", "annual_cycle", "annual_modulation_of_daily_cycle"):
    df[slot_var] = pd.Categorical(
        df[slot_var],
        categories=["None", "Geostat.", "Exp. sin\N{SUPERSCRIPT TWO}", "Cosines"],
        ordered=True
    )

print(
    "Do the various slots improve the fit?",
    smf.ols(
        "cross_validation_error ~ has_daily_cycle + "
        "daily_cycle_has_modulation + has_annual_cycle",
        df
    ).fit().summary(),
    sep="\n"
)
print(
    "Does having something in the slots improve the fit and does having parameters improve on that?",
    smf.ols(
        "cross_validation_error ~ has_daily_cycle + "
        "daily_cycle_has_modulation + has_annual_cycle + "
        "daily_cycle_has_parameters + daily_cycle_modulation_has_parameters + "
        "annual_cycle_has_parameters",
        df
    ).fit().summary(),
    sep="\n"
)
print(
    "Which functional form does best in each slot?",
    smf.ols(
        "cross_validation_error ~ daily_cycle + "
        "annual_modulation_of_daily_cycle + annual_cycle",
        df
    ).fit().summary(),
    sep="\n"
)

# import formulaic
import patsy
models = []
results = []

# Three slots for things to go in
for i in range(1, 3 + 1):
    formula = (
        "cross_validation_error ~ "
        "(daily_cycle + annual_modulation_of_daily_cycle + annual_cycle)"
        " ** {i:d}".format(i=i)
    )
    # full_y, full_X = formulaic.Formula(
    #     formula
    # ).get_model_matrix(df.dropna())
    full_y, full_X = patsy.dmatrices(formula, df, return_type="dataframe")
    col_index_to_keep = [
        i
        for i, col in enumerate(full_X.columns)
        if (
                (
                    ":daily_cycle[T.Geostat.]" not in col and
                    not col.startswith("daily_cycle[T.Geostat.]:")
                ) or
                "annual_modulation_of_daily_cycle" not in col
        )
    ]
    reduced_X = full_X.iloc[:, col_index_to_keep]
    reduced_X.design_info = patsy.DesignInfo(
        column_names=list(reduced_X.columns),
        factor_infos=full_X.design_info.factor_infos,
        term_codings=collections.OrderedDict([
            (
                term,
                [
                    subterm
                    if (
                            len(subterm.factors) == 1 or not
                            (
                                patsy.EvalFactor("daily_cycle") in subterm.factors
                                and
                                patsy.EvalFactor("annual_modulation_of_daily_cycle") in subterm.factors
                            )
                    ) else
                    patsy.SubtermInfo(
                        subterm.factors,
                        {
                            factor: (
                                matrix
                                if factor != patsy.EvalFactor("daily_cycle") else
                                patsy.ContrastMatrix(
                                    # Skip daily_cycle[T.Geostat.]
                                    matrix.matrix[[0, 0, 2, 3], 1:],
                                    # matrix.column_suffixes
                                    [name for name in matrix.column_suffixes
                                     if "Geostat" not in name]
                                )
                            )
                            for factor, matrix in subterm.contrast_matrices.items()
                        },
                        # Four things to go in each slot
                        (4 - 1) ** len(subterm.factors) -
                        (4 - 1) ** (len(subterm.factors) - 1) +
                        0,
                    )
                    for subterm in subterms
                ]
            )
            for term, subterms in full_X.design_info.term_codings.items()
        ]),
    )
    model = sm.OLS(full_y, reduced_X)
    result = model.fit()
    models.append(model)
    results.append(result)

from statsmodels.stats.anova import anova_lm
anova_lm(*results)

df_for_plot = df.rename(
    columns={
        "annual_modulation_of_daily_cycle": "Daily Cycle\nModulation",
        "annual_cycle": "Annual Cycle",
        "daily_cycle": "Daily Cycle",
    }
)

############################################################
# Draw boxplots showing details of distribution
grid = sns.catplot(
    x="cross_validation_error", y="Daily Cycle\nModulation",
    row="Daily Cycle", col="Annual Cycle",
    data=df_for_plot, height=1.7, aspect=1.7,
    margin_titles=True, kind="box",
    sharex=True, sharey=True,
    showmeans=True,
    meanprops={"markerfacecolor": "white",
               "markeredgecolor": "k"},
)

for ax in grid.axes[:, -1]:
    for child in ax.get_children():
        if isinstance(child, plt.Text):
            child.set_visible(False)

grid.axes[0, -1].set_title("", visible=True)
grid.axes[0, 0].set_xlim(0, None)
grid.set_titles(
    row_template="{row_var: ^11s}\n{row_name: ^11s}",
    col_template="{col_var: ^11s}\n{col_name: ^11s}"
)
grid.set_xlabels("Cross-Validation\nError")
grid.fig.tight_layout()
grid.fig.savefig("multi-tower-cross-validation-error-by-function.pdf")
grid.fig.savefig("multi-tower-cross-validation-error-by-function.png")

for ax in grid.axes.flat:
    ax.set_xscale("log")
    ax.set_xlim(0.08e9, 4e9)

grid.fig.savefig("multi-tower-log-cross-validation-error-by-function.pdf")
grid.fig.savefig("multi-tower-log-cross-validation-error-by-function.png")

############################################################
# Sorted box plots
mean_sort_order = df.groupby("correlation_function").mean().sort_values(
    "cross_validation_error"
).index

# Horizontal box plots
fig = plt.figure(figsize=(5.5, 11))
ax = sns.boxplot(
    x="cross_validation_error", y="correlation_function_short_name",
    data=df_for_plot.reindex(index=mean_sort_order, level=0),
    showmeans=True,
    meanprops={"markerfacecolor": "white",
               "markeredgecolor": "k"},
    # kind="box",
)
fig.subplots_adjust(left=0.21, top=1, bottom=0.05)
ax.set_ylabel("Correlation Function Short Name")
ax.set_xlabel("Cross-Validation Error")
fig.tight_layout()
fig.savefig("multi-tower-cross-validation-error-sorted-long.pdf")
fig.savefig("multi-tower-cross-validation-error-sorted-long.png")

ax.set_xscale("log")
fig.savefig("multi-tower-log-cross-validation-error-sorted-long.pdf")
fig.savefig("multi-tower-log-cross-validation-error-sorted-long.png")

# Vertical box plots
fig = plt.figure(figsize=(12, 5.5))
ax = sns.boxplot(
    y="cross_validation_error", x="correlation_function_short_name",
    data=df_for_plot.reindex(index=mean_sort_order, level=0),
    showmeans=True,
    meanprops={"markerfacecolor": "white",
               "markeredgecolor": "k"},
    # kind="box",
)
fig.autofmt_xdate()
fig.subplots_adjust(left=0.21, top=1, bottom=0.05)
ax.set_xlabel("Correlation Function Short Name")
ax.set_ylabel("Cross-Validation Error")
fig.tight_layout()
fig.savefig("multi-tower-cross-validation-error-sorted-wide.pdf")
fig.savefig("multi-tower-cross-validation-error-sorted-wide.png")

ax.set_yscale("log")
fig.savefig("multi-tower-cross-validation-error-sorted-wide.pdf")
fig.savefig("multi-tower-cross-validation-error-sorted-wide.png")

############################################################
# Calculate summary statistics
ldesc = long_description(
    # Make the column names shorter
    df.reset_index()
    .drop(columns="correlation_function")
    .rename(columns={"correlation_function_short_name": "correlation_function"})
    .set_index(["correlation_function", "splits"])
    # I only care about the cross-validation error for this
    ["cross_validation_error"]
    # Turn it back into a rectangle: rows are splits, columns are functions
    .unstack(0)
)
ldesc.loc["n_parameters", :] = ds.coords["n_parameters"].to_dataframe(
).set_index("correlation_function_short_name")["n_parameters"].iloc[:, 0]

############################################################
# Plot cross-validation error as a function of complexity
fig = plt.figure(figsize=(4.5, 3.5))
ax = sns.scatterplot(x="n_parameters", y="mean", data=ldesc.T, x_jitter=True)
ax.set_ylabel("Mean Cross-Validation Error")
ax.set_xlabel("Number of Parameters")
fig.tight_layout()
fig.savefig("multi-tower-cross-validation-error-vs-n-params.pdf")
fig.savefig("multi-tower-cross-validation-error-vs-n-params.png")

ax.set_yscale("log")
fig.savefig("multi-tower-log-cross-validation-error-vs-n-params.pdf")
fig.savefig("multi-tower-log-cross-validation-error-vs-n-params.png")

plt.pause(1)

ldesc_ds = xarray.Dataset.from_dataframe(ldesc.T)
ldesc_ds["correlation_function"] = ldesc_ds["correlation_function"].astype("U9")
ldesc_ds["count"] = ldesc_ds["count"].astype("i2")
ldesc_ds["n_parameters"] = ldesc_ds["n_parameters"].astype("i1")
encoding = {name: {"_FillValue": None} for name in ldesc_ds.coords}
encoding.update({name: {"_FillValue": None} for name in ldesc_ds.data_vars})
ldesc_ds.to_netcdf("multi-tower-cross-validation-error-summary-1050-splits.nc4",
                   encoding=encoding, format="NETCDF4_CLASSIC")
