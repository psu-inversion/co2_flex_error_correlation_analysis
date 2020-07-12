#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray
import statsmodels.formula.api as smf

ds = xarray.open_dataset(
    "ameriflux-minus-casa-autocorrelation-function-multi-tower-fits-200splits-run1.nc4"
)
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

df_for_plot = df.rename(
    columns={
        "annual_modulation_of_daily_cycle": "Daily Cycle\nModulation",
        "annual_cycle": "Annual Cycle",
        "daily_cycle": "Daily Cycle",
    }
)

grid = sns.catplot(
    x="cross_validation_error", y="Daily Cycle\nModulation",
    row="Daily Cycle", col="Annual Cycle",
    data=df_for_plot, height=1.6, aspect=1.4,
    margin_titles=True, kind="box",
    sharex=True, sharey=True,
)
for ax in grid.axes[:, -1]:
    for child in ax.get_children():
        if isinstance(child, plt.Text):
            child.set_visible(False)

grid.axes[0, -1].set_title("", visible=True)
grid.set_titles(
    row_template="{row_var: ^11s}\n{row_name: ^11s}",
    col_template="{col_var: ^11s}\n{col_name: ^11s}"
)
grid.set_xlabels("Cross-Validation\nError")
grid.fig.savefig("multi-tower-cross-validation-error-by-function.pdf")

median_sort_order = df.groupby("correlation_function").median().sort_values(
    "cross_validation_error"
).index

fig = plt.figure(figsize=(5.5, 10))
ax = sns.boxplot(
    x="cross_validation_error", y="correlation_function_short_name",
    data=df_for_plot.reindex(index=median_sort_order, level=0),
    # kind="box",
)
fig.subplots_adjust(left=0.21, top=1, bottom=0.05)
ax.set_ylabel("Correlation Function Short Name")
ax.set_xlabel("Cross-Validation Error")

plt.pause(1)
