#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray
import statsmodels.formula.api as smf

ds = xarray.open_dataset("ameriflux-minus-casa-autocorrelation-function-multi-tower-fits-20.nc4")
df = ds["cross_validation_error"].to_dataframe()
df["daily_cycle"] = pd.Categorical(df["daily_cycle"], categories=["None", "Geostatistical", "Exponential sine-squared", "3-term cosine series"], ordered=True)
df["annual_cycle"] = pd.Categorical(df["annual_cycle"], categories=["None", "Geostatistical", "Exponential sine-squared", "3-term cosine series"], ordered=True)
df["annual_modulation_of_daily_cycle"] = pd.Categorical(df["annual_modulation_of_daily_cycle"], categories=["None", "Geostatistical", "Exponential sine-squared", "3-term cosine series"], ordered=True)

print(smf.ols("cross_validation_error ~ has_daily_cycle + daily_cycle_has_modulation + has_annual_cycle", df).fit().summary())
print(smf.ols("cross_validation_error ~ has_daily_cycle + daily_cycle_has_modulation + has_annual_cycle + daily_cycle_has_parameters + daily_cycle_modulation_has_parameters + annual_cycle_has_parameters", df).fit().summary())
print(smf.ols("cross_validation_error ~ daily_cycle + annual_modulation_of_daily_cycle + annual_cycle", df).fit().summary())

df_for_plot = df.replace("Geostatistical", "Geostat.").replace(
    "3-term cosine series", "Cosines"
).replace("Exponential sine-squared", "Exp. sin\N{SUPERSCRIPT TWO}").rename(
    columns={"annual_modulation_of_daily_cycle": "daily cycle\nmodulation"}
)
grid = sns.catplot(
    x="cross_validation_error", y="daily cycle\nmodulation",
    row="daily_cycle", col="annual_cycle",
    data=df_for_plot, height=1.9, aspect=1.4,
    margin_titles=True, kind="box",
)
grid.fig.savefig("multi-tower-cross-validation-error-by-function.pdf")

plt.pause(1)
