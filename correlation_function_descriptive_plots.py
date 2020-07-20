#!/usr/bin/env python
# ~*~ coding: utf8 ~*~
"""Make plots to explain the correlation functions.

They will need to have only four or six days per year to be
understandable.
"""
import textwrap

import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
import seaborn as sns

import correlation_function_fits

############################################################
# Set up constants for plots

# For deomonstration purposes only
DAYS_PER_YEAR = 6
DAYS_PER_DAY = 1
EPS = 1e-13

GLOBAL_DICT = correlation_function_fits.GLOBAL_DICT.copy()
GLOBAL_DICT.update({
    "DAYS_PER_DAY": DAYS_PER_DAY,
    "PI_OVER_DAY": np.pi / DAYS_PER_DAY,
    "TWO_PI_OVER_DAY": 2 * np.pi / DAYS_PER_DAY,
    "FOUR_PI_OVER_DAY": 4 * np.pi / DAYS_PER_DAY,
    "DAYS_PER_YEAR": DAYS_PER_YEAR,
    "PI_OVER_YEAR": np.pi / DAYS_PER_YEAR,
    "TWO_PI_OVER_YEAR": 2 * np.pi / DAYS_PER_YEAR,
    "FOUR_PI_OVER_YEAR": 4 * np.pi / DAYS_PER_YEAR,
})

TIMES = np.linspace(0, DAYS_PER_YEAR, 601)
AX_HEIGHT = 5.5
AX_WIDTH = 6

LOCAL_DICT = {"tdata": TIMES}
for part in ("daily", "dm", "ann"):
    LOCAL_DICT.update({
        "{part:s}_coef".format(part=part): 1,
        "{part:s}_coef1".format(part=part): 0.5,
        "{part:s}_coef2".format(part=part): 0.25,
        "{part:s}_width".format(part=part): 0.4,
        # I'm ignoring this for now
        "{part:s}_timescale".format(part=part): 100,
    })

############################################################
# Set plotting defaults
sns.set_context("paper")
sns.set(style="whitegrid")
sns.set_palette("colorblind")

############################################################
# create plots

# Describe axes I'm using
fig, ax = plt.subplots(1, 1, figsize=(AX_WIDTH, AX_HEIGHT))
ax.set_xlabel("Lag Time (days)")
ax.set_xticks(np.arange(0, DAYS_PER_YEAR + EPS, 1))
ax.set_title("One year")

ax.set_xlim(0, DAYS_PER_YEAR)
ax.set_ylim(-1, 1)
ax.set_xticks(np.arange(0, DAYS_PER_YEAR + EPS, 3))
ax.set_xticks(np.arange(0, DAYS_PER_YEAR + EPS, 1), minor=True)

fig.tight_layout()
fig.savefig("demonstration-setup.pdf")
plt.close(fig)


# Show what the functions look like
PART_FORMS = sorted(
    correlation_function_fits.PartForm,
    key=lambda part_form: len(
        part_form.get_parameters(
            correlation_function_fits.CorrelationPart.DAILY
        )
    )
)

fig, axes = plt.subplots(
    len(correlation_function_fits.PartForm),
    len(correlation_function_fits.CorrelationPart),
    figsize=(AX_WIDTH, AX_HEIGHT),
    sharex=True, sharey=True,
)

for axes_row, part_form in zip(
        axes,
        PART_FORMS
):
    for ax, func_part in zip(
            axes_row, correlation_function_fits.CorrelationPart
    ):
        if func_part.is_modulation():
            expression = part_form.get_expression(func_part)
            expression += " * cos(TWO_PI_OVER_DAY * tdata)"
        else:
            expression = part_form.get_expression(func_part)
        ax.plot(
            *np.broadcast_arrays(
                TIMES,
                ne.evaluate(
                    expression,
                    global_dict=GLOBAL_DICT,
                    local_dict=LOCAL_DICT,
                ),
            )
        )
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, DAYS_PER_YEAR)
        ax.set_xticks(np.arange(0, DAYS_PER_YEAR + EPS, 3))
        ax.set_xticks(np.arange(0, DAYS_PER_YEAR + EPS, 1), minor=True)

for ax, part_form in zip(axes[:, 0], PART_FORMS):
    ax.set_ylabel(textwrap.fill(part_form.value, 14))

for ax, func_part in zip(
        axes[0, :],
        ("Daily Cycle", "Modulation\nof Daily Cycle", "Annual Cycle")
):
    ax.set_title(func_part)

for ax in axes[-1, :]:
    ax.set_xlabel("Time Lag (days)\n(6 days = 1 year)")

fig.tight_layout()
fig.savefig("demonstration-corr-fun-slots-and-forms.pdf")
fig.savefig("demonstration-corr-fun-slots-and-forms.png")
plt.close(fig)
