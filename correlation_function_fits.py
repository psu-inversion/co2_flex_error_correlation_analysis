#!/usr/bin/env python
# ~*~ coding: utf8 ~*~
"""Find how well the correlation functions fit each tower.

I need to find a function that fits all towers decently.
"""
from __future__ import print_function, division

from enum import Enum
from math import pi
import collections
import itertools
import operator
import pprint

import numpy as np
import matplotlib.pyplot as plt
import numexpr as ne
import pandas as pd

HOURS_PER_DAY=24
DAYS_PER_DAY=1
DAYS_PER_WEEK=7
DAYS_PER_YEAR=365.2425

GLOBAL_DICT = dict(
    HOURS_PER_DAY=HOURS_PER_DAY,
    DAYS_PER_DAY=DAYS_PER_DAY,
    DAYS_PER_WEEK=DAYS_PER_WEEK,
    DAYS_PER_YEAR=DAYS_PER_YEAR,
    PI_OVER_DAY=pi / DAYS_PER_DAY,
    TWO_PI_OVER_DAY=2 * pi / DAYS_PER_DAY,
    FOUR_PI_OVER_DAY=4 * pi / DAYS_PER_DAY,
    PI_OVER_YEAR=pi / DAYS_PER_YEAR,
    TWO_PI_OVER_YEAR=2 * pi / DAYS_PER_YEAR,
    FOUR_PI_OVER_YEAR=4 * pi / DAYS_PER_YEAR,
)

TOPK = 7


class CorrelationPart(Enum):
    """The parts of a correlation function."""

    DAILY = "Daily"
    DM = "Annual Modulation of Daily"
    ANN = "Annual"

    D = "Daily"
    A = "Annual"

    # DAILY = D
    DAILY_MODULATION = DM
    ANNUAL = A

    def is_modulation(self):
        """Find whether CorrelationPart is its own part or modulates another.

        Returns
        -------
        bool
        """
        return self == CorrelationPart.DAILY_MODULATION

    def get_period(self):
        """Get the period of variations in this part.

        Returns
        -------
        period: str
        """
        if self == CorrelationPart.DAILY:
            return "DAY"
        return "YEAR"

    def get_short_name(self):
        """Get a one- or two-character description of the class.

        Returns
        -------
        str
        """
        if self == CorrelationPart.DAILY_MODULATION:
            return "dm"
        return self.name[0].lower()


assert list(CorrelationPart) == [CorrelationPart.DAILY, CorrelationPart.DM, CorrelationPart.ANN]

class PartForm(Enum):
    """Describe one part of a correlation function."""

    O = "None"
    C = "3-term cosine series"
    P = "Exponential sine-squared"

    NONE = O
    COSINE = C
    PERIODIC = P
    EXPSIN2 = P

    def get_short_name(self):
        """Get a short name for the form.

        Returns
        -------
        char
        """
        return self.name.lower().replace("o", "0")

    def get_parameters(self, part):
        """Get the parameters added by including this form in that part.

        Parameters
        ----------
        part: CorrelationPart

        Returns
        -------
        list of str
        """
        result = []
        if not part.is_modulation():
            result.extend(("{0:s}_coef", "{0:s}_timescale"))
        if self == PartForm.COSINE:
            result.extend(("{0:s}_coef1", "{0:s}_coef2"))
        elif self == PartForm.PERIODIC:
            result.append("{0:s}_width")
        part_name_lower = part.name.lower()
        return [coef.format(part_name_lower) for coef in result]

    def get_expression(self, part):
        """Get the expression for this form in that part.

        Parameters
        ----------
        part: CorrelationPart

        Returns
        -------
        expression: str
        """
        if self == PartForm.NONE:
            if part.is_modulation():
                return "1"
            return "0"
        if part == CorrelationPart.DAILY:
            prefix = "{0:s}_coef * exp(-tdata / ({0:s}_timescale * DAYS_PER_WEEK))"
        elif part == CorrelationPart.ANNUAL:
            prefix = "{0:s}_coef * exp(-tdata / ({0:s}_timescale * DAYS_PER_YEAR))"
        if self == PartForm.COSINE:
            main = (
                "((1 - {0:s}_coef1 - {0:s}_coef2) + "
                "{0:s}_coef1 * cos(TWO_PI_OVER_{time:s} * tdata) + "
                "{0:s}_coef2 * cos(FOUR_PI_OVER_{time:s} * tdata))"
            )
        elif self == PartForm.PERIODIC:
            main = (
                "exp(-(sin(PI_OVER_{time:s} * tdata) / {0:s}_width) ** 2)"
            )

        if not part.is_modulation():
            # The exponential die-off is only for the main
            # correlations.  Including it also on the modulation would
            # introduce problems with how to tell the two apart.
            main = "{0:s} * {1:s}".format(prefix, main)

        return main.format(
            part.name.lower(),
            time=part.get_period(),
        )


def is_valid_combination(part_daily, part_day_mod, part_annual):
    """Find whether this is a valid combination.

    Don't modulate a nonexistent cycle.

    Parameters
    ----------
    part_daily: PartForm
    part_day_mod: PartForm
    part_annual: PartForm

    Returns
    -------
    bool
    """
    if part_daily == PartForm.NONE and part_day_mod != PartForm.NONE:
        return False
    return True

def get_full_expression(part_daily, part_day_mod, part_annual):
    """Get the full expression with the given parts.

    Expression is for a single value of tdata, which requires either
    numpy semantics for tdata or an explicit loop handled by
    surrounding code.

    There is no attempt to re-use storage, which may make pure-numpy
    implementations slower.  Numexpr will do this for you.

    Requires tdata, DAYS_PER_{DAY,WEEK,YEAR}, HOURS_PER_DAY,
    {,TWO_,FOUR_}PI_OVER_{DAY,YEAR} and parameters.

    Parameters
    ----------
    part_daily: PartForm
    part_day_mod: PartForm
    part_annual: PartForm

    Returns
    -------
    expression: str
    """
    return (
        "{0:s} * {1:s} + {2:s} + "
        "resid_coef * exp(-tdata / (resid_timescale * DAYS_PER_WEEK)) + "
        "ec_coef * exp(-tdata / ec_timescale * HOURS_PER_DAY)"
    ).format(
        part_daily.get_expression(CorrelationPart.DAILY),
        part_day_mod.get_expression(CorrelationPart.DAILY_MODULATION),
        part_annual.get_expression(CorrelationPart.ANNUAL),
    )

def get_full_parameter_list(part_daily, part_day_mod, part_annual):
    """Get the full parameter list for the given expression.

    Parameters
    ----------
    part_daily: PartForm
    part_day_mod: PartForm
    part_annual: PartForm

    Returns
    -------
    param_list: list of str
    """
    result = [
        param
        for form, time in zip(
            (part_daily, part_day_mod, part_annual),
            (CorrelationPart.DAILY, CorrelationPart.DAILY_MODULATION,
             CorrelationPart.ANNUAL),
        )
        for param in form.get_parameters(time)
    ]
    result.extend(["resid_coef", "resid_timescale", "ec_coef", "ec_timescale"])
    return result

def get_weighted_fit_expression(part_daily, part_day_mod, part_annual):
    """Get the full expression with the given parts.

    Expression is for a single value of tdata, which requires either
    numpy semantics for tdata or an explicit loop handled by
    surrounding code.

    There is no attempt to re-use storage, which may make pure-numpy
    implementations slower.  Numexpr will do this for you.

    Expression will require tdata, num_pairs, exmpirical_correlogram,
    DAYS_PER_{DAY,WEEK,YEAR}, HOURS_PER_DAY,
    {,TWO_,FOUR_}PI_OVER_{DAY,YEAR} and parameters.

    Parameters
    ----------
    part_daily: PartForm
    part_day_mod: PartForm
    part_annual: PartForm

    Returns
    -------
    expression: str

    """
    return (
        "sum(num_pairs * (empirical_correlogram - ({0:s})) ** 2)"
        .format(get_full_expression(part_daily, part_day_mod, part_annual))
    )


if __name__ == "__main__":
    print("Reading coefficient data", flush=True)
    coef_data = pd.read_csv(
        "ameriflux-minus-casa-all-towers-parameters.csv",
        index_col=["Site", "Correlation Function"],
    ).dropna(how="all")
    amf_sites = coef_data.index.get_level_values(0).unique()
    print("Reading correlation data", flush=True)
    corr_data1 = pd.read_csv(
        "ameriflux-minus-casa-half-hour-towers-autocorrelation-functions.csv",
        index_col=0
    )
    corr_data2 = pd.read_csv(
        "ameriflux-minus-casa-hour-towers-autocorrelation-functions.csv",
        index_col=0
    )
    corr_data = pd.concat([corr_data1, corr_data2], axis=1)
    corr_data.index = pd.TimedeltaIndex(corr_data.index)
    corr_data.index.name = "Time separation"
    corr_data = corr_data.astype(np.float32)
    print("Have correlation data", flush=True)

    pair_counts1 = pd.read_csv(
        "ameriflux-minus-casa-half-hour-towers-pair-counts.csv",
        index_col=0
    )
    pair_counts2 = pd.read_csv(
        "ameriflux-minus-casa-hour-towers-pair-counts.csv",
        index_col=0
    )
    pair_counts = pd.concat([pair_counts1, pair_counts2], axis=1)
    pair_counts.index = pd.TimedeltaIndex(pair_counts.index)
    print("Have pair counts", flush=True)

    tower_lags = pair_counts.index.values.astype("m8[h]").astype("u8")
    tower_lags -= tower_lags[0]
    # In units of days
    tower_lags = tower_lags.astype(np.float32) / HOURS_PER_DAY

    topk_fits = {}
    topk_counts = collections.Counter()
    np.set_printoptions(formatter={"float": "{0:11.0f}".format})

    for tower in amf_sites:
        num_pairs = pair_counts.loc[:, tower].dropna()
        tdata = tower_lags[:len(num_pairs)]
        tdata = tdata[num_pairs > 0]
        num_pairs = num_pairs[num_pairs > 0]
        empirical_correlogram = corr_data.loc[num_pairs.index, tower]
        fit_quality = {}
        for form_daily, form_day_mod, form_annual in itertools.product(
                PartForm, PartForm, PartForm
        ):
            if not is_valid_combination(form_daily, form_day_mod, form_annual):
                continue
            try:
                tower_coefficients = dict(coef_data.loc[
                    (
                        tower,
                        "d{0:s}_dm{1:s}_a{2:s}_numexpr_fn".format(
                            form_daily.get_short_name(),
                            form_day_mod.get_short_name(),
                            form_annual.get_short_name(),
                        ),
                    ),
                    :
                ].dropna())
            except KeyError:
                # Fit crashed, or not enough data for me to be
                # comfortable fitting it.
                continue
            for newname, oldname in zip(
                    ("daily", "ann", "resid", "ec"),
                    ("Td", "Ta", "To", "Tec")
            ):
                tower_coefficients[newname + "_timescale"] = (
                    tower_coefficients[oldname]
                )
            local_dict = tower_coefficients.copy()
            local_dict["num_pairs"] = num_pairs
            local_dict["tdata"] = tdata
            local_dict["empirical_correlogram"] = empirical_correlogram
            fit_quality[form_daily, form_day_mod, form_annual] = ne.evaluate(
                get_weighted_fit_expression(form_daily, form_day_mod, form_annual),
                local_dict=local_dict,
                global_dict=GLOBAL_DICT
            )
        sorted_fits = sorted(
            list(fit_quality.keys()),
            key=fit_quality.__getitem__
        )
        topk_fits[tower] = sorted_fits[:TOPK]
        print(
            tower,
            ["d{0:s}_dm{1:s}_a{2:s}".format(
                *[form.get_short_name() for form in fun]
            ) for fun in sorted_fits[:TOPK]],
            np.array(
                [fit_quality[fun] for fun in sorted_fits[:TOPK]],
                dtype=np.float32,
            ),
        )
        for good_fit in sorted_fits[:TOPK]:
            topk_counts[good_fit] += 1

    print("Number of towers:", len(amf_sites))
    for fun, count in topk_counts.most_common(5):
        print(
            "Good function: d{0:s}_dm{1:s}_a{2:s}".format(
                *[form.get_short_name() for form in fun]
            ),
            "\tCount in top {0:d}:".format(TOPK),
            count
        )
            

### TOPK = 1
# Number of towers: 69
# Good function: dc_dmc_ac        Count in top 1: 27
# Good function: dc_dmc_ap        Count in top 1: 22
# Good function: dp_dmc_ac        Count in top 1: 8
# Good function: dp_dmp_ac        Count in top 1: 5
# Good function: dc_dmp_ap        Count in top 1: 2
### TOPK = 2
# Number of towers: 69
# Good function: dc_dmc_ac        Count in top 2: 46
# Good function: dc_dmc_ap        Count in top 2: 37
# Good function: dp_dmc_ac        Count in top 2: 16
# Good function: dc_dmp_ac        Count in top 2: 11
# Good function: dp_dmp_ac        Count in top 2: 9
### TOPK = 3
# Number of towers: 69
# Good function: dc_dmc_ac        Count in top 3: 52
# Good function: dc_dmc_ap        Count in top 3: 41
# Good function: dp_dmc_ac        Count in top 3: 27
# Good function: dc_dmp_ac        Count in top 3: 19
# Good function: dc_dmp_ap        Count in top 3: 18
### TOPK = 4
# Number of towers: 69
# Good function: dc_dmc_ac        Count in top 4: 56
# Good function: dc_dmc_ap        Count in top 4: 47
# Good function: dp_dmc_ac        Count in top 4: 36
# Good function: dc_dmp_ac        Count in top 4: 31
# Good function: dc_dmp_ap        Count in top 4: 29
### TOPK = 5
# Number of towers: 69
# Good function: dc_dmc_ac        Count in top 5: 59
# Good function: dc_dmc_ap        Count in top 5: 50
# Good function: dc_dmp_ac        Count in top 5: 48
# Good function: dp_dmc_ac        Count in top 5: 45
# Good function: dc_dmc_a0        Count in top 5: 33
### TOPK = 6
# Number of towers: 69
# Good function: dc_dmc_ac        Count in top 6: 65
# Good function: dc_dmp_ac        Count in top 6: 53
# Good function: dc_dmc_ap        Count in top 6: 51
# Good function: dp_dmc_ac        Count in top 6: 49
# Good function: dc_dmp_ap        Count in top 6: 41
### TOPK = 7
# Number of towers: 69
# Good function: dc_dmc_ac        Count in top 7: 66
# Good function: dc_dmp_ac        Count in top 7: 62
# Good function: dp_dmc_ac        Count in top 7: 61
# Good function: dc_dmc_ap        Count in top 7: 53
# Good function: dc_dmp_ap        Count in top 7: 47
