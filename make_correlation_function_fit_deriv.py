#!/usr/bin/env python
# ~*~ coding: utf8 ~*~
"""Write a file containing cost functions.

Also derivatives, ideally.
"""
from __future__ import print_function, division

import itertools

import numpy as np

from correlation_function_fits import (
    GLOBAL_DICT, CorrelationPart, PartForm,
    is_valid_combination, get_full_expression,
    get_full_parameter_list,
    get_weighted_fit_expression,
)

OUT_FILE_NAME = "flux_correlation_function_fits.pyx"

with open(OUT_FILE_NAME, "w") as out_file:
    out_file.write("""# cython: embedsignature=True
# cython: language_level=3str
from libc cimport math

import numpy as np
cimport numpy as np
import numexpr as ne

ctypedef fused floating_type:
    np.float32_t
    np.float64_t

cdef extern from "<math.h>" nogil:
    float sinf(float x)
    float cosf(float x)
    float expf(float x)

cdef floating_type cycos(floating_type x):
    if floating_type is float:
        return cosf(x)
    elif floating_type is double:
        return math.cos(x)

cdef floating_type cysin(floating_type x):
    if floating_type is float:
        return sinf(x)
    elif floating_type is double:
        return math.sin(x)

cdef floating_type cyexp(floating_type x):
    if floating_type is float:
        return expf(x)
    elif floating_type is double:
        return math.exp(x)

ctypedef floating_type (*float_fun)(floating_type)

cdef float HOURS_PER_DAY = 24.
cdef float DAYS_PER_DAY = 1.
cdef float DAYS_PER_WEEK = 7.
cdef float DAYS_PER_YEAR = 365.2425

cdef float HOURS_PER_YEAR = HOURS_PER_DAY * DAYS_PER_YEAR

cdef float PI_OVER_DAY = np.pi / DAYS_PER_DAY
cdef float TWO_PI_OVER_DAY = 2 * PI_OVER_DAY
cdef float FOUR_PI_OVER_DAY = 2 * TWO_PI_OVER_DAY

cdef float PI_OVER_YEAR = np.pi / DAYS_PER_YEAR
cdef float TWO_PI_OVER_YEAR = 2 * PI_OVER_YEAR
cdef float FOUR_PI_OVER_YEAR = 2 * TWO_PI_OVER_YEAR

cdef dict GLOBAL_DICT = {
    "HOURS_PER_DAY": HOURS_PER_DAY,
    "DAYS_PER_WEEK": DAYS_PER_WEEK,
    "DAYS_PER_YEAR": DAYS_PER_YEAR,
    "PI_OVER_DAY": PI_OVER_DAY,
    "TWO_PI_OVER_DAY": TWO_PI_OVER_DAY,
    "FOUR_PI_OVER_DAY": FOUR_PI_OVER_DAY,
    "PI_OVER_YEAR": PI_OVER_YEAR,
    "TWO_PI_OVER_YEAR": TWO_PI_OVER_YEAR,
    "FOUR_PI_OVER_YEAR": FOUR_PI_OVER_YEAR,
}
""")
    def pr(thing):
        print(thing, flush=True)
        return thing
    for forms in itertools.product(PartForm, PartForm, PartForm):
        if not is_valid_combination(*forms):
            continue
        print(forms)
        out_file.write("""
def {func_name:s}_ne(parameters, tdata, empirical_correlogram, pair_count):
    return ne.evaluate(
        "{weighted_sum_expr:s}",
        local_dict={{
{param_names_from_parameters:s}
            "tdata": tdata,
            "empirical_correlogram": empirical_correlogram,
            "pair_count": pair_count,
        }},
        global_dict=GLOBAL_DICT,
    )
""".format(
    func_name="_".join([
        "{0:s}{1:s}".format(
            part.get_short_name(),
            form.get_short_name(),
        )
        for part, form in zip(CorrelationPart, forms)
    ]),
    weighted_sum_expr=get_weighted_fit_expression(*forms),
    param_names_from_parameters="".join([
        "            \"{param_name:s}\": parameters[{i:d}],\n"
        .format(i=i, param_name=param_name)
        for i, param_name in enumerate(get_full_parameter_list(*forms))
    ])
))

        out_file.write("""
def {function_name:s}_loop(
    floating_type[::1] parameters,
    floating_type[::1] tdata_base,
    floating_type[::1] empirical_correlogram,
    floating_type[::1] pair_count,
):
    cdef floating_type weighted_fit = 0.0
    cdef floating_type[::1] deriv = np.zeros_like(parameters)
    cdef long int n_parameters = len(parameters)

{params_from_parameters:s}

    cdef long int i = 0, j = 0

    cdef floating_type daily_corr, dm_corr
    cdef floating_type ann_corr
    cdef floating_type resid_corr, ec_corr

    cdef float_fun exp = cyexp
    cdef float_fun cos = cycos
    cdef float_fun sin = cysin

    resid_timescale *= DAYS_PER_WEEK
    ec_timescale /= HOURS_PER_DAY

    for i in range(len(tdata_base)):
        tdata = tdata_base[i]

        daily_corr = {daily_form:s}
        dm_corr = {daily_modulation_form:s}
        weighted_fit += daily_corr * dm_corr
        {accum_day_deriv:s}
        {accum_dm_deriv:s}

        ann_corr = {annual_form:s}
        weighted_fit += ann_corr
        {accum_ann_deriv:s}

        resid_corr = resid_coef * exp(-tdata / resid_timescale)
        weighted_fit += resid_corr
        deriv[n_parameters - 4] += exp(-tdata / resid_timescale)
        deriv[n_parameters - 3] += resid_corr * tdata / ec_timescale ** 2

        ec_corr = ec_coef * exp(-tdata / ec_timescale)
        weighted_fit += ec_corr
        deriv[n_parameters - 2] += exp(-tdata / ec_timescale)
        deriv[n_parameters - 1] += ec_corr * tdata / ec_timescale ** 2

    return weighted_fit
""".format(
    function_name="_".join([
        "{0:s}{1:s}".format(
            part.get_short_name(),
            form.get_short_name(),
        )
        for part, form in zip(CorrelationPart, forms)
    ]),
    params_from_parameters="".join([
        "    cdef floating_type {param_name:s} = parameters[{i:d}]\n"
        .format(i=i, param_name=param_name)
        for i, param_name in enumerate(get_full_parameter_list(*forms))
    ]),
    daily_form=forms[0].get_expression(CorrelationPart.DAILY),
    daily_modulation_form=forms[1].get_expression(CorrelationPart.DAILY_MODULATION),
    annual_form=forms[2].get_expression(CorrelationPart.ANNUAL),
    accum_day_deriv="",
    accum_dm_deriv="",
    accum_ann_deriv="",
))


from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    author="DWesl",
    version="0.0.0.dev0",
    ext_modules=cythonize(
        Extension(
            "flux_correlation_functions",
            [OUT_FILE_NAME],
            include_dirs=[np.get_include()],
        ),
        include_path=[np.get_include()],
        compiler_directives=dict(embedsignature=True),
    )
)
