#!/usr/bin/env python
# ~*~ coding: utf8 ~*~
"""Write a file containing cost functions.

Also derivatives, ideally.
"""
from __future__ import print_function, division

import itertools

import numpy as np

from setuptools import setup, Extension
from Cython.Build import cythonize

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
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: gdb_debug=False
from libc cimport math
from cython.view cimport array as cvarray

import numpy as np
cimport numpy as np
import numexpr as ne
# from numpy cimport PyArray_Where as where

ctypedef fused floating_type:
    np.float32_t
    np.float64_t

cdef extern from "<math.h>" nogil:
    float sinf(float x)
    float cosf(float x)
    float expf(float x)

cdef inline floating_type cycos(floating_type x) nogil:
    if floating_type is float:
        return cosf(x)
    elif floating_type is double:
        return math.cos(x)

cdef inline floating_type cysin(floating_type x) nogil:
    if floating_type is float:
        return sinf(x)
    elif floating_type is double:
        return math.sin(x)

cdef inline floating_type cyexp(floating_type x) nogil:
    if floating_type is float:
        return expf(x)
    elif floating_type is double:
        return math.exp(x)

cdef inline floating_type where(bint cond, floating_type a, floating_type b) nogil:
    if cond:
        return a
    return b

ctypedef floating_type (*float_fun)(floating_type) nogil

cdef float HOURS_PER_DAY = 24.
cdef float DAYS_PER_DAY = 1.
cdef float DAYS_PER_WEEK = 7.
cdef float DAYS_PER_FORTNIGHT = 14.
cdef float DAYS_PER_YEAR = 365.2425
cdef float DAYS_PER_DECADE = 10 * DAYS_PER_YEAR

cdef float HOURS_PER_YEAR = HOURS_PER_DAY * DAYS_PER_YEAR

cdef float PI_OVER_DAY = np.pi / DAYS_PER_DAY
cdef float TWO_PI_OVER_DAY = 2 * PI_OVER_DAY
cdef float FOUR_PI_OVER_DAY = 2 * TWO_PI_OVER_DAY

cdef float PI_OVER_YEAR = np.pi / DAYS_PER_YEAR
cdef float TWO_PI_OVER_YEAR = 2 * PI_OVER_YEAR
cdef float FOUR_PI_OVER_YEAR = 2 * TWO_PI_OVER_YEAR

cdef dict GLOBAL_DICT = {
    "HOURS_PER_DAY": HOURS_PER_DAY,
    "DAYS_PER_DAY": DAYS_PER_DAY,
    "DAYS_PER_WEEK": DAYS_PER_WEEK,
    "DAYS_PER_FORTNIGHT": DAYS_PER_FORTNIGHT,
    "DAYS_PER_YEAR": DAYS_PER_YEAR,
    "DAYS_PER_DECADE": DAYS_PER_DECADE,
    "PI_OVER_DAY": PI_OVER_DAY,
    "TWO_PI_OVER_DAY": TWO_PI_OVER_DAY,
    "FOUR_PI_OVER_DAY": FOUR_PI_OVER_DAY,
    "PI_OVER_YEAR": PI_OVER_YEAR,
    "TWO_PI_OVER_YEAR": TWO_PI_OVER_YEAR,
    "FOUR_PI_OVER_YEAR": FOUR_PI_OVER_YEAR,
}
""")
    for forms in itertools.product(PartForm, PartForm, PartForm):
        if not is_valid_combination(*forms):
            continue
        print(forms)
        out_file.write("""
def {func_name:s}_fit_ne(parameters, tdata, empirical_correlogram, pair_count):
    return ne.evaluate(
        "{weighted_sum_expr:s}",
        local_dict={{
{param_names_from_parameters:s}
            "tdata": tdata,
            "empirical_correlogram": empirical_correlogram,
            "num_pairs": pair_count,
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
def {func_name:s}_curve_ne(
    tdata,
{parameters:s}
):
    return ne.evaluate(
        "{full_expr:s}",
        local_dict={{
{param_names_from_parameters:s}
            "tdata": tdata
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
    param_names_from_parameters="".join([
        "            \"{param_name:s}\": {param_name:s},\n"
        .format(param_name=param_name)
        for param_name in get_full_parameter_list(*forms)
    ]),
    parameters="".join([
        "    {param:s},\n".format(param=param_name)
        for param_name in get_full_parameter_list(*forms)
    ]),
    full_expr=get_full_expression(*forms)
))

        out_file.write("""
def {function_name:s}_fit_loop(
    np.float64_t[::1] parameters not None,
    floating_type[::1] tdata_base not None,
    floating_type[::1] empirical_correlogram not None,
    floating_type[::1] pair_count not None,
):
    cdef floating_type weighted_fit = 0.0
    cdef floating_type deriv[{n_parameters:d}]
    cdef floating_type here_deriv[{n_parameters:d}]
    cdef long int n_parameters = {n_parameters:d}

{params_from_parameters:s}

    cdef long int i = 0, j = 0

    cdef floating_type tdata
    cdef floating_type daily_corr, dm_corr
    cdef floating_type ann_corr
    cdef floating_type resid_corr, ec_corr
    cdef floating_type here_corr
    cdef floating_type deriv_common

    cdef float_fun exp = cyexp
    cdef float_fun cos = cycos
    cdef float_fun sin = cysin

    resid_timescale *= DAYS_PER_FORTNIGHT

    for j in range(n_parameters):
        deriv[j] = 0.0

    for i in range(len(tdata_base)):
        tdata = tdata_base[i]
        here_corr = 0.0

        daily_corr = {daily_form:s}
        dm_corr = {daily_modulation_form:s}
        here_corr += daily_corr * dm_corr
        {accum_day_deriv:s}
        {accum_dm_deriv:s}

        ann_corr = {annual_form:s}
        here_corr += ann_corr
        {accum_ann_deriv:s}

        if resid_timescale > 0:
            resid_corr = resid_coef * exp(-tdata / resid_timescale)
            here_corr += resid_corr
            here_deriv[n_parameters - 3] = exp(-tdata / resid_timescale)
            here_deriv[n_parameters - 2] = resid_corr * tdata / resid_timescale ** 2

        if tdata == 0:
            ec_corr = ec_coef
            here_corr += ec_corr
            here_deriv[n_parameters - 1] = 1

        weighted_fit += pair_count[i] * (here_corr - empirical_correlogram[i]) ** 2
        deriv_common = pair_count[i] * 2 * (here_corr - empirical_correlogram[i])
        for j in range(n_parameters):
            deriv[j] += deriv_common * here_deriv[j]

    deriv[n_parameters - 2] *= DAYS_PER_FORTNIGHT

    return weighted_fit, np.asarray(<floating_type[:n_parameters]>deriv).astype(np.float64)
""".format(
    function_name="_".join([
        "{0:s}{1:s}".format(
            part.get_short_name(),
            form.get_short_name(),
        )
        for part, form in zip(CorrelationPart, forms)
    ]),
    n_parameters=len(get_full_parameter_list(*forms)),
    params_from_parameters="".join([
        "    cdef floating_type {param_name:s} = parameters[{i:d}]\n"
        .format(i=i, param_name=param_name)
        for i, param_name in enumerate(get_full_parameter_list(*forms))
    ]),
    daily_form=forms[0].get_expression(CorrelationPart.DAILY),
    daily_modulation_form=forms[1].get_expression(
        CorrelationPart.DAILY_MODULATION
    ),
    annual_form=forms[2].get_expression(CorrelationPart.ANNUAL),
    accum_day_deriv="\n        ".join(
        "here_deriv[{i:d}] = {deriv_piece:s} * dm_corr".format(
            i=i, deriv_piece=deriv_piece
        )
        for i, deriv_piece in enumerate(
                forms[0].get_derivative(CorrelationPart.DAILY)
        )
    ),
    accum_dm_deriv="\n        ".join(
        "here_deriv[{i:d}] = daily_corr * {deriv_piece:s}".format(
            i=i, deriv_piece=deriv_piece
        )
        for i, deriv_piece in enumerate(
                forms[1].get_derivative(CorrelationPart.DAILY_MODULATION),
                len(forms[0].get_parameters(CorrelationPart.DAILY))
        )
    ),
    accum_ann_deriv="\n        ".join(
        "here_deriv[{i:d}] = {deriv_piece:s}".format(
            i=i, deriv_piece=deriv_piece
        )
        for i, deriv_piece in enumerate(
                forms[2].get_derivative(CorrelationPart.ANNUAL),
                len(forms[0].get_parameters(CorrelationPart.DAILY)) +
                len(forms[1].get_parameters(CorrelationPart.DAILY_MODULATION))
        )
    ),
))

        out_file.write("""
def {function_name:s}_curve_loop(
    floating_type[::1] tdata_base not None,
{parameters:s}
):
    cdef floating_type weighted_fit = 0.0
    cdef long int n_times = len(tdata_base)
    if floating_type == np.float32_t:
        typecode = "f"
    elif floating_type == np.float64_t:
        typecode = "d"
    cdef floating_type[::1] curve = cvarray(
        shape=(n_times,),
        itemsize=sizeof(floating_type),
        format=typecode,
    )
    cdef floating_type[:, ::1] deriv = cvarray(
        shape=(n_times, {n_parameters:d}),
        itemsize=sizeof(floating_type),
        format=typecode,
    )
    cdef long int n_parameters = {n_parameters:d}

    cdef long int i = 0, j = 0

    cdef floating_type tdata
    cdef floating_type daily_corr, dm_corr
    cdef floating_type ann_corr
    cdef floating_type resid_corr, ec_corr
    cdef floating_type here_corr
    cdef floating_type deriv_common

    cdef float_fun exp = cyexp
    cdef float_fun cos = cycos
    cdef float_fun sin = cysin

    resid_timescale *= DAYS_PER_FORTNIGHT

    for i in range(n_times):
        tdata = tdata_base[i]
        here_corr = 0.0

        daily_corr = {daily_form:s}
        dm_corr = {daily_modulation_form:s}
        here_corr += daily_corr * dm_corr
        {accum_day_deriv:s}
        {accum_dm_deriv:s}

        ann_corr = {annual_form:s}
        here_corr += ann_corr
        {accum_ann_deriv:s}

        if resid_timescale > 0:
            resid_corr = resid_coef * exp(-tdata / resid_timescale)
            here_corr += resid_corr
            deriv[i, n_parameters - 3] = exp(-tdata / resid_timescale)
            deriv[i, n_parameters - 2] = resid_corr * tdata / resid_timescale ** 2 * DAYS_PER_FORTNIGHT

        if tdata == 0:
            ec_corr = ec_coef
            here_corr += ec_corr
            deriv[i, n_parameters - 1] = 1

        curve[i] = here_corr

    return (
        np.asarray(curve),
        np.asarray(deriv),
    )
""".format(
    function_name="_".join([
        "{0:s}{1:s}".format(
            part.get_short_name(),
            form.get_short_name(),
        )
        for part, form in zip(CorrelationPart, forms)
    ]),
    n_parameters=len(get_full_parameter_list(*forms)),
    parameters="".join([
        "    floating_type {param_name:s},\n"
        .format(i=i, param_name=param_name)
        for i, param_name in enumerate(get_full_parameter_list(*forms))
    ]),
    daily_form=forms[0].get_expression(CorrelationPart.DAILY),
    daily_modulation_form=forms[1].get_expression(
        CorrelationPart.DAILY_MODULATION
    ),
    annual_form=forms[2].get_expression(CorrelationPart.ANNUAL),
    accum_day_deriv="\n        ".join(
        "deriv[i, {j:d}] = {deriv_piece:s} * dm_corr".format(
            j=j, deriv_piece=deriv_piece
        )
        for j, deriv_piece in enumerate(
                forms[0].get_derivative(CorrelationPart.DAILY)
        )
    ),
    accum_dm_deriv="\n        ".join(
        "deriv[i, {j:d}] = daily_corr * {deriv_piece:s}".format(
            j=j, deriv_piece=deriv_piece
        )
        for j, deriv_piece in enumerate(
                forms[1].get_derivative(CorrelationPart.DAILY_MODULATION),
                len(forms[0].get_parameters(CorrelationPart.DAILY))
        )
    ),
    accum_ann_deriv="\n        ".join(
        "deriv[i, {j:d}] = {deriv_piece:s}".format(
            j=j, deriv_piece=deriv_piece
        )
        for j, deriv_piece in enumerate(
                forms[2].get_derivative(CorrelationPart.ANNUAL),
                len(forms[0].get_parameters(CorrelationPart.DAILY)) +
                len(forms[1].get_parameters(CorrelationPart.DAILY_MODULATION))
        )
    ),
))


############################################################
# Now build the module
setup(
    name="co2_flux_correlation_analysis",
    author="DWesl",
    version="0.0.0.dev1",
    py_modules=["correlation_function_fits", "correlation_utils"],
    ext_modules=cythonize(
        [
            Extension(
                OUT_FILE_NAME.replace(".pyx", ""),
                [OUT_FILE_NAME],
                include_dirs=[np.get_include()],
            ),
        ],
        include_path=[np.get_include()],
        compiler_directives=dict(
            embedsignature=True,
            cdivision=True,
            wraparound=True,
            boundscheck=True,
        ),
        annotate=True,
        gdb_debug=True,
    ),
)
