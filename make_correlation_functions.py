"""Write all possible correlation functions given parts."""
import itertools

import numpy as np
import numexpr as ne
import bottleneck as bn

HOURS_PER_DAY = 24.
DAYS_PER_DAY = 1.
DAYS_PER_WEEK = 7.
DAYS_PER_YEAR = 365.2425

PI_OVER_DAY = np.pi / DAYS_PER_DAY
TWO_PI_OVER_DAY = 2 * PI_OVER_DAY
FOUR_PI_OVER_DAY = 2 * TWO_PI_OVER_DAY

PI_OVER_YEAR = np.pi / DAYS_PER_YEAR
TWO_PI_OVER_YEAR = 2 * PI_OVER_YEAR
FOUR_PI_OVER_YEAR = 2 * TWO_PI_OVER_YEAR

############################################################
# Forms each part of the correlation function can take
N_CORRELATION_FUNCTIONS = 3
SHORT_NAMES = ("0", "c", "p")
NAMES = ("None", "Cosine", "Periodic")
LONG_NAMES = ("None", "3-term Cosine series", "Exponential of negative sine-squared")

FORM_STRING = (
    "0",
    "((1 - {part:s}_coef1 - {part:s}_coef2) +"
    " {part:s}_coef1 * cos(TWO_PI_OVER_{time:s} * tdata) +"
    " {part:s}_coef2 * cos(FOUR_PI_OVER_{time:s} * tdata))",
    "exp(-(sin(PI_OVER_{time:s} * tdata) / {part:s}_width) ** 2)",
)

COEFFICIENTS = (
    "",
    "floating_type {part:s}_coef1, floating_type {part:s}_coef2",
    "floating_type {part:s}_width",
)

############################################################
# Parts of the correlation function
PARTS = (
    "daily_cycle",
    "annual_modulation_of_daily_cycle",
    "annual_cycle",
)
SHORT_PARTS = ("d", "dm", "a")
PARTS_FOR_VAR = ("daily", "dm", "ann")

PYTHON_FUNCTION_SCRIPT = """
def {function_name:s}(
    tdata,
    daily_coef, ann_coef,
    {parameters:s}
    Td, Ta,
    resid_coef, To,
    ec_coef, Tec,
):
    result = np.empty_like(tdata)
    Tec /= HOURS_PER_DAY
    Td *= DAYS_PER_WEEK
    Ta *= DAYS_PER_YEAR
    To *= DAYS_PER_WEEK
    exp = np.exp
    cos = np.cos
    sin = np.sin
    result[:] = daily_coef * (
        {daily_form:s} *
        {daily_modulation_form:s}
    ) * exp(-tdata / Td)
    result += ann_coef * {annual_form:s} * exp(-tdata / Ta)
    result += resid_coef * exp(-tdata / To)
    result += ec_coef * exp(-tdata / Tec)
    return result
"""

NUMPY_FUNCTION_SCRIPT = """
cpdef np.ndarray[floating_type, ndim=1] {function_name:s}(
    np.ndarray[floating_type, ndim=1] tdata,
    floating_type daily_coef, floating_type ann_coef,
    {parameters:s}
    floating_type Td, floating_type Ta,
    floating_type resid_coef, floating_type To,
    floating_type ec_coef, floating_type Tec,
):
    cdef np.ndarray[floating_type, ndim=1] result = np.empty_like(tdata)
    Tec /= HOURS_PER_DAY
    Td *= DAYS_PER_WEEK
    Ta *= DAYS_PER_YEAR
    To *= DAYS_PER_WEEK
    exp = np.exp
    cos = np.cos
    sin = np.sin
    result[:] = daily_coef * (
        {daily_form:s} *
        {daily_modulation_form:s}
    ) * exp(-tdata / Td)
    result += ann_coef * {annual_form:s} * exp(-tdata / Ta)
    result += resid_coef * exp(-tdata / To)
    result += ec_coef * exp(-tdata / Tec)
    return result
"""

NONUMPY_FUNCTION_SCRIPT = """
cpdef floating_type[::1] {function_name:s}(
    floating_type[::1] tdata_base,
    floating_type daily_coef,
    floating_type ann_coef,
    {parameters:s}
    floating_type Td, floating_type Ta,
    floating_type resid_coef, floating_type To,
    floating_type ec_coef, floating_type Tec,
):
    cdef floating_type[::1] result = np.empty_like(tdata_base)
    cdef long int i = 0
    Tec /= HOURS_PER_DAY
    Td *= DAYS_PER_WEEK
    Ta *= DAYS_PER_YEAR
    To *= DAYS_PER_WEEK
    cdef float_fun exp = cyexp
    cdef float_fun cos = cycos
    cdef float_fun sin = cysin
    for i in range(len(result)):
        tdata = tdata_base[i]
        result[i] = daily_coef * (
            {daily_form:s} *
            {daily_modulation_form:s}
        ) * exp(-tdata / Td)
        result[i] += ann_coef * {annual_form:s} * exp(-tdata / Ta)
        result[i] += resid_coef * exp(-tdata / To)
        result[i] += ec_coef * exp(-tdata / Tec)
    return result
"""

NUMEXPR_EXPRESSION_SCRIPT = (
    "daily_coef * ({daily_form:s} * {daily_modulation_form:s}) * exp(-tdata / (Td * DAYS_PER_WEEK)) + "
    "ann_coef * {annual_form:s} * exp(-tdata / (Ta * DAYS_PER_YEAR)) + "
    "resid_coef * exp(-tdata / (To * DAYS_PER_WEEK)) + "
    "ec_coef * exp(-tdata / Tec * HOURS_PER_DAY)"
)
NUMEXPR_FUNCTION_SCRIPT = """
def {function_name:s}(
    tdata,
    daily_coef,
    ann_coef,
    {parameters:s}
    Td, Ta,
    resid_coef, To,
    ec_coef, Tec,
):
    return numexpr.evaluate(
        "{expression:s}",
        local_dict={{
            "tdata": tdata,
            "daily_coef": daily_coef,
            "ann_coef": ann_coef,
            {parameter_dict:s}
            "Td": Td,
            "Ta": Ta,
            "resid_coef": resid_coef,
            "To": To,
            "ec_coef": ec_coef,
            "Tec": Tec,
        }},
        global_dict=GLOBAL_DICT
    )
"""

# Weighted least squares, all within numexpr
NUMEXPR_WEIGHTING_SCRIPT = "sum(pair_count * ({function:s} - corr_data) ** 2)"

OUT_FILE_NAME = "flux_correlation_functions.pyx"
with open(OUT_FILE_NAME, "w") as out_file, \
     open(OUT_FILE_NAME.replace(".pyx", "_py.py"), "w") as out_file_py:
    out_file.write("""# cython: embedsignature=True
# cython: language_level=3str
from libc cimport math

import numpy as np
cimport numpy as np
import numexpr 

cdef extern from "<math.h>" nogil:
    float sinf(float x)
    float cosf(float x)
    float expf(float x)

ctypedef fused floating_type:
    np.float32_t
    np.float64_t

cdef floating_type cycos(floating_type x):
    if floating_type is float:
        return cosf(x)
    elif floating_type is double:
        return math.cos(x)

ctypedef floating_type (*float_fun)(floating_type)

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
    out_file_py.write("""
import numpy as np
import numexpr 

HOURS_PER_DAY = 24.
DAYS_PER_DAY = 1.
DAYS_PER_WEEK = 7.
DAYS_PER_YEAR = 365.2425

HOURS_PER_YEAR = HOURS_PER_DAY * DAYS_PER_YEAR

PI_OVER_DAY = np.pi / DAYS_PER_DAY
TWO_PI_OVER_DAY = 2 * PI_OVER_DAY
FOUR_PI_OVER_DAY = 2 * TWO_PI_OVER_DAY

PI_OVER_YEAR = np.pi / DAYS_PER_YEAR
TWO_PI_OVER_YEAR = 2 * PI_OVER_YEAR
FOUR_PI_OVER_YEAR = 2 * TWO_PI_OVER_YEAR

GLOBAL_DICT = {
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
    for d_fn_i, dm_fn_i, ann_fn_i in itertools.product(
            range(N_CORRELATION_FUNCTIONS),
            range(N_CORRELATION_FUNCTIONS),
            range(N_CORRELATION_FUNCTIONS),
    ):
        if d_fn_i == 0 and dm_fn_i != 0:
            # Daily modulation without daily cycle is silly
            continue
        function_name = "d{daily:s}_dm{dm:s}_a{ann:s}".format(
            daily=SHORT_NAMES[d_fn_i],
            dm=SHORT_NAMES[dm_fn_i],
            ann=SHORT_NAMES[ann_fn_i],
        )
        indices = (d_fn_i, dm_fn_i, ann_fn_i)
        parameter_list = [
            COEFFICIENTS[indices[i]].format(part=PARTS_FOR_VAR[i])
            for i in range(len(PARTS_FOR_VAR))
        ]
        parameters = ", ".join(
            [params for params in parameter_list
             if params != ""]
        )
        if parameters != "":
            parameters += ","
        numpy_function = NUMPY_FUNCTION_SCRIPT.format(
            function_name=function_name + "_numpy",
            parameters=parameters,
            daily_form=FORM_STRING[d_fn_i].format(
                part="daily", time="DAY"
            ),
            daily_modulation_form=(
                # No modulation is times one, not times zero
                FORM_STRING[dm_fn_i] if dm_fn_i != 0 else "1"
            ).format(part="dm", time="YEAR"),
            annual_form=FORM_STRING[ann_fn_i].format(
                part="ann", time="YEAR"
            ),
        )
        print(numpy_function, file=out_file)
        nonumpy_function = NONUMPY_FUNCTION_SCRIPT.format(
            function_name=function_name + "_loop",
            parameters=parameters,
            daily_form=FORM_STRING[d_fn_i].format(
                part="daily", time="DAY"
            ),
            daily_modulation_form=(
                # No modulation is times one, not times zero
                FORM_STRING[dm_fn_i] if dm_fn_i != 0 else "1"
            ).format(part="dm", time="YEAR"),
            annual_form=FORM_STRING[ann_fn_i].format(
                part="ann", time="YEAR"
            ),
        )
        print(nonumpy_function, file=out_file)
        numexpr_expr = NUMEXPR_EXPRESSION_SCRIPT.format(
            function_name=function_name + "_numexpr",
            parameters=parameters,
            daily_form=FORM_STRING[d_fn_i].format(
                part="daily", time="DAY"
            ),
            daily_modulation_form=(
                # No modulation is times one, not times zero
                FORM_STRING[dm_fn_i] if dm_fn_i != 0 else "1"
            ).format(part="dm", time="YEAR"),
            annual_form=FORM_STRING[ann_fn_i].format(
                part="ann", time="YEAR"
            ),
        )
        print("".join([function_name, '_numexpr_expr = "', numexpr_expr, '"']),
              file=out_file)
        numexpr_fn = NUMEXPR_FUNCTION_SCRIPT.format(
            function_name=function_name + "_numexpr_fn",
            parameters=parameters.replace("floating_type ", ""),
            # Turn the parameters into a "key": value list
            parameter_dict="".join(
                '            '
                '"{param:s}": {param:s},\n'.format(param=param.strip(","))
                for param in parameters.replace(
                        "floating_type ", ""
                ).split(", ")
                if param
            ),
            expression=numexpr_expr,
        )
        print(numexpr_fn, file=out_file)
        print(numexpr_fn, file=out_file_py)
        python_function = PYTHON_FUNCTION_SCRIPT.format(
            function_name=function_name + "_python",
            parameters=parameters.replace("floating_type ", ""),
            daily_form=FORM_STRING[d_fn_i].format(
                part="daily", time="DAY"
            ),
            daily_modulation_form=(
                # No modulation is times one, not times zero
                FORM_STRING[dm_fn_i] if dm_fn_i != 0 else "1"
            ).format(part="dm", time="YEAR"),
            annual_form=FORM_STRING[ann_fn_i].format(
                part="ann", time="YEAR"
            ),
        )
        print(python_function, file=out_file)
        print(python_function, file=out_file_py)


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
