# co2_flux_error_correlation_analysis
Codes for calculation of temporal correlations in model-data differences, creating and fitting mathematical models, and cross-validating the fits.

# Functions
The functions of the mathematical models are of the following form:
```math
corr_{ijk}(\Delta t) = C_d d_i(\Delta t) dm_j(\Delta t) \exp(\Delta t/T_d) + C_a a_k(\Delta t) \exp(\Delta t/T_a) + C_o \exp(\Delta t/T_o) + C_{ec} \exp(\Delta t/T_{ec})
```
$`d_i`$, $`dm_j`$, and $`a_k`$ are periodic functions with periods of one day, one year, and one year, respectively.  
They may take on one of the functional forms in the following section.
$`d_i`$ models the daily cycle, $`dm_j`$ models the annual modulation of the amplitude of the daily cycle, and $`a_k`$ models the annual cycle.

The last term is there to account for possibly-correlated instrumental noise, expected to vanish within a few hours.

The second-to-last term is there to model any correlations not accounted for by the other terms.  
It seems to play the role of constant offset in many cases.

## Functional forms
### Nothing
Baseline, used to explore behavior when that part of the function is absent.
Identically zero for the daily and annual cycles, and identically one for the annual modulation of the daily cycle.

### Decoupled
Used in CarbonTracker-Lagrange, which has a three-hourly time step with correlations from one day to the next but none from any period within a day to another within the same day.
I want to use hourly data, so I linearly interpolate between the three-hourly values that gives.
```math
f_d(\Delta t) = \begin{cases}
  1 - 8 \frac{\Delta t}{\tau} & \text{if } \frac{\Delta t}{\tau} \bmod 1 < \frac{1}{8} \\
  8 \frac{\Delta t}{\tau} - 7 & \text{if }\frac{\Delta t}{\tau} \bmod 1 > \frac{7}{8} \\
  0 & \text{if } \frac{1}{8} \le \left(\frac{\Delta t}{\tau} \bmod 1 \right) \le \frac{7}{8}
\end{cases}
```
### Exponential sine-squared
This form was inspired by the "Periodic" kernel in scikit-learn.
```math
f_p(\Delta t) = \exp(-\sin^2(\frac{\Delta t}{\tau})/w^2)
```
This form has one parameter, $`w`$.

### Cosines
A three-term cosine series, re-parameterized to always be one at zero:
```math
f_c(\Delta t) = 1 - b_1 - b_2 + b_1 \cos(\frac{2\pi\Delta t}{\tau}) + b_2 \cos(\frac{4\pi\Delta t}{\tau})
```
This form has two parameters, $`b_1`$ and $`b_2`$.

# Implementation
The installation process generates the code for a Cython extension names `flux_correlation_function_fits`, with function names like
`dc_dmp_ad_fit_ne` or `d0_dm0_ac_curve_loop`.  
The `fit` or `curve` just before the end of the name describe whether the function returns a correlogram for the given times (`curve`), or the fit of the correlogram to a provided correlogram.
The `ne` and `loop` at the end describe whether the function is entirely in Cython, or passes the actual work to numexpr.  
Numexpr tends to be faster, but the Cython version also returns the derivative of the curve with respect to the various parameters.
The first three groups of letters describe which functional form is used to describe the daily cycle, annual modulation of the daily cycle, and annual cycle.
For each part, `0` denotes the "Nothing" form, `d` for "Decoupled", `p` for the exponential of a squared sine, and `c` for the three-term cosine series.
