# %reset -f

import warnings
import numpy as np
from matplotlib import pyplot as plt  


"""
code modified from that retrieved at 
https://gitlab.tudelft.nl/drama/stereoid/-/blob/a8a31da38369a326a5172d76a51d73bba8bc2d58/stereoid/oceans/cmod5n.py

"""

# Ignore overflow errors for wind calculations over land
warnings.simplefilter("ignore", RuntimeWarning)

def cmod5n_forward(v, phi, theta, CMOD5):
    """!     ---------
    !     cmod5n_forward(v, phi, theta)
    !         inputs:
    !              v     in [m/s] wind velocity (always >= 0)
    !              phi   in [deg] angle between azimuth and wind direction
    !                    (= D - AZM)
    !              theta in [deg] incidence angle
    !         output:
    !              CMOD5_N NORMALIZED BACKSCATTER (LINEAR)
    !
    !        All inputs must be Numpy arrays of equal sizes
    !
    !     A. STOFFELEN              MAY  1991 ECMWF  CMOD4
    !     A. STOFFELEN, S. DE HAAN  DEC  2001 KNMI   CMOD5 PROTOTYPE
    !     H. HERSBACH               JUNE 2002 ECMWF  COMPLETE REVISION
    !     J. de Kloe                JULI 2003 KNMI,  rewritten in fortan90
    !     A. Verhoef                JAN  2008 KNMI,  CMOD5 for neutral winds
    !     K.F.Dagestad              OCT 2011 NERSC,  Vectorized Python version
    !---------------------------------------------------------------------
       """

    thetm = 40.0
    thethr = 25.0
    z_pow = 1.6

    # NB: 0 added as first element below, to avoid switching from
    # 1-indexing to 0-indexing
    C = [              # CMOD 5.N coefficients
        0,
        -0.6878,
        -0.7957,
        0.3380,
        -0.1728,
        0.0000,
        0.0040,
        0.1103,
        0.0159,
        6.7329,
        2.7713,
        -2.2885,
        0.4971,
        -0.7250,
        0.0450,
        0.0066,
        0.3222,
        0.0120,
        22.7000,
        2.0813,
        3.0000,
        8.3659,
        -3.3428,
        1.3236,
        6.2437,
        2.3893,
        0.3249,
        4.1590,
        1.6930,
    ]
    if CMOD5 == True:
        C = [           # CMOD 5 coefficients
              0,
              -0.6880,
              -0.7930,
              0.3380,
              -0.1730,
              0.0000,
              0.0040,
              0.1110,
              0.0162,
              6.3400,
              2.5700,
              -2.1800,
              0.4000,
              -0.6000,
              0.0450,
              0.0070,
              0.3300,
              0.0120,
              22.000,
              1.9500,
              3.0000,
              8.3900,
              -3.4400,
              1.3600,
              5.3500,
              1.9900,
              0.2900,
              3.8000,
              1.5300,
              ]    
    
    
    y0 = C[19]
    PN = C[20]
    a = C[19] - (C[19] - 1) / C[20]
    b = 1.0 / (C[20] * (C[19] - 1.0) ** (3 - 1))

    #  !  ANGLES
    fi = np.radians(phi)
    csfi = np.cos(fi)
    cs2_fi = 2.00 * csfi ** 2 - 1.00

    x = (theta - thetm) / thethr
    xx = x ** 2

    #  ! B0: FUNCTION OF WIND SPEED AND INCIDENCE ANGLE
    a0 = C[1] + C[2] * x + C[3] * xx + C[4] * x * xx
    a1 = C[5] + C[6] * x
    a2 = C[7] + C[8] * x

    GAM = C[9] + C[10] * x + C[11] * xx
    s0 = C[12] + C[13] * x

    # V is missing! Using V=v as substitute, this is apparently correct
    # V = v
    s = a2 * v
    # S_vec = s.copy()
    s_vec = np.where(s < s0, s0, s)
    # SlS0 = [S_vec < S0]
    # S_vec[SlS0] = S0[SlS0]
    a3 = 1.0 / (1.0 + np.exp(-s_vec))

    SlS0 = s < s0
    a3[SlS0] = a3[SlS0] * (s[SlS0] / s0[SlS0]) ** (s0[SlS0] * (1.0 - a3[SlS0]))
    # A3=A3*(S/S0)**( S0*(1.- A3))
    B0 = (a3 ** GAM) * 10.0 ** (a0 + a1 * v)

    #  !  B1: FUNCTION OF WIND SPEED AND INCIDENCE ANGLE
    B1 = C[15] * v * (0.5 + x - np.tanh(4.0 * (x + C[16] + C[17] * v)))
    B1 = C[14] * (1.0 + x) - B1
    B1 = B1 / (np.exp(0.34 * (v - C[18])) + 1.0)

    #  !  B2: FUNCTION OF WIND SPEED AND INCIDENCE ANGLE
    V0 = C[21] + C[22] * x + C[23] * xx
    D1 = C[24] + C[25] * x + C[26] * xx
    D2 = C[27] + C[28] * x

    V2 = v / V0 + 1.0
    V2ltY0 = V2 < y0
    V2[V2ltY0] = a + b * (V2[V2ltY0] - 1.0) ** PN
    B2 = (-D1 + D2 * V2) * np.exp(-V2)

    #  !  CMOD5_N: COMBINE THE THREE FOURIER TERMS
    cmod5_n = B0 * (1.0 + B1 * csfi + B2 * cs2_fi) ** z_pow
    return cmod5_n


def cmod5n_inverse(sigma0_obs, phi, incidence, CMOD5, iterations=10):
    """!     ---------
    !     cmod5n_inverse(sigma0_obs, phi, incidence, iterations)
    !         inputs:
    !              sigma0_obs     Normalized Radar Cross Section [linear units]
    !              phi   in [deg] angle between azimuth and wind direction
    !                    (= D - AZM)
    !              incidence in [deg] incidence angle
    !              iterations: number of iterations to run
    !         output:
    !              Wind speed, 10 m, neutral stratification
    !
    !        All inputs must be Numpy arrays of equal sizes
    !
    !    This function iterates the forward CMOD5N function
    !    until agreement with input (observed) sigma0 values
    !---------------------------------------------------------------------
       """
    from numpy import ones, array

    # First guess wind speed
    V = array([10.0]) * ones(sigma0_obs.shape)
    step = 10.0

    # Iterating until error is smaller than threshold
    for iterno in range(1, iterations):
        # print(iterno)
        sigma0_calc = cmod5n_forward(V, phi, incidence, CMOD5)
        ind = sigma0_calc - sigma0_obs > 0
        V = V + step
        V[ind] = V[ind] - 2 * step
        step = step / 2


    return V




