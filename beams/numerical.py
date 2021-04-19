# -*- coding: utf-8 -*-
"""Beams, Numerical module

Performs numerical operations on beam shapes

"""

# Created on Wed Jul  8 10:28:12 2020

# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com

# Copyright (c) 2020 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)

import numpy as np


# =============================================================================
# FUNCTIONS
# =============================================================================
def calc_OTF_1d(psf, x, f=70e-3, lambda0=488e-9, NFFT=4096):
    '''
    Calculates the Optical Transfer Function from the PSF vector

    OTF is the Fourier transform of the PSF (Intensity), i.e., |E|^2
    or mathematically eqivalent to the autocorrelation of the pupil function

    Parameters
    ----------
    psf : numpy vector array (1d)
        psf intensity.
    x : numpy vector array (1d)
        x-vector, e.g., made by np.linspace().
    f : float, optional
        focal length. The default is 70e-3.
    lambda0 : float, optional
        central wavelength. The default is 488e-9.

    Returns
    -------
    None.

    '''

    # Calc spatial sampling frequency
    pixPerM = 1 / (x[1] - x[0])
    ux = np.linspace(pixPerM / 2, -pixPerM / 2, NFFT)
    ux = ux  # *lambda0*f

    # FT
    OTF = np.fft.fftshift(np.fft.fft(psf, NFFT))
    OTF = OTF / np.max(np.abs(OTF))

    return OTF, ux
