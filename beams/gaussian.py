# -*- coding: utf-8 -*-
"""Gaussain module

Generates the Gaussian beam shape based on experimental or theoretical parameters
"""

# Created on Thu Mar  21, 2020

# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com

# Copyright (c) 2020 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# FUNCTIONS
# =============================================================================
def gen_gaussian(r, z, lambda0=488e-9, **kwargs):
    """Generates a Gaussian beam along the r,z coordinates

    E = gen_gaussian(r, z, lambda0, wg=)
    E = gen_gaussian(r, z, lambda0, f=, w0=)

    Example
    -------
    r = np.linspace(-50e-6, 50e-6, 201)
    z = np.linspace(0e-6, 300e-6, 201)
    E = gen_gaussian(r, z, lambda0=488e-9, f=70e-3, w0=5e-4)

    Parameters
    ----------
    x : np array
        Vector of r values (m)
    z : np array
        Vector of z values, relative to the focal plane (m).
    lambda0 : float, optional
        Wavelength (m). The default is 488e-9.
    **kwargs : Keyword options
        Gaussian is specified by either (in the focal plane)
            wg : float
                 Gaussian waist size at focus (1/e^2 radius of intensity)
        OR (in the pupil plane)
            f : float
                focal length
            w0 : float
                 waist size of the Gaussian at the pupil (1/e^2 radius of intensity)

    Returns
    -------
    E : np array, dtype=complex
        2D array (r,z)

    """

    # Constants
    k = 2 * np.pi / lambda0

    if 'wg' in kwargs:
        wg = kwargs['wg']
    elif 'f' in kwargs and 'w0' in kwargs:
        f = kwargs['f']
        w0 = kwargs['w0']
        # Derived focal parameters
        wg = (lambda0 * f) / (w0 * np.pi)
    else:
        print('Not enough input options. Define either wg or f,w0')
        return 0

    # Derived
    n = 1  # (hardcoded for now)
    zr = np.pi * wg**2 * n / lambda0

    # BG propagation
    E = np.zeros([r.shape[0], z.shape[0]], dtype=complex)
    for zi, zv in enumerate(z):
        wz = wg * np.sqrt(1 + (zv / zr)**2)
        psi = np.arctan(zv / zr)
        zrz = zv**2 + zr**2

        for ri, rv in enumerate(r):
            E[ri, zi] = wg / wz * \
                np.exp(-rv**2 / wz**2) * \
                np.exp(-1j * (k * zv + zv * k * rv**2 / (2 * zrz) - psi))

    return E


def gen_gaussian_vect(r, z, lambda0=488e-9, **kwargs):
    """Generates a Gaussian beam at the vector coordinates specified by (x,z)

    E = gen_gaussian(r, z, lambda0, wg=)
    E = gen_gaussian(r, z, lambda0, f=, w0=)

    Example
    -------
    r = np.linspace(-50e-6, 50e-6, 201)
    z = np.linspace(0e-6, 300e-6, 201)
    E = gen_gaussian(r, z, lambda0=488e-9, f=70e-3, w0=5e-4)

    Parameters
    ----------
    x : np array
        Vector of r values (m)
    z : np array
        Vector of z values, relative to the focal plane (m).
    lambda0 : float, optional
        Wavelength (m). The default is 488e-9.
    **kwargs : Keyword options
        Gaussian is specified by either (in the focal plane)
            wg : float
                 Gaussian waist size at focus (1/e^2 radius of intensity)
        OR (in the pupil plane)
            f : float
                focal length
            w0 : float
                 waist size of the Gaussian at the pupil (1/e^2 radius of intensity)

    Returns
    -------
    E : np array, dtype=complex
        2D array (r,z)

    """

    # Constants
    k = 2 * np.pi / lambda0

    if 'wg' in kwargs:
        wg = kwargs['wg']
    elif 'f' in kwargs and 'w0' in kwargs:
        f = kwargs['f']
        w0 = kwargs['w0']
        # Derived focal parameters
        wg = (lambda0 * f) / (w0 * np.pi)
    else:
        print('Not enough input options. Define either wg or f,w0')
        return 0

    # Derived
    n = 1  # (hardcoded for now)
    zr = np.pi * wg**2 * n / lambda0

    # BG propagation
    E = np.zeros(r.shape[0], dtype=complex)
    for q in range(len(r)):
        wz = wg * np.sqrt(1 + (z[q] / zr)**2)
        psi = np.arctan(z[q] / zr)
        zrz = z[q]**2 + zr**2

        E[q] = wg / wz * \
            np.exp(-r[q]**2 / wz**2) * \
            np.exp(-1j * (k * z[q] + z[q] * k * r[q]**2 / (2 * zrz) - psi))

    return E


def plot_gen_gaussian(r, z, lambda0=488e-9, **kwargs):
    """Plots and returns the output of gen_gaussian()

    Usage identical to gen_gaussian()
    """

    E = gen_gaussian(r, z, lambda0=lambda0, **kwargs)

    fig, ax = plt.subplots()
    ax.imshow(np.abs(E)**2,
              extent=[min(z) * 1e6, max(z) * 1e6, min(r) * 1e6, max(r) * 1e6],
              cmap=plt.get_cmap('gray'))

    kwargs_str = ','.join('{}={}'.format(k, v) for k, v in kwargs.items())
    ax.set(xlabel='z (um)', ylabel='r (um)', title=kwargs_str)

    return E


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Run this section as a script, while making all defined functions
    # available to the module

    help(gen_gaussian)

    # focal coordinates
    r = np.linspace(-50e-6, 50e-6, 301)
    z = np.linspace(-100e-6, 300e-6, 301)

    E = plot_gen_gaussian(r, z, lambda0=488e-9, f=70e-3, w0=5e-3)
