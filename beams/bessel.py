# -*- coding: utf-8 -*-
"""Bessel module

Generates the Bessel-Gauss beam shape based on experimental or theoretical parameters
"""

# Created on Tue Apr  16, 2020

# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com

# Copyright (c) 2020 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)

import numpy as np
import matplotlib.pyplot as plt

from scipy import special


# =============================================================================
# FUNCTIONS
# =============================================================================
def gen_bessel(r, z, lambda0=488e-9, **kwargs):
    """Generates a Bessel-Gauss beam along the r,z coordinates

    E = gen_bessel(r, z, lambda0, wg=, kr=)
    E = gen_bessel(r, z, lambda0, f=, rr=, w0=)

    Example
    -------
    r = np.linspace(-50e-6, 50e-6, 201)
    z = np.linspace(0e-6, 300e-6, 201)
    E = gen_bessel(r, z, lambda0=488e-9, f=70e-3, rr=5e-3, w0=5e-4)

    Parameters
    ----------
    x : np array
        Vector of r values (m)
    z : np array
        Vector of z values, relative to the focal plane (m).
    lambda0 : float, optional
        Wavelength (m). The default is 488e-9.
    **kwargs : Keyword options
        Bessel-Gauss is specified by either (in the focal plane)
            wg : float
                 Gaussian envelope waist size
            kr : float
                 radial scaling vector of the Bessel function
        OR (in the pupil plane)
            f : float
                focal length
            rr : float
                 ring radius
            w0 : float
                 waist size of the ring

    Returns
    -------
    E : np array, dtype=complex
        2D array (r,z)

    """

    # Constants
    k = 2 * np.pi / lambda0

    if 'wg' in kwargs and 'kr' in kwargs:
        kr = kwargs['kr']
        wg = kwargs['wg']
    elif 'f' in kwargs and 'rr' in kwargs and 'w0' in kwargs:
        f = kwargs['f']
        rr = kwargs['rr']
        w0 = kwargs['w0']
        # Derived focal parameters
        wg = 2 * f / (w0 * k)
        kr = rr * k / f
    else:
        print('Not enough input options. Define either wg,kr or f,rr,w0')
        return 0

    # Derived parameters
    L = k * wg**2 / 2

    # BG propagation
    E2 = np.zeros([r.shape[0], z.shape[0]], dtype=complex)
    for zi, zv in enumerate(z):
        wz = wg * (1 + (zv / L)**2)**(1 / 2)
        Phiz = np.arctan(zv / L)
        # Rz = zv + L**2/zv
        # We substitute R(z) with z*R(z) to avoid division by 0 and numerical error
        zRz = zv**2 + L**2

        for ri, rv in enumerate(r):
            E2[ri, zi] = ((wg / wz) *
                          np.exp(1j * ((k - kr**2 / (2 * k)) * zv - Phiz)) *
                          special.jv(0, kr * rv / (1 + 1j * zv / L)) *
                          np.exp((-1 / wz**2 + 1j * k * zv / (2 * zRz)) * (rv**2 + kr**2 * zv**2 / k**2)))

    return E2


def gen_bessel_vect(r, z, lambda0=488e-9, **kwargs):
    """Generates a Bessel-Gauss beam at the vector coordinates specified by (x,z)

    E = gen_bessel(r, z, lambda0, wg=, kr=)
    E = gen_bessel(r, z, lambda0, f=, rr=, w0=)

    Example
    -------
    r = np.linspace(-50e-6, 50e-6, 201)
    z = np.linspace(0e-6, 300e-6, 201)
    E = gen_bessel(r, z, lambda0=488e-9, f=70e-3, rr=5e-3, w0=5e-4)

    Parameters
    ----------
    x : np array
        Vector of r values (m)
    z : np array
        Vector of z values, relative to the focal plane (m).
    lambda0 : float, optional
        Wavelength (m). The default is 488e-9.
    **kwargs : Keyword options
        Bessel-Gauss is specified by either (in the focal plane)
            wg : float
                 Gaussian envelope waist size
            kr : float
                 radial scaling vector of the Bessel function
        OR (in the pupil plane)
            f : float
                focal length
            rr : float
                 ring radius
            w0 : float
                 waist size of the ring

    Returns
    -------
    E : np array, dtype=complex
        2D array (r,z)

    """

    # Constants
    k = 2 * np.pi / lambda0

    if 'wg' in kwargs and 'kr' in kwargs:
        kr = kwargs['kr']
        wg = kwargs['wg']
    elif 'f' in kwargs and 'rr' in kwargs and 'w0' in kwargs:
        f = kwargs['f']
        rr = kwargs['rr']
        w0 = kwargs['w0']
        # Derived focal parameters
        wg = 2 * f / (w0 * k)
        kr = rr * k / f
    else:
        print('Not enough input options. Define either wg,ke or f,rr,w0')
        return 0

    # Derived parameters
    L = k * wg**2 / 2

    # BG propagation
    E2 = np.zeros([r.shape[0], z.shape[0]], dtype=complex)
    for q in range(len(r)):
        wz = wg * (1 + (z[q] / L)**2)**(1 / 2)
        Phiz = np.arctan(z[q] / L)
        # Rz = zv + L**2/zv
        # We substitute R(z) with z*R(z) to avoid division by 0 and numerical error
        zRz = z[q]**2 + L**2

        E2[q] = ((wg / wz) *
                 np.exp(1j * ((k - kr**2 / (2 * k)) * z[q] - Phiz)) *
                 special.jv(0, kr * r[q] / (1 + 1j * z[q] / L)) *
                 np.exp((-1 / wz**2 + 1j * k * z[q] / (2 * zRz)) * (r[q]**2 + kr**2 * z[q]**2 / k**2)))

    return E2


def plot_gen_bessel(r, z, lambda0=488e-9, **kwargs):
    """Plots and returns the output of gen_bessel()

    Usage identical to gen_bessel()
    """

    E = gen_bessel(r, z, lambda0=lambda0, **kwargs)

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

    help(gen_bessel)

    # focal coordinates
    r = np.linspace(-50e-6, 50e-6, 301)
    z = np.linspace(-100e-6, 300e-6, 301)

    E = plot_gen_bessel(r, z, lambda0=488e-9, f=70e-3, rr=5e-3, w0=5e-4)
