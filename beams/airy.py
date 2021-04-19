# -*- coding: utf-8 -*-
"""Airy module

Generates the Airy beam shape based on experimental or theoretical parameters
"""

# Created on Tue Apr  7 10:01:32 2020

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
def gen_airy(x, z, lambda0=488e-9, **kwargs):
    """Generates an Airy beam along the x,z coordinates

    Phi = gen_airy(x, z, lambda0, x0=, alpha=)
    Phi = gen_airy(x, z, lambda0, f=, beta=, w0=)

    Example
    -------
    x = np.linspace(-50e-6, 50e-6, 201)
    z = np.linspace(0e-6, 300e-6, 201)
    Phi = gen_airy(x, z, lambda0=488e-9, f=70e-3, beta=7, w0=5e-3)

    Parameters
    ----------
    x : np array
        Vector of x values (m)
    z : np array
        Vector of z values, relative to the focal plane (m).
    lambda0 : float, optional
        Wavelength (m). The default is 488e-9.
    **kwargs : Keyword options
        Airy is specified by either (in the focal plane)
            x0 : float
                 scaling factor
            alpha : float
                    alpha value (exponential apodisation)
        OR (in the pupil plane)
            f : float
                focal length
            w0 : float
                 beam waist at pupil
            (beta OR gamma) : float
                              beta, cubic modulation
                              gamma, w0 scaled cubic modulation

    Returns
    -------
    Phi : np array, dtype=complex
          2D array (x,z)
    """

    if 'x0' in kwargs and 'alpha' in kwargs:
        x0 = kwargs['x0']
        alpha = kwargs['alpha']
    elif 'f' in kwargs and 'beta' in kwargs and 'w0' in kwargs:
        f = kwargs['f']
        beta = kwargs['beta']
        w0 = kwargs['w0']
        # Derived focal parameters
        beta_p = beta / lambda0
        alpha = w0**(-2) * (3 * beta_p)**(-2 / 3)
        x0 = lambda0 * f * (3 * beta_p)**(1 / 3) / (2 * np.pi)
    elif 'f' in kwargs and 'gamma' in kwargs and 'w0' in kwargs:
        f = kwargs['f']
        w0 = kwargs['w0']
        gamma = kwargs['gamma']
        # Derived focal parameters
        beta_p = 2 * np.pi * gamma / (w0 * (np.log(2))**(1 / 2))**3
        alpha = w0**(-2) * (3 * beta_p)**(-2 / 3)
        x0 = lambda0 * f * (3 * beta_p)**(1 / 3) / (2 * np.pi)
    else:
        print('Not enough input options. Define either x0,alpha or f,beta,w0')
        return 0

    # Constants
    k = 2 * np.pi / lambda0

    # Output scaling factor
    s = -x / x0
    xi = z / (k * x0**2)

    # Airy propagation
    Phi = np.zeros([x.shape[0], z.shape[0]], dtype=complex)
    for u in range(len(s)):
        for v in range(len(xi)):
            ai = special.airy(s[u] - (xi[v] / 2)**2 + 1j * alpha * xi[v])
            exponent = np.exp(alpha * s[u] - alpha * xi[v]**2 / 2 -
                              1j * xi[v]**3 / 12 + 1j * alpha**2 * xi[v] / 2 +
                              1j * s[u] * xi[v] / 2)
            Phi[u, v] = ai[0] * exponent

    return Phi


def gen_airy_vect(x, z, lambda0=488e-9, **kwargs):
    """Generates an Airy beam at the vector coordinates specified by (x,z)

    Phi = gen_airy(x, z, lambda0, x0=, alpha=)
    Phi = gen_airy(x, z, lambda0, f=, beta=, w0=)

    Example
    -------
    x = np.linspace(-50e-6, 50e-6, 201)
    z = np.linspace(0e-6, 300e-6, 201)
    xv, zv = np.meshgrid(x, z, sparse=False, indexing='ij')

    Phi = gen_airy(xv.flatten, zv.flatten, lambda0=488e-9, f=70e-3, beta=7, w0=5e-3)

    Parameters
    ----------
    x : np array
        Vector of x coordinates
    z : np array
        Vector of z coordinates, corresponding to x (x.shape = z.shape)
    lambda0 : float, optional
        Wavelength (m). The default is 488e-9.
    **kwargs : Keyword options
        Airy is specified by either (in the focal plane)
            x0 : float
                 scaling factor
            alpha : float
                    alpha value (exponential apodisation)
        OR (in the pupil plane)
            f : float
                focal length
            w0 : float
                 beam waist at pupil
            (beta OR gamma) : float
                              beta, cubic modulation
                              gamma, w0 scaled cubic modulation

    Returns
    -------
    Phi : np array, dtype=complex
          2D array (x,z)
    """

    if 'x0' in kwargs and 'alpha' in kwargs:
        x0 = kwargs['x0']
        alpha = kwargs['alpha']
    elif 'f' in kwargs and 'beta' in kwargs and 'w0' in kwargs:
        f = kwargs['f']
        beta = kwargs['beta']
        w0 = kwargs['w0']
        # Derived focal parameters
        beta_p = beta / lambda0
        alpha = w0**(-2) * (3 * beta_p)**(-2 / 3)
        x0 = lambda0 * f * (3 * beta_p)**(1 / 3) / (2 * np.pi)
    elif 'f' in kwargs and 'gamma' in kwargs and 'w0' in kwargs:
        f = kwargs['f']
        w0 = kwargs['w0']
        gamma = kwargs['gamma']
        # Derived focal parameters
        beta_p = 2 * np.pi * gamma / (w0 * (np.log(2))**(1 / 2))**3
        alpha = w0**(-2) * (3 * beta_p)**(-2 / 3)
        x0 = lambda0 * f * (3 * beta_p)**(1 / 3) / (2 * np.pi)
    else:
        print('Not enough input options. Define either x0,alpha or f,beta,w0')
        return 0

    # Constants
    k = 2 * np.pi / lambda0

    # Output scaling factor
    s = -x / x0
    chi = z / (k * x0**2)

    # Airy propagation
    E = np.zeros(x.shape[0], dtype=complex)
    for q in range(len(x)):
        ai = special.airy(s[q] - (chi[q] / 2)**2 + 1j * alpha * chi[q])
        exponent = np.exp(alpha * s[q] - alpha * chi[q]**2 / 2 -
                          1j * chi[q]**3 / 12 + 1j * alpha**2 * chi[q] / 2 +
                          1j * s[q] * chi[q] / 2)
        E[q] = ai[0] * exponent

    return E


def plot_gen_airy(x, z, lambda0=488e-9, **kwargs):
    """Plots and returns the output of gen_airy()

    Usage identical to gen_airy()
    """

    Phi = gen_airy(x, z, lambda0=lambda0, **kwargs)

    fig, ax = plt.subplots()
    ax.imshow(np.abs(Phi)**2,
              extent=[min(z) * 1e6, max(z) * 1e6, min(x) * 1e6, max(x) * 1e6],
              cmap=plt.get_cmap('gray'))

    kwargs_str = ','.join('{}={}'.format(k, v) for k, v in kwargs.items())
    ax.set(xlabel='z (um)', ylabel='x (um)', title=kwargs_str)

    return Phi


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Run this section as a script, while making all defined functions
    # available to the module

    help(gen_airy)

    # focal coordinates
    x = np.linspace(-50e-6, 50e-6, 301)
    z = np.linspace(-100e-6, 300e-6, 301)

    Phi = plot_gen_airy(x, z, lambda0=488e-9, f=70e-3, beta=10, w0=8e-3)
