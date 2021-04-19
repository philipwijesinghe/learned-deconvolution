# -*- coding: utf-8 -*-
"""Test Gaussian beam generation

Testing the description of the Gaussian beam prepagation
"""

# Created on 2020-05-21

# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com

# Copyright (c) 2020 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# FUNCTIONS
# =============================================================================
def plot_pupil_function():
    u = np.linspace(-10e-3, 10e-3, 2048)

    w0 = 5e-3

    E2 = np.exp(-u**2 / w0**2)

    plt.plot(u, np.abs(E2))


def plot_gaussian():
    # Constants
    lambda0 = 488e-9
    k = 2 * np.pi / lambda0

    # Params
    f = 12e-3
    n = 1
    w0 = 3e-3

    # Derived
    w1 = (lambda0 * f) / (w0 * np.pi)
    zr = np.pi * w1**2 * n / lambda0

    print(w1)

    # focal coordinates
    x = np.linspace(-20e-6, 20e-6, 201)
    z = np.linspace(0e-6, 100e-6, 201)

    # Gaussian propagation
    E1 = np.zeros([x.shape[0], z.shape[0]], dtype=complex)
    E12 = np.zeros([x.shape[0], z.shape[0]], dtype=complex)
    for zi, zv in enumerate(z):
        wz = w1 * np.sqrt(1 + (zv / zr)**2)
        rz = zv * (1 + (zr / zv)**2)
        psi = np.arctan(zv / zr)

        zrz = zv**2 + zr**2

        for xi, xv in enumerate(x):
            E1[xi, zi] = w1 / wz * \
                np.exp(-xv**2 / wz**2) * \
                np.exp(-1j * (k * zv + k * xv**2 / (2 * rz) - psi))

            # Modified function to avoid division by 0
            E12[xi, zi] = w1 / wz * \
                np.exp(-xv**2 / wz**2) * \
                np.exp(-1j * (k * zv + zv * k * xv**2 / (2 * zrz) - psi))

    fig, ax = plt.subplots()
    ax.imshow(np.abs(E12) / np.max(np.abs(E12)), extent=[min(z) * 1e6, max(z) * 1e6, min(x) * 1e6, max(x) * 1e6])
    ax.set(xlabel='z (um)', ylabel='x (um)')


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Run this section as a script, while making all defined functions
    # available to the module

    # plot_pupil_function()
    plot_gaussian()
