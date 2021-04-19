# -*- coding: utf-8 -*-
"""Test Bessel-Gauss beam generation

Testing different descriptors of the experimental Bessel beam from theory and numerical simulations
"""

# Created on Tue Apr  16 10:59:32 2020

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
def plot_bessel_function():
    r = np.linspace(0, 50e-6, 101)
    kr = 1e6

    E2 = special.j0(kr * r)

    plt.plot(r, E2, 'r', label='Bessel')
    # plt.ylim(-0.5, 1.0)
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()


def plot_bessel_gauss_function():
    r = np.linspace(0, 50e-6, 101)
    kr = 1e6
    wg = 20e-6

    E2 = special.j0(kr * r) * np.exp(-r**2 / wg**2)

    plt.plot(r, E2, 'r', label='Bessel-Gauss')
    # plt.ylim(-0.5, 1.0)
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()


def plot_bg_pupil_full():
    lambda0 = 488e-9
    k = 2 * np.pi / lambda0
    f = 30e-3

    r1 = np.linspace(0, 5e-3, 101)

    kr = 1e6
    wg = 20e-6

    w0 = 2 * f / (k * wg)
    rr = kr * f / k

    E1 = wg / w0 * np.exp(-(rr**2 + r1**2) / w0**2) * special.i0(2 * rr * r1 / w0**2)
    # E1 = np.log(wg/w0*np.exp(-(rr**2+r1**2)/w0**2))
    # E1 = np.log(wg/w0*special.i0(2*rr*r1/w0**2))

    plt.plot(r1, E1, 'r', label='Bessel-Gauss pupil')
    # plt.ylim(-0.5, 1.0)
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()


def plot_bg_propagation(kr=1e6, wg=10e-6):
    # Constants
    lambda0 = 488e-9
    k = 2 * np.pi / lambda0

    A = 1
    kr = 0.5e6
    # kk = k*np.sin(1e-3)
    # print(kk)
    wg = 20e-6

    L = k * wg**2 / 2

    r = np.linspace(-50e-6, 50e-6, 201)
    z = np.linspace(-100e-6, 300e-6, 201)

    # BG propagation
    E2 = np.zeros([r.shape[0], z.shape[0]], dtype=complex)
    for zi, zv in enumerate(z):
        wz = wg * (1 + (zv / L)**2)**(1 / 2)
        Phiz = np.arctan(zv / L)
        Rz = zv + L**2 / zv

        for ri, rv in enumerate(r):

            E2[ri, zi] = ((A * wg / wz) *
                          np.exp(1j * ((k - kr**2 / (2 * k)) * zv - Phiz)) *
                          special.jv(0, kr * rv / (1 + 1j * zv / L)) *
                          np.exp((-1 / wz**2 + 1j * k / (2 * Rz)) * (rv**2 + kr**2 * zv**2 / k**2)))

    fig, ax = plt.subplots()
    ax.imshow(np.abs(E2), extent=[min(z) * 1e6, max(z) * 1e6, min(r) * 1e6, max(r) * 1e6])


def plot_bg_propagation_pupil(rr=2e-3, w0=2e-4, f=30e-3):
    # Constants
    lambda0 = 488e-9
    k = 2 * np.pi / lambda0

    wg = 2 * f / (w0 * k)
    kr = rr * k / f

    A = 1

    L = k * wg**2 / 2

    r = np.linspace(-100e-6, 100e-6, 501)
    z = np.linspace(-800e-6, 800e-6, 1001)

    # BG propagation
    E2 = np.zeros([r.shape[0], z.shape[0]], dtype=complex)
    for zi, zv in enumerate(z):
        wz = wg * (1 + (zv / L)**2)**(1 / 2)
        Phiz = np.arctan(zv / L)
        Rz = zv + L**2 / zv

        for ri, rv in enumerate(r):

            E2[ri, zi] = ((A * wg / wz) *
                          np.exp(1j * ((k - kr**2 / (2 * k)) * zv - Phiz)) *
                          special.jv(0, kr * rv / (1 + 1j * zv / L)) *
                          np.exp((-1 / wz**2 + 1j * k / (2 * Rz)) * (rv**2 + kr**2 * zv**2 / k**2)))

    fig, ax = plt.subplots()
    # ax.imshow(np.abs(E2*np.conj(E2)), extent=[min(z)*1e6, max(z)*1e6, min(r)*1e6, max(r)*1e6])
    ax.imshow(np.abs(E2), extent=[min(z) * 1e6, max(z) * 1e6, min(r) * 1e6, max(r) * 1e6])
    ax.set(xlabel='normalised distance', ylabel='normalised intensity',
           title='w0: %.3f (mm); f: %.1f (mm); rr: %.1f (mm)' % (w0 * 1e3, f * 1e3, rr * 1e3))


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Run this section as a script, while making all defined functions
    # available to the module

    # plot_bessel_function()
    # plot_bessel_gauss_function()
    # plot_bg_pupil_full()
    plot_bg_propagation_pupil()
    #
