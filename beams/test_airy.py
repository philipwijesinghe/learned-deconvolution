# -*- coding: utf-8 -*-
"""Test Airy beam generation

Testing different descriptors of the Airy beam from theory and numerical simulations
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
def plot_airy_function():
    x = np.linspace(-15, 5, 201)
    ai, aip, bi, bip = special.airy(x)

    plt.plot(x, ai, 'r', label='Ai(x)')
    plt.plot(x, bi, 'b--', label='Bi(x)')
    plt.ylim(-0.5, 1.0)
    plt.grid()
    plt.legend(loc='upper left')
    plt.show()


def plot_pupil_function(beta=5):
    u = np.linspace(-10e-3, 10e-3, 2048)

    beta_p = beta / 488e-9

    P = np.exp(2 * np.pi * 1j * beta_p * u**3)

    plt.plot(np.angle(P))


def plot_airy(alpha=0.1):
    # Constants
    lambda0 = 488e-9
    k = 2 * np.pi / lambda0

    # Derived
    w1 = 1e-6

    # focal coordinates
    x = np.linspace(-50e-6, 50e-6, 201)
    z = np.linspace(0e-6, 200e-6, 201)

    # Output scaling factor
    s = x / w1
    eta = z / (k * w1**2)

    # Airy propagation
    Phi = np.zeros([x.shape[0], z.shape[0]])
    for xi, xv in enumerate(x):
        for zi, zv in enumerate(z):
            ai = special.airy(s[xi] - (eta[zi] / 2)**2 + 1j * alpha * eta[zi])
            exponent = np.exp(alpha * s[xi] - alpha * eta[zi]**2 / 2 - 1j * eta[zi]**3 / 12 + 1j * alpha**2 * eta[zi] / 2 + 1j * s[xi] * eta[zi] / 2)
            Phi[xi, zi] = np.abs(ai[0] * exponent)

    fig, ax = plt.subplots()
    ax.imshow(Phi, extent=[min(z) * 1e6, max(z) * 1e6, min(x) * 1e6, max(x) * 1e6])


def plot_airy_cubic(beta=8, f=70e-3, w0=7e-3):
    # Constants
    lambda0 = 488e-9
    k = 2 * np.pi / lambda0

    # Derived
    beta_p = beta / lambda0
    alpha = w0**(-2) * (3 * beta_p)**(-2 / 3)
    w1 = lambda0 * f * (3 * beta_p)**(1 / 3) / (2 * np.pi)

    # focal coordinates
    x = np.linspace(-20e-6, 20e-6, 201)
    z = np.linspace(0e-6, 100e-6, 201)

    # Output scaling factor
    s = x / w1
    eta = z / (k * w1**2)

    # Airy propagation
    Phi = np.zeros([x.shape[0], z.shape[0]])
    for xi, xv in enumerate(x):
        for zi, zv in enumerate(z):
            ai = special.airy(s[xi] - (eta[zi] / 2)**2 + 1j * alpha * eta[zi])
            exponent = np.exp(alpha * s[xi] - alpha * eta[zi]**2 / 2 - 1j * eta[zi]**3 / 12 + 1j * alpha**2 * eta[zi] / 2 + 1j * s[xi] * eta[zi] / 2)
            Phi[xi, zi] = np.abs(ai[0] * exponent)

    fig, ax = plt.subplots()
    ax.imshow(Phi, extent=[min(z) * 1e6, max(z) * 1e6, min(x) * 1e6, max(x) * 1e6])
    ax.set(xlabel='z (um)', ylabel='x (um)',
           title='alpha: %.3f; f: %.1f (mm); beta: %.1f; w0: %.1f (mm)' % (alpha, f * 1e3, beta_p * lambda0, w0 * 1e3))


def airy_numerical(beta=8, f=100e-3, w0=7e-3):
    # Constants
    lambda0 = 488e-9

    # Derived
    beta_p = beta / lambda0
    alpha = w0**(-2) * (3 * beta_p)**(-2 / 3)
    w1 = lambda0 * f * (3 * beta_p)**(1 / 3) / (2 * np.pi)

    # Focal coordinate
    x = np.linspace(-50e-6, 50e-6, 1024)

    # fft scaling
    pixPerM = 1 / (x[1] - x[0])
    ux = np.linspace(pixPerM / 2, -pixPerM / 2, x.shape[0])
    ux = ux * lambda0 * f

    # Numerical
    E1 = np.exp(1j * beta_p * ux**3) * np.exp(-ux**2 / w0**2)
    E2 = np.fft.fftshift(np.fft.fft(E1))

    fig, ax = plt.subplots()
    ax.plot(np.abs(E1) / np.max(np.abs(E1)), label='pupil')
    ax.plot(np.abs(E2) / np.max(np.abs(E2)), label='numerical')

    # Output scaling factor
    s = x / w1

    # Theoretical
    ai = special.airy(s)
    exponent = np.exp(alpha * s)
    Phi = np.abs(ai[0] * exponent)

    ax.plot(np.abs(Phi) / np.max(np.abs(Phi)), 'k:', label='theory')

    ax.legend(loc='upper right', shadow=True, fontsize='large')
    ax.set(xlabel='normalised distance', ylabel='normalised intensity',
           title='alpha: %.3f; f: %.1f (mm); beta: %.1f; w0: %.1f (mm)' % (alpha, f * 1e3, beta_p * lambda0, w0 * 1e3))


def airy_scale_inv(gamma=1 * 1, f=12e-3, w0=3e-3 * 1**(1 / 3)):
    # Constants
    lambda0 = 488e-9

    # Derived
    r0 = w0 * np.log(2)**(1 / 2)
    alpha = np.log(2) * (6 * np.pi * gamma)**(-2 / 3)
    x0 = lambda0 * f * (6 * np.pi * gamma)**(1 / 3) / (2 * np.pi * r0)

    # Focal coordinate
    x = np.linspace(-50e-6, 50e-6, 1024)

    # fft scaling
    pixPerM = 1 / (x[1] - x[0])
    ux = np.linspace(pixPerM / 2, -pixPerM / 2, x.shape[0])
    ux = ux * lambda0 * f

    # Numerical
    E1 = np.exp(1j * (2 * np.pi * gamma / (r0)**3) * ux**3) * np.exp(-ux**2 / w0**2)
    E2 = np.fft.fftshift(np.fft.fft(E1))

    fig, ax = plt.subplots()
    ax.plot(x * 1e6, np.abs(E1) / np.max(np.abs(E1)), label='pupil')
    ax.plot(x * 1e6, np.abs(E2) / np.max(np.abs(E2)), label='numerical')

    # Output scaling factor
    s = x / x0

    # Theoretical
    ai = special.airy(s)
    exponent = np.exp(alpha * s)
    Phi = np.abs(ai[0] * exponent)

    ax.plot(x * 1e6, np.abs(Phi) / np.max(np.abs(Phi)), 'k:', label='theory')

    ax.legend(loc='upper right', shadow=True, fontsize='large')
    ax.set(xlabel='normalised distance', ylabel='normalised intensity',
           title='alpha: %.3f; f: %.1f (mm); gamma: %.1f; w0: %.1f (mm)' % (alpha, f * 1e3, gamma, w0 * 1e3))

    plot_airy_cubic(beta=lambda0 * 2 * np.pi * gamma / (w0 * (np.log(2))**(1 / 2))**3, f=f, w0=w0)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Run this section as a script, while making all defined functions
    # available to the module

    # plot_pupil_function()
    # plot_airy_cubic()
    # airy_numerical()
    # plot_airy()
    airy_scale_inv()
