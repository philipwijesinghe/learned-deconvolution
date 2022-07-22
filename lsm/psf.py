# -*- coding: utf-8 -*-
"""PSF module

Generates an LSM point-spread function
"""

# Created on Thu Mar  21, 2020

# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com

# Copyright (c) 2020 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)

import numpy as np

from beams.airy import gen_airy, gen_airy_vect
from beams.bessel import gen_bessel, gen_bessel_vect
from beams.gaussian import gen_gaussian, gen_gaussian_vect


# =============================================================================
# CONFIGURATION CLASS
# =============================================================================
# Since the generation of the PSF requires a long list of properties related to
# the LSM system, we create a default Class to provide a template and some form
# of validation

class PSFConfig:
    # Creates a configuration class for PSF generation
    # Addressable by
    def __init__(self):
        # Use a nested dict for now
        # Default only required parameters!
        default_conf = {
            'experimental': {
                'dir': ''
            },
            'illumination': {
                'beam': 'gaussian',
                'lambda': 488e-9,
                # 'f': 70e-3,
                # 'w0': 2.5e-3,
                # 'beta': 15,
                # 'gamma': 1,
                # 'rr': 2.5e-3,
                'defocus': 0,
                # 'wg': 2e-6,
                # 'plane': 'pupil'
            },
            'detection': {
                'wg': 2e-6
            }
        }

        self.config = default_conf  # set it to conf

    # We are gonna make a generic config, and directly addressable, for now.
    # This is ok for testing now, but will confuse everyone trying to use this
    # later. What configs need to be set is not obvious. Errors incoming.
    # Later, we'll make a proper template and make inherited classes for each
    # beam shape.
    # TODO: this was a todo 2 years ago!

    # def get_property(self, property_name):
    #    if property_name not in self._config.keys(): # we don't want KeyError
    #        return None  # just return None if not found
    #    return self._config[property_name]


# =============================================================================
# FUNCTIONS
# =============================================================================
def gen_psf_vect(x, z, y=0, lambda0=488e-9, psfconfig=PSFConfig(), **kwargs):
    """Generates an LSM PSF (intensity) at (x,z) vectors

    I = gen_psf(r, z, lambda0, psfconfig)

    Example
    -------

    psfconfig = PSFConfig()
    E = gen_psf(r, z, lambda0=488e-9, psfconfig=psfconfig)

    Parameters
    ----------
    x : np array
        Vector of r values (m)
    z : np array
        Vector of z values, relative to the focal plane (m).
    y : floar or np array, optional
        Float or vector of defocus values (m)
    lambda0 : float, optional
        Wavelength (m). The default is 488e-9.
    psfconfig: PSFConfig Object
        Sets psf properties
    **kwargs : Keyword options
        NOT IMPLEMENTED

    Returns
    -------
    I : np array, dtype=float64
        2D array (z,x)

    """

    lambda0_illumination = psfconfig.config['illumination']['lambda'] \
        if 'lambda' in psfconfig.config['illumination'] else lambda0
    lambda0_detection = psfconfig.config['detection']['lambda'] \
        if 'lambda' in psfconfig.config['detection'] else lambda0

    # Illumination PSF
    #   Out-of-focus simulation
    zf = x - psfconfig.config['illumination']['defocus']
    #   Due to orthogonal illumination, we transpose x and z
    if psfconfig.config['illumination']['beam'] == 'gaussian':
        illumination_e_field = gen_gaussian_vect(
            r=z, z=zf, lambda0=lambda0_illumination,
            **psfconfig.config['illumination']
        )
    elif psfconfig.config['illumination']['beam'] == 'airy':
        illumination_e_field = gen_airy_vect(
            x=z, z=zf, lambda0=lambda0_illumination,
            **psfconfig.config['illumination']
        )
    elif psfconfig.config['illumination']['beam'] == 'bessel':
        illumination_e_field = gen_bessel_vect(
            r=z, z=zf, lambda0=lambda0_illumination,
            **psfconfig.config['illumination']
        )
    else:
        print('Undefined beam type')
        return

    # Detection PSF
    r = (x ** 2 + y ** 2) ** (1 / 2)
    detection_e_field = gen_gaussian_vect(
        r=r, z=z, lambda0=lambda0_detection,
        wg=psfconfig.config['detection']['wg']
    )
    detection_e_field = detection_e_field.transpose()

    # Combined PSF
    intensity = np.abs(illumination_e_field) ** 2 * np.abs(detection_e_field) ** 2

    return intensity


# DEPRECATED
def gen_psf(x, z, lambda0=488e-9, psfconfig=PSFConfig()):
    print("Warning. Function gen_psf deprecated. Use gen_psf_vect.")

    xv, zv = np.meshgrid(x, z, sparse=False, indexing='ij')

    return gen_psf_vect(xv, zv, 0, lambda0, psfconfig)


# def gen_psf(x, z, lambda0=488e-9, psfconfig=PSFConfig(), **kwargs):
#     """Generates an LSM PSF (intensity)
#
#     I = gen_psf(r, z, lambda0, psfconfig)
#
#     Example
#     -------
#
#     psfconfig = PSFConfig()
#     z = np.linspace(-50e-6, 50e-6, 201)
#     x = np.linspace(-150e-6, 150e-6, 201)
#     E = gen_psf(r, z, lambda0=488e-9, psfconfig=psfconfig)
#
#     Parameters
#     ----------
#     x : np array
#         Vector of r values (m)
#     z : np array
#         Vector of z values, relative to the focal plane (m).
#     lambda0 : float, optional
#         Wavelength (m). The default is 488e-9.
#     psfconfig: PSFConfig Object
#         Sets psf properties
#     **kwargs : Keyword options
#         NOT IMPLEMENTED
#
#     Returns
#     -------
#     I : np array, dtype=float64
#         2D array (z,x)
#
#     """
#
#     # Illumination PSF
#     #   Out-of-focus simulation
#     zf = x - psfconfig.config['illumination']['defocus']
#     #   Due to orthogonal illumination, we transpose x and z
#     if psfconfig.config['illumination']['beam'] == 'gaussian':
#         illumination_e_field = gen_gaussian(
#             r=z, z=zf, lambda0=lambda0,
#             **psfconfig.config['illumination']
#         )
#     elif psfconfig.config['illumination']['beam'] == 'airy':
#         illumination_e_field = gen_airy(
#             x=z, z=zf, lambda0=lambda0,
#             **psfconfig.config['illumination']
#         )
#     elif psfconfig.config['illumination']['beam'] == 'bessel':
#         illumination_e_field = gen_bessel(
#             r=z, z=zf, lambda0=lambda0,
#             **psfconfig.config['illumination']
#         )
#     else:
#         print('Undefined beam type')
#         return
#
#     # Detection PSF
#     detection_e_field = gen_gaussian(r=x, z=z, lambda0=lambda0, wg=psfconfig.config['detection']['wg'])
#     detection_e_field = detection_e_field.transpose()
#
#     # Combined PSF
#     intensity = np.abs(illumination_e_field) ** 2 * np.abs(detection_e_field) ** 2
#
#     return intensity


# =============================================================================
# MAIN
# =============================================================================
# if __name__ == "__main__":
    # Run this section as a script, while making all defined functions
    # available to the module

    # NOTE: uses parent modules
    # from beam import    will not work unless modules are loaded or the module
    # is run as a script with the '-m' modifier

    # # load default configuration
    # psfconfig = PSFConfig()
    #
    # # focal coordinates
    # z = np.linspace(-30e-6, 30e-6, 511)
    # x = np.linspace(-20e-6, 20e-6, 511)
    #
    # E = gen_psf(x, z, lambda0=488e-9, psfconfig=psfconfig)
