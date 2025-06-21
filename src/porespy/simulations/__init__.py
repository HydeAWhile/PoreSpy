r"""

Collection of functions for performing numerical simulations on images
######################################################################

This module contains routines for performing simulations directly on images.

.. currentmodule:: porespy

.. autosummary::
   :template: mybase.rst
   :toctree: generated/

    simulations.drainage
    simulations.drainage_dsi
    simulations.drainage_dt
    simulations.drainage_dt_fft
    simulations.drainage_fft
    simulations.ibip
    simulations.imbibition
    simulations.imbibition_dsi
    simulations.imbibition_dt
    simulations.imbibition_dt_fft
    simulations.imbibition_fft
    simulations.injection
    simulations.qbip
    simulations.tortuosity_fd

"""

from ._dns import *
from ._drainage import *
from ._imbibition import *
from ._injection import *
