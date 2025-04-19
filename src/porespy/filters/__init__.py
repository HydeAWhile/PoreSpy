r"""

Collection of functions for altering images based on structural properties
##########################################################################

This module contains a variety of functions for altering images based on
the structural characteristics, such as pore sizes.  A definition of a
*filter* is a function that returns an image the same shape as the original
image, but with altered values.

.. currentmodule:: porespy

.. autosummary::
   :template: mybase.rst
   :toctree: generated/

    filters.apply_chords
    filters.apply_chords_3D
    filters.apply_padded
    filters.capillary_transform
    filters.chunked_func
    filters.dilate
    filters.distance_transform_lin
    filters.erode
    filters.fftmorphology
    filters.fill_closed_pores
    filters.fill_trapped_voxels
    filters.find_closed_pores
    filters.find_disconnected_voxels
    filters.find_dt_artifacts
    filters.find_invalid_pores
    filters.find_peaks
    filters.find_surface_pores
    filters.find_trapped_regions
    filters.flood
    filters.flood_func
    filters.hold_peaks
    filters.local_thickness
    filters.nl_means_layered
    filters.nphase_border
    filters.pc_to_satn
    filters.pc_to_seq
    filters.porosimetry
    filters.prune_branches
    filters.reduce_peaks
    filters.region_size
    filters.satn_to_seq
    filters.seq_to_satn
    filters.size_to_pc
    filters.size_to_satn
    filters.size_to_seq
    filters.snow_partitioning
    filters.snow_partitioning_n
    filters.snow_partitioning_parallel
    filters.trim_disconnected_blobs
    filters.trim_extrema
    filters.trim_floating_solid
    filters.trim_nearby_peaks
    filters.trim_nonpercolating_paths
    filters.trim_saddle_points
    filters.trim_saddle_points_legacy
    filters.trim_small_clusters

"""

from ._fftmorphology import *
from ._funcs import *
from ._fill_and_find_funcs import *
from ._nlmeans import *
from ._size_seq_satn import *
from ._snows import *
from ._transforms import *
from ._invasion import *
from ._morphology import *
