import logging
import numpy as np
import numpy.typing as npt
import scipy.ndimage as spim
from typing import Literal
from skimage.segmentation import clear_border
from skimage.morphology import ball, disk, square, cube
from porespy.tools import (
    _check_for_singleton_axes,
    get_edt,
)


__all__ = [
    "fill_closed_pores",
    "find_disconnected_voxels",
    "find_surface_pores",
    "find_closed_pores",
    "find_invalid_pores",
    "trim_floating_solid",
    "trim_nonpercolating_paths",
    "trim_small_clusters",
]


edt = get_edt()
logger = logging.getLogger(__name__)
strel = {2: {'min': disk(1), 'max': square(3)}, 3: {'min': ball(1), 'max': cube(3)}}


def trim_small_clusters(
    im: npt.NDArray,
    size: int = 1,
):
    r"""
    Remove isolated voxels or clusters of a given size or smaller

    Parameters
    ----------
    im : ndarray
        The binary image from which voxels are to be removed.
    size : scalar
        The threshold size of clusters to trim.  As clusters with this
        many voxels or fewer will be trimmed.  The default is 1 so only
        single voxels are removed.

    Returns
    -------
    im : ndarray
        A copy of `im` with clusters of voxels smaller than the given
        `size` removed.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/trim_small_clusters.html>`__
    to view online example.

    """
    se = strel[im.ndim]['min']
    filtered_array = np.copy(im)
    labels, N = spim.label(filtered_array, structure=se)
    id_sizes = np.array(spim.sum(im, labels, range(N + 1)))
    area_mask = id_sizes <= size
    filtered_array[area_mask[labels]] = 0
    return filtered_array


def find_disconnected_voxels(
    im: npt.NDArray,
    conn: Literal['min', 'max'] = "max",
    surface: bool = False,
):
    r"""
    Identifies all voxels that are not connected to the edge of the image.

    Parameters
    ----------
    im : ndarray
        A Boolean image, with `True` values indicating the phase for which
        disconnected voxels are sought.
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.
    fill_surface : bool
        If `True` any isolated regions touching the edge of the image are
        considered disconnected.

    Returns
    -------
    image : ndarray
        An ndarray the same size as `im`, with `True` values indicating
        voxels of the phase of interest (i.e. `True` values in the original
        image) that are not connected to the outer edges.

    See Also
    --------
    fill_closed_pores
    trim_floating_solid

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/find_disconnected_voxels.html>`__
    to view online example.
    """
    _check_for_singleton_axes(im)

    se = strel[im.ndim][conn].copy()
    labels, N = spim.label(input=im, structure=se)
    if not surface:
        holes = clear_border(labels=labels) > 0
    else:
        keep = set(np.unique(labels))
        for ax in range(labels.ndim):
            labels = np.swapaxes(labels, 0, ax)
            keep.intersection_update(set(np.unique(labels[0, ...])))
            keep.intersection_update(set(np.unique(labels[-1, ...])))
            labels = np.swapaxes(labels, 0, ax)
        holes = np.isin(labels, list(keep), invert=True)
    return holes


def find_closed_pores(
    im: npt.NDArray,
    conn: Literal['max', 'min'] = 'min',
):
    r"""
    Finds closed pores that a not connected to any surface

    Parameters
    ----------
    im : ndarray
        A boolean array with `True` indicating the phase of interest
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.

    Returns
    -------
    closed : ndarray
        A array containing boolean values indicating voxels which belong to closed
        pores.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/find_closed_pores.html>`__
    to view online example.
    """
    from porespy.generators import borders
    se = strel[im.ndim][conn].copy()
    labels, N = spim.label(input=im, structure=se)
    mask = borders(im.shape, mode='faces')
    hits = np.unique(labels[mask])
    closed = np.isin(labels, hits, invert=True)
    return closed


def find_surface_pores(
    im: npt.NDArray,
    conn: Literal['max', 'min'] = 'min',
):
    r"""
    Finds surface pores that do not span the domain

    Parameters
    ----------
    im : ndarray
        A boolean array with `True` indicating the phase of interest
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.

    Returns
    -------
    surface : ndarray
        A array containing boolean values indicating voxels which belong to surface
        pores.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/find_surface_pores.html>`__
    to view online example.
    """
    se = strel[im.ndim][conn].copy()
    labels, N = spim.label(input=im, structure=se)
    keep = set()
    for ax in range(labels.ndim):
        labels = np.swapaxes(labels, 0, ax)
        s1 = set(np.unique(labels[0, ...]))
        s2 = set(np.unique(labels[-1, ...]))
        tmp = s1.intersection(s2)
        keep.update(tmp)
        labels = np.swapaxes(labels, 0, ax)
    closed = find_closed_pores(im, conn=conn)
    surface = np.isin(labels, list(keep), invert=True) * ~closed
    return surface


def find_invalid_pores(
    im: npt.NDArray,
    conn: Literal['max', 'min'] = 'min',
):
    r"""
    Finds invalid pores which are either closed or do not span the domain

    Parameters
    ----------
    im : ndarray
        A boolean array with `True` indicating the phase of interest
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.

    Returns
    -------
    invalid : ndarray
        A array containing `1` indicated closed pores and `2` indicating surface
        pores.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/find_invalid_pores.html>`__
    to view online example.
    """
    closed = find_closed_pores(im=im, conn=conn)
    surface = find_surface_pores(im=im, conn=conn)
    invalid = closed.astype(int) + 2*surface.astype(int)
    return invalid


def fill_closed_pores(
    im: npt.NDArray,
    conn: Literal['max', 'min'] = 'min',
    surface: bool = False,
):
    r"""
    Fills all closed pores that are isolated from the main void space.

    Parameters
    ----------
    im : ndarray
        The image of the porous material

    Returns
    -------
    im : ndarray
        A Boolean image, with `True` values indicating the phase of interest.
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.
    fill_surface : bool
        If `True`, any isolated pore regions that are connected to the
        sufaces of the image are but not connected to the main percolating
        path are also removed. When this is enabled, only the voxels
        belonging to the largest region are kept. This can be
        problematic if image contains non-intersecting tube-like structures,
        for instance, since only the largest tube will be preserved.

    Returns
    -------
    im : ndarray
        A version of `im` but with all the closed or disconnected pores converted
        to solid (i.e. `False`)

    See Also
    --------
    find_disconnected_voxels
    trim_nonpercolating_paths

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/fill_closed_pores.html>`__
    to view online example.

    """
    im = np.copy(im)
    holes = find_disconnected_voxels(im, conn=conn, surface=surface)
    im[holes] = False
    return im


def trim_floating_solid(
    im: npt.NDArray,
    conn: Literal['max', 'min'] = 'min',
    surface: bool = False,
):
    r"""
    Removes all solid that that is not attached to main solid structure.

    Parameters
    ----------
    im : ndarray
        The image of the porous material
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.
    fill_surface : bool
        If `True`, any isolated solid regions that are connected to the
        surfaces of the image but not the main body of the solid are also
        removed.  Voxels are deemed to be surface voxels if they are part of a
        cluster that does not span the domain. In other words, a cluster of voxels
        touching the `x=0` face but not the `x=-1` face will be trimmed if this
        is enabled.

    Returns
    -------
    image : ndarray
        A version of `im` but with all the disconnected solid removed.

    See Also
    --------
    find_disconnected_voxels
    trim_nonpercolating_paths

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/trim_floating_solid.html>`__
    to view online example.

    """
    im = np.copy(im)
    holes = find_disconnected_voxels(~im, conn=conn, surface=surface)
    im[holes] = True
    return im


def trim_nonpercolating_paths(
    im: npt.NDArray,
    axis: int = None,
    inlets: npt.NDArray = None,
    outlets: npt.NDArray = None,
    conn: Literal['max', 'min'] = 'min',
):
    r"""
    Remove all nonpercolating pores between specified locations

    Parameters
    ----------
    im : ndarray
        The image of the porous material with `True` values indicating the
        phase of interest
    axis : int, optional
        An integer indicating that axis along which the inlet and outlet faces
        should be applied.  For instance if `axis=0` then the inlets will be
        at `im[0, ...]` and the outlets will be at `im[-1, ...]`. If this argument
        is given then `inlets` and `outlets` are ignored.
    inlets outlets : ndarray, optional
        A boolean mask indicating locations of inlets and outlets, such as produced
        by `porespy.generators.faces`. This can be used instead of `axis` to provide
        more control. This is ignored if `axis` is provided.
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.

    Returns
    -------
    image : ndarray
        A copy of `im` with all the nonpercolating paths removed

    Notes
    -----
    This function is essential when performing transport simulations on an
    image since regions that do not span between the desired inlet and
    outlet do not contribute to the transport.

    See Also
    --------
    find_disconnected_voxels
    trim_floating_solid
    fill_closed_pores

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/trim_nonpercolating_paths.html>`__
    to view online example.

    """
    if axis is not None:
        from porespy.generators import faces
        inlets = faces(im.shape, inlet=axis)
        outlets = faces(im.shape, outlet=axis)
    se = strel[im.ndim][conn].copy()
    labels = spim.label(im, structure=se)[0]
    IN = np.unique(labels * inlets)
    OUT = np.unique(labels * outlets)
    hits = np.array(list(set(IN).intersection(set(OUT))))
    new_im = np.isin(labels, hits[hits > 0])
    return new_im
