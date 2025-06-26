import numpy as np
from porespy.tools import (
    get_edt,
    _insert_disk_at_points,
)
from numba import njit, prange


edt = get_edt()


def local_thickness_bf(im, dt=None, mask=None, smooth=True):
    r"""
    Insert a maximally inscribed sphere at every pixel labelled by sphere radius

    Parameters
    ----------
    im : ndarray
        Boolean image of the porous material
    dt : ndarray, optional
        The distance transform of the image
    mask : ndarray, optional
        A boolean mask indicating which sites to insert spheres at. If not provided
        then all `True` values in `im` are used.
    smooth : bool, optional
        Indicates if protrusions should be removed from the faces of the spheres
        or not. Default is `True`.

    Returns
    -------
    lt : ndarray
        The local thickness of the image with each voxel labelled according to the
        radius of the largest sphere which overlaps it

    Notes
    -----
    This function uses brute force, meaning that is inserts spheres at every single
    pixel or voxel without making any attempt to reduce the number of insertion
    sites. This should probably be considered a reference implementation.
    """
    if dt is None:
        dt = edt(im)
    if mask is None:
        mask = im
    args = np.argsort(dt.flatten())
    inds = np.vstack(np.unravel_index(args, dt.shape)).T
    if im.ndim == 2:
        lt = _run2D_bf(im, dt, mask, inds, smooth)
    elif im.ndim == 3:
        lt = _run3D_bf(im, dt, mask, inds, smooth)
    return lt


@njit
def _run2D_bf(im, dt, mask, inds, smooth):
    im2 = np.zeros(im.shape, dtype=float)
    # if im2.ndim == 2:
    for idx in inds:
        i = idx[0]
        j = idx[1]
        idx = np.array([[i, j]]).T
        r = dt[i, j]
        if mask[i, j]:
            im2 = _insert_disk_at_points(
                im=im2, coords=idx, r=int(r), v=r, overwrite=True, smooth=smooth)
    return im2


@njit
def _run3D_bf(im, dt, mask, inds, smooth):
    im3 = np.zeros(im.shape, dtype=float)
    for idx in inds:
        i = idx[0]
        j = idx[1]
        k = idx[2]
        idx = np.array([[i, j, k]]).T
        r = dt[i, j, k]
        if mask[i, j, k]:
            im3 = _insert_disk_at_points(
                im=im3, coords=idx, r=int(r), v=r, overwrite=True, smooth=smooth)
    return im3


def local_thickness(im, dt=None):
    r"""
    Insert a maximally inscribed sphere at every pixel labelled by sphere radius

    This version uses some logic to only insert spheres at locations which
    are not fully overlapped by larger spheres to reduce the number of insertions

    Parameters
    ----------
    im : ndarray
        Boolean image of the porous material
    dt : ndarray, optional
        The distance transform of the image

    Returns
    -------
    lt : ndarray
        The local thickness of the image with each voxel labelled according to the
        radius of the largest sphere which overlaps it
    """
    if dt is None:
        dt = edt(im)

    # Sort dt to scan sites from largest to smallest
    args = np.argsort(dt.flatten())[-1::-1]
    ijk = np.vstack(np.unravel_index(args, dt.shape)).T

    # Call jitted, parallelized function to draw spheres
    if im.ndim == 2:
        lt = _run2D(im, dt, ijk)
    elif im.ndim == 3:
        lt = _run3D(im, dt, ijk)

    return lt


@njit
def _run2D(im, dt, ijk):
    valid = np.copy(im)
    lt = np.zeros(im.shape, dtype=float)
    count = 0
    for idx in ijk:
        i = idx[0]
        j = idx[1]
        rval = dt[i, j]
        r = int(rval)
        if r == 0:
            break
        # Only process if point has not yet been engulfed on previous step
        if valid[i, j]:
            # Scan neighborhood around current pixel
            for m in range(-r, r + 1):
                if ((i + m) >= 0) and ((i + m) < im.shape[0]):
                    for n in range(-r, r + 1):
                        if ((j + n) >= 0) and ((j + n) < im.shape[1]):
                            # Draw spheres within L of point (i, j)
                            L = r - ((m)**2 + (n)**2)**0.5 + 1
                            if (lt[i+m, j+n] == 0) and (L >= 1):
                                lt[i+m, j+n] = rval
                            # Use ints here since it's about actual sphere sizes
                            # not exact distances between pixel centers
                            if int(dt[i+m, j+n]) < int(L):
                                valid[i+m, j+n] = False
            count += 1
    return lt, count


@njit
def _run3D(im, dt, ijk):
    valid = np.copy(im)
    lt = np.zeros(im.shape, dtype=float)
    count = 0
    for idx in ijk:
        i = idx[0]
        j = idx[1]
        k = idx[2]
        rval = dt[i, j, k]
        r = int(rval)
        if r == 0:
            break
        # Only process if point has not yet been engulfed on previous step
        if valid[i, j, k]:
            # Scan neighborhood around current pixel
            for m in range(-r, r + 1):
                if ((i + m) >= 0) and ((i + m) < im.shape[0]):
                    for n in range(-r, r + 1):
                        if ((j + n) >= 0) and ((j + n) < im.shape[1]):
                            for o in range(-r, r + 1):
                                if ((k + o) >= 0) and ((k + o) < im.shape[2]):
                                    # Draw spheres within L of point (i, j, k)
                                    L = r - (m**2 + n**2 + o**2)**0.5 + 1
                                    if (lt[i+m, j+n, k+o] == 0) and (L >= 1):
                                        lt[i+m, j+n, k+o] = rval
                                    # Use ints here since it's about actual sphere
                                    # sizes not exact distances between pixel centers
                                    if int(dt[i+m, j+n, k+o]) < int(L):
                                        valid[i+m, j+n, k+o] = False
            count += 1
    return lt, count


if __name__ == "__main__":
    import porespy as ps
    import matplotlib.pyplot as plt

    im = ~ps.generators.random_spheres([200, 200, 200], r=10, clearance=10, seed=0)
    dt = edt(im)
    ps.tools.tic()
    lt1, count = local_thickness(im)
    t1 = ps.tools.toc()
    # ps.tools.tic()
    # lt2 = local_thickness_bf(im, smooth=False)
    # t2 = ps.tools.toc()
    ps.tools.tic()
    lt3 = ps.filters.local_thickness(im, sizes=np.unique(dt[im].astype(int)))
    t3 = ps.tools.toc()
    print(f"Times are: {t1} and {t3}")

    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(lt2 / im)
    # ax[0].set_title('Reference')
    # ax[0].axis('off')
    # ax[1].imshow(lt1 / im)
    # ax[1].set_title('New Method')
    # ax[1].axis('off')
    # ax[2].imshow(lt3 / im)
    # ax[2].set_title('PoreSpy')
    # ax[2].axis('off')
