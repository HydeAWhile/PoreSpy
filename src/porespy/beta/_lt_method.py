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


def local_thickness(im, dt=None, smooth=False, approx=False):
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
    smooth : bool, optional
        Indicates if protrusions should be removed from the faces of the spheres
        or not. Default is `True`.
    approx : bool, optional
        If `True` the algorithm is more agressive at skipping voxels to process,
        which speeds things up, but this sacrifices accuracy in terms of a
        voxel-by-voxel match with the reference implementation. The default is
        `False`, meaning full accuracy is the default.

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

    # Call jitted function to draw spheres
    if im.ndim == 2:
        lt = _run2D(im, dt, ijk, smooth, approx)
    elif im.ndim == 3:
        lt = _run3D(im, dt, ijk, smooth, approx)

    return lt


@njit(parallel=True)
def _run2D(im, dt, ijk, smooth, approx):
    valid = np.copy(im)
    lt = np.zeros(im.shape, dtype=float)
    used = np.copy(lt)
    count = 0
    for idx in ijk:
        i = idx[0]
        j = idx[1]
        rval = dt[i, j]
        r = int(rval)
        # Since entries in ijk are sorted by size, once we reach an entry with
        # r = 0, then we know all remain entries will also be 0 so we can stop
        if r == 0:
            break
        # Only process if point has not yet been engulfed on previous step
        if valid[i, j]:
            used[i, j] = 1.0
            # Scan neighborhood around current pixel
            mn = r_to_inds_2d(r)
            for row in prange(len(mn[0])):
                m = mn[0][row] - r
                n = mn[1][row] - r
                if ((i + m) >= 0) and ((i + m) < im.shape[0]) \
                        and ((j + n) >= 0) and ((j + n) < im.shape[1]):
                    # Draw spheres within L of point (i, j)
                    L = r - ((m)**2 + (n)**2)**0.5 + 1
                    if (lt[i+m, j+n] == 0) and (L > 1 if smooth else L >= 1):
                        lt[i+m, j+n] = rval
                    # Use ints here since it's about actual sphere sizes
                    # not exact distances between pixel centers
                    if approx:
                        if int(dt[i+m, j+n]) <= int(L):
                            valid[i+m, j+n] = False
                    else:
                        if int(dt[i+m, j+n]) < int(L):
                            valid[i+m, j+n] = False
            count += 1
    return lt, count, used


@njit(parallel=True)
def _run3D(im, dt, ijk, smooth, approx):
    valid = np.copy(im)
    lt = np.zeros(im.shape, dtype=float)
    used = np.copy(lt)
    count = 0
    for idx in ijk:
        i = idx[0]
        j = idx[1]
        k = idx[2]
        rval = dt[i, j, k]
        r = int(rval)
        # Since entries in ijk are sorted by size, once we reach an entry with
        # r = 0, then we know all remain entries will also be 0 so we can stop
        if r == 0:
            break
        # Only process if point has not yet been engulfed on a previous step
        if valid[i, j, k]:
            used[i, j, k] = True
            # Scan neighborhood around current voxel
            mno = r_to_inds_3d(r)
            for row in prange(len(mno[0])):
                m = mno[0][row] - r
                n = mno[1][row] - r
                o = mno[2][row] - r
                if ((i + m) >= 0) and ((i + m) < im.shape[0]) \
                    and ((j + n) >= 0) and ((j + n) < im.shape[1]) \
                        and ((k + o) >= 0) and ((k + o) < im.shape[2]):
                    # Draw spheres within L of point (i, j, k)
                    L = r - (m**2 + n**2 + o**2)**0.5 + 1
                    if (lt[i+m, j+n, k+o] == 0) and \
                            (L > 1 if smooth else L >= 1):
                        lt[i+m, j+n, k+o] = rval
                    # Use ints here since it's about actual sphere
                    # sizes not exact distances between pixel centers
                    if approx:
                        if int(dt[i+m, j+n, k+o]) <= int(L):
                            valid[i+m, j+n, k+o] = False
                    else:
                        if int(dt[i+m, j+n, k+o]) < int(L):
                            valid[i+m, j+n, k+o] = False
            count += 1
    return lt, count, used


def r_to_inds(r, ndim):
    m = np.meshgrid(*[np.arange(2*r+1) for _ in range(ndim)])
    inds = np.vstack([n.flatten() for n in m]).T
    return inds


@njit
def r_to_inds_3d(r):
    size = 2*r + 1
    xx = np.empty(shape=(size**3), dtype=np.int_)
    yy = np.empty_like(xx)
    zz = np.empty_like(xx)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                xx[i*size**2 + j*size + k] = i
                yy[i*size**2 + j*size + k] = j
                zz[i*size**2 + j*size + k] = k
    return xx, yy, zz


@njit
def r_to_inds_2d(r):
    size = 2*r + 1
    xx = np.empty(shape=(size**2), dtype=np.int_)
    yy = np.empty_like(xx)
    for i in range(size):
        for j in range(size):
            xx[i*size + j] = i
            yy[i*size + j] = j
    return xx, yy


if __name__ == "__main__":
    import porespy as ps
    import matplotlib.pyplot as plt
    from localthickness import local_thickness as loct

    im = ~ps.generators.random_spheres([150, 150, 150], r=10, clearance=10, seed=0)
    dt = edt(im)
    ps.tools.tic()
    lt1, count, used = local_thickness(im, dt=dt, smooth=True, approx=True)
    t1 = ps.tools.toc(quiet=True)
    ps.tools.tic()
    lt2, count, used = local_thickness(im, dt=dt, smooth=True, approx=False)
    t2 = ps.tools.toc(quiet=True)
    ps.tools.tic()
    lt3 = ps.filters.local_thickness(im, sizes=np.unique(dt[im].astype(int)), mode='dt')
    t3 = ps.tools.toc(quiet=True)
    ps.tools.tic()
    lt4 = loct(im)
    t4 = ps.tools.toc(quiet=True)
    print(f"Times are:")
    print(f" Reference: {t2}")
    print(f" New Method: {t1}")
    print(f" PoreSpy: {t3}")
    print(f" Dahl: {t4}")
    print(f"Errors are:")
    print(f" New Method: {np.sum(lt2 != lt1)/im.sum()}")
    print(f" PoreSpy: {np.sum(lt2 != lt3)/im.sum()}")
    print(f" Dahl: {np.sum(lt2 != lt4)/im.sum()}")
    print(f"New method used {round(count/im.sum()*100, 2)}% of pixels")

    if im.ndim == 2:
        fig, ax = plt.subplots(1, 4)
        # ax[0].imshow(lt2 / im)
        ax[0].set_title('Reference')
        ax[0].axis('off')
        ax[1].imshow(lt1 / im)
        ax[1].set_title('New Method')
        ax[1].axis('off')
        ax[2].imshow(lt3 / im)
        ax[2].set_title('PoreSpy')
        ax[2].axis('off')
        ax[3].imshow(lt4 / im)
        ax[3].set_title('Dahl')
        ax[3].axis('off')
