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
    lt = _run_bf(im, dt, mask, inds, smooth)
    return lt


@njit
def _run_bf(im, dt, mask, inds, smooth):
    im2 = np.zeros(im.shape, dtype=float)
    for idx in inds:
        i = idx[0]
        j = idx[1]
        idx = np.array([[i, j]]).T
        r = dt[i, j]
        if mask[i, j]:
            im2 = _insert_disk_at_points(
                im=im2, coords=idx, r=int(r), v=r, overwrite=True, smooth=smooth)
    return im2


def local_thickness(im, dt=None):
    r"""
    Insert a maximally inscribed sphere at every pixel labelled by sphere radius

    This version uses special logic to only insert spheres at locations which
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
    lt = _run(im, dt, ijk)

    return lt


# @njit(parallel=True)
@njit(parallel=False)
def _run(im, dt, ijk):
    valid = np.copy(im)
    lt = np.zeros(im.shape, dtype=float)
    count = 0
    rval = 0.0
    r = 0
    for idx in ijk:
        i = idx[0]
        j = idx[1]
        # Only process if point has not been engulfed yet on previous step
        if valid[i, j]:
            rval = dt[i, j]
            if rval == 0:
                break
            r = int(rval)
            # Scan neighborhood around current pixel
            # for ptr in prange(indptr[r-1], indptr[r]):  # Parallel seems slower!
            for m in range(-r, r + 1):
                if ((i + m) >= 0) and ((i + m) < im.shape[0]):
                    for n in range(-r, r + 1):
                        if ((j + n) >= 0) and ((j + n) < im.shape[1]):
                            # Draw spheres within L of point (i, j)
                            L = r - ((m)**2 + (n)**2)**0.5 + 1
                            if (lt[i+m, j+n] == 0) and (L >= 1):
                                lt[i+m, j+n] = rval
                            # Use ints here since it's about actual sphere sizes not
                            # exact distances between pixel centers.
                            if int(dt[i+m, j+n]) < int(L):
                                valid[i+m, j+n] = False
            count += 1
    return lt, count


if __name__ == "__main__":
    import porespy as ps
    import matplotlib.pyplot as plt

    im = ~ps.generators.random_spheres([600, 600], r=10, clearance=10, seed=0)
    lt, count = local_thickness(im)
    im3 = local_thickness_bf(im, smooth=False)
    print(f"Total steps: {count/im.sum()*100}%")
    print(f"Error: {np.sum(im3 != lt)/im.sum()*100}% ")

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(im3 / im)
    ax[0].set_title('Reference')
    ax[0].axis('off')
    ax[1].imshow(lt / im)
    ax[1].set_title('New Method')
    ax[1].axis('off')
    ax[2].imshow((im3 / lt)/im, vmin=1, vmax=1.1)
    ax[2].set_title('Difference')
    ax[2].axis('off')
