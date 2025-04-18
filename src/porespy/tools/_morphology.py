import numpy as np
from scipy.signal import fftconvolve
from porespy.tools import (
    ps_round,
    get_edt
)


edt = get_edt()


__all__ = [
    'erode',
    'dilate',
    'get_conns',
    'ball',
    'disk',
    'cube',
    'square',
]


def ball(r):
    se = np.ones([r*2+1]*3, dtype=bool)
    se[r, r, r] = False
    se = edt(se) <= r
    return se


def disk(r):
    se = np.ones([r*2+1]*2, dtype=bool)
    se[r, r] = False
    se = edt(se) <= r
    return se


def cube(w):
    se = np.ones([w, w, w], dtype=bool)
    return se


def square(w):
    se = np.ones([w, w], dtype=bool)
    return se


def get_conns():
    se = {2: {'min': disk(1),
              'max': square(3)},
          3: {'min': ball(1),
              'max': cube(3)}}
    return se


def erode(im, r, dt=None, method='dt', smooth=True):
    from porespy import settings
    if dt is None:
        dt = edt(im, parallel=settings.ncores)
    if method == 'dt':
        ero = dt >= r if smooth else dt > r
    elif method.startswith('conv'):
        se = ps_round(r=r, ndim=im.ndim, smooth=smooth)
        ero = ~(fftconvolve(~im, se, mode='same') > 0.1)
    return ero


def dilate(im, r, method='dt', smooth=True):
    from porespy import settings
    im = im == 1
    if method == 'dt':
        dt = edt(~im, parallel=settings.ncores)
        dil = dt < r if smooth else dt <= r
        dil += im
    elif method.startswith('conv'):
        se = ps_round(r=r, ndim=im.ndim, smooth=smooth)
        dil = fftconvolve(im, se, mode='same') > 0.1
    return dil


if __name__ == "__main__":
    import porespy as ps
    import matplotlib.pyplot as plt

    im = ps.generators.blobs([200, 200], porosity=0.6, seed=5)

    ero1 = erode(im, 5, method='dt').astype(int)
    ero1[~im] = -1

    ero2 = erode(im, 5, method='conv').astype(int)
    ero2[~im] = -1

    fig, ax = plt.subplots(2, 2)
    ax[0][0].imshow(ero1)
    ax[0][1].imshow(ero2)

    ero1 = erode(im, 10, method='dt').astype(int)
    dil1 = dilate(ero1, 10, method='dt').astype(int)
    dil1[~im] = -1

    dil2 = dilate(ero1, 10, method='conv').astype(int)
    dil2[~im] = -1

    ax[1][0].imshow(dil1)
    ax[1][1].imshow(dil2)
