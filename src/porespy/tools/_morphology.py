import numpy as np
from scipy.signal import fftconvolve
from porespy.tools import (
    ps_round,
    get_edt
)


edt = get_edt()


__all__ = [
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
