import numpy as np
import porespy as ps
from edt import edt


def fibers(shape, r, n):
    r"""

    [1]_ Beckman IP, Beckman PM, Cho H, Riveros G. Modeling uniform random
         distributions of nonwoven fibers for computational analysis of
         composite materials. Composite Structures. 301(12) 116242 (2022).
         `doi<https://doi.org/10.1016/j.compstruct.2022.116242>`_
    """
    im = np.zeros(shape)
    x1, y1 = im.shape[0]/2, im.shape[1]/2
    Lmax = (x1**2 + y1**2)**0.5
    for _ in range(n):
        phi = np.random.rand()*2*np.pi
        L = 2*(0.5 - np.random.rand())*Lmax
        x2 = x1 + L*np.cos(phi)
        y2 = y1 + L*np.sin(phi)
        x3 = x2 + 2*Lmax*np.cos(phi + np.pi/2)
        y3 = y2 + 2*Lmax*np.sin(phi + np.pi/2)
        x4 = x2 + 2*Lmax*np.cos(phi - np.pi/2)
        y4 = y2 + 2*Lmax*np.sin(phi - np.pi/2)
        X = ps.generators.line_segment((x3, y3), (x4, y4))
        mask = (X[0] >= 0)*(X[0] < im.shape[0])*(X[1] >= 0)*(X[1] < im.shape[1])
        X = X[0][mask], X[1][mask]
        if im.ndim == 3:
            X = X[0], X[1], np.random.randint(im.shape[0])
        im[*X] += 1
    if r is not None:
        dt = edt(im == 0)
        im = dt >= r
    return im


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fibs = fibers([500, 500, 500], r=5, n=1000)
    print(f"Porosity: {fibs.sum()/fibs.size}")

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(ps.visualization.xray(fibs > 0, axis=2))
    ax[1].imshow(ps.visualization.xray(fibs > 0, axis=1).T)
    ax[2].imshow(ps.visualization.xray(fibs > 0, axis=0).T)
