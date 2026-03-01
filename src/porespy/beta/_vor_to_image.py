import numpy as np
import scipy.ndimage as spim
from skimage.segmentation import watershed
from porespy.generators import line_segment
from porespy.tools import get_edt


__all__ = [
    'vor_to_im',
    'pts_to_voronoi',
]


edt = get_edt()


def vor_to_im(vor, im, centroids=True):
    im = np.zeros_like(im)
    Nx, Ny = im.shape
    for v in vor.ridge_vertices:
        if -1 not in v:  # Need a way to deal with instead of ignore -1's
            X0, X1 = vor.vertices[v]
            inds = line_segment(X0, X1)
            masks = [(inds[i] >= 0)*(inds[i] < im.shape[i]) for i in range(im.ndim)]
            mask = np.all(np.vstack(masks), axis=0)
            inds = [inds[i][mask] for i in range(im.ndim)]
            if len(inds[0]) > 0:
                im[tuple(inds)] = True
    if centroids:
        pts = vor.points
        pts = np.vstack(pts)
        im[tuple((pts).astype(int).T)] = True
    return im


def pts_to_voronoi(im, r=0, centroids=True, borders=True):
    dt = edt(~im)
    markers, N = spim.label(im)
    ws = watershed(dt, markers, watershed_line=borders)
    for _ in range(r):
        pts = spim.center_of_mass(input=ws, labels=ws, index=range(1, N))
        pts = np.vstack(pts)
        im = np.zeros_like(im, dtype=bool)
        im[tuple((pts).astype(int).T)] = 1
        ws = pts_to_voronoi(im, r=0, centroids=False, borders=borders)
    if centroids:
        pts = spim.center_of_mass(input=ws, labels=ws, index=range(1, N))
        pts = np.vstack(pts)
        ws[tuple((pts).astype(int).T)] = -1
    return ws


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import porespy as ps
    import scipy.spatial as sptl

    fig, ax = plt.subplots(1, 2)

    np.random.seed(0)
    Np = 500
    Nx = 500
    Ny = 500
    r = 2

    im = np.zeros([Nx, Ny], dtype=bool)
    pts = np.random.rand(Np, 2)*im.shape

    vor = sptl.Voronoi(pts)
    im1 = vor_to_im(vor, im)
    im1 = spim.label(~im1)[0]
    im1 = ps.tools.randomize_colors(im1)
    ax[0].imshow(im1, origin='lower', interpolation='none', cmap=plt.cm.plasma)

    im = np.zeros([Nx, Ny], dtype=bool)
    im[tuple((pts).astype(int).T)] = 1
    im2 = pts_to_voronoi(im, r=r, borders=True)
    im2 = ps.tools.randomize_colors(im2)
    ax[1].imshow(im2, origin='lower', interpolation='none', cmap=plt.cm.plasma)
