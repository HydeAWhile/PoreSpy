import numpy as np


__all__ = [
    'gyroid',
    'tile',
]


def gyroid(shape, method='schoen', skew=0.5, phi=0.5):
    step = np.linspace(0, 2*np.pi, shape)
    x, y, z = np.meshgrid(step, step, step)

    if method == 'primitive':
        v = np.cos(x) + np.cos(y) + np.cos(z)
    if method == 'schoen':
        v = np.sin(x)*np.cos(y) + np.sin(y)*np.cos(z) + np.sin(z)*np.cos(x)
    if method == 'diamond':
        v = np.cos(x)*np.cos(y)*np.cos(z) + np.sin(x)*np.sin(y)*np.sin(z)
    if method == 'diagonal':
        v = 2*(np.cos(x)*np.cos(y) + np.cos(y)*np.cos(z) + np.cos(z)*np.cos(x)) \
            - (np.cos(2*x) + np.cos(2*y)+np.cos(2*z))
    if method == 'diamond2':
        v = np.sin(x)*np.sin(y)*np.sin(z) + np.sin(x)*np.cos(y)*np.cos(z) \
            + np.cos(x)*np.sin(y)*np.cos(z) + np.cos(x)*np.cos(y)*np.sin(z)
    if method == 'lidinoid':
        v = np.sin(2*x)*np.cos(y)*np.sin(z) + np.sin(x)*np.sin(2*y)*np.cos(z) \
            + np.cos(x)*np.sin(y)*np.sin(2*z) - np.cos(2*x)*np.cos(2*y) \
            - np.cos(2*y)*np.cos(2*z)-np.cos(2*z)*np.cos(2*x) + 0.3
    if method == 'split-p':
        v = 1.1*(np.sin(2*x)*np.cos(y)*np.sin(z)
                 + np.sin(x)*np.sin(2*y)*np.cos(z)
                 + np.cos(x)*np.sin(y)*np.sin(2*z)) \
            - 0.2*(np.cos(2*x)*np.cos(2*y)
                   + np.cos(2*y)*np.cos(2*z)
                   + np.cos(2*z)*np.cos(2*x)) \
            - 0.4*(np.cos(2*x) + np.cos(2*y) + np.cos(2*z))
    if method == 'neovius':
        v = 3.0 * (np.cos(x) + np.cos(x) + np.cos(z)) \
            + 4.0 * np.cos(x)*np.cos(y)*np.cos(z)
    if method == 'FKS':
        v = np.cos(2*x)*np.sin(y)*np.cos(z) + np.cos(2*y)*np.sin(z)*np.cos(x) \
            + np.cos(2*z)*np.sin(x)*np.cos(y)
    if method == 'FRD':
        v = 4*np.cos(x)*np.cos(y)*np.cos(z) \
            - (np.cos(2*x)*np.cos(2*y) + np.cos(2*x)*np.cos(2*z)
               + np.cos(2*y)*np.cos(2*z))
    if method == 'pw-hybrid':
        v = 4.0*(np.cos(x)*np.cos(y) + np.cos(y)*np.cos(z) + np.cos(z)*np.cos(x)) \
            - 3*np.cos(x)*np.cos(y)*np.cos(z) + 2.4
    if method == 'iWP':
        v = 2.0*(np.cos(x)*np.cos(y) + np.cos(z)*np.cos(x) + np.cos(y)*np.cos(z)) \
            - (np.cos(2*x) + np.cos(2*y) + np.cos(2*z))

    im = (v > (skew - phi))*(v < (skew + phi))
    return im


def tile(im, n, mode='periodic'):
    if mode == 'periodic':
        im2 = np.tile(im, n)
    elif mode == 'reflect':
        shape = np.array(im.shape)*(np.array(n)-1)
        pw = [(0, shape[i]) for i in range(im.ndim)]
        im2 = np.pad(im, pad_width=pw, mode='reflect')
    return im2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import porespy as ps
    ps.visualization.set_mpl_style()

    im = gyroid(shape=300, phi=0.5, skew=0.5, method='schoen')
    im2 = tile(im, n=(3, 3, 1), mode='periodic')
    print(im2.shape)
    fig, ax = plt.subplots(2, 2)
    ax[0][0].imshow(ps.visualization.sem(im2, axis=0), cmap=plt.cm.bone, vmax=1.5)
    ax[0][0].axis(False)
    ax[0][1].imshow(ps.visualization.xray(im2, axis=2), cmap=plt.cm.bone)
    ax[0][1].axis(False)
    ax[1][0].imshow(ps.visualization.sem(im2, axis=2).T, cmap=plt.cm.bone, vmax=1.5)
    ax[1][0].axis(False)
    ax[1][1].imshow(ps.visualization.xray(im2, axis=0), cmap=plt.cm.bone)
    ax[1][1].axis(False)
