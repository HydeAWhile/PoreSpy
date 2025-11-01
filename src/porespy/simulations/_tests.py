import numpy as np
import matplotlib.pyplot as plt
import porespy as ps

edt = ps.tools.get_edt()

im = ps.generators.blobs([200, 200], porosity=0.75, seed=0)
dt = edt(im).astype(int)

smooth = False
Rs = [13, 12]

# bf
nwp_bf = np.zeros_like(im)
seeds_prev = dt >= Rs[0] if smooth else dt >= Rs[0]
seeds_bf = dt >= Rs[1] if smooth else dt >= Rs[1]
edges = seeds_bf * ~seeds_prev
crds = np.vstack(np.where(edges))
nwp_bf = ps.tools._insert_disk_at_points(
    im=nwp_bf,
    coords=crds,
    r=Rs[1],
    v=1,
    smooth=smooth,
)
nwp_bf[seeds_bf] = True

# dt
seeds_dt = dt >= Rs[1] if smooth else dt >= Rs[1]
nwp_dt = edt(~seeds_dt) < Rs[1] if smooth else edt(~seeds_dt) <= Rs[1]

# dt_fft
seeds_dt_fft = dt >= Rs[1] if smooth else dt >= Rs[1]
se = ps.tools.ps_round(Rs[1], ndim=im.ndim, smooth=smooth)
nwp_dt_fft = ps.filters.fftmorphology(seeds_dt_fft, strel=se, mode='dilation')

# fft
se = ps.tools.ps_round(Rs[1], ndim=im.ndim, smooth=True)
seeds_fft = ~ps.filters.fftmorphology(~im, strel=se, mode='dilation')
se = ps.tools.ps_round(Rs[1], ndim=im.ndim, smooth=smooth)
nwp_fft = ps.filters.fftmorphology(seeds_fft, strel=se, mode='dilation')

fig, ax = plt.subplots(2, 2)
ax[0][0].imshow(nwp_bf/~edges/im)
ax[0][1].imshow(nwp_dt/~seeds_dt/im)
ax[1][0].imshow(nwp_dt_fft/~seeds_dt_fft/im)
ax[1][1].imshow(nwp_fft/~seeds_fft/im)

assert np.all(nwp_bf == nwp_dt)
assert np.all(nwp_bf == nwp_dt_fft)
assert np.all(nwp_bf == nwp_fft)

assert np.all(seeds_bf == seeds_dt)
assert np.all(seeds_bf == seeds_dt_fft)
assert np.all(seeds_bf == seeds_fft)


# bf
# nwp_bf = np.zeros_like(im)
# seeds_prev = dt >= Rs[0]
# seeds_bf = dt >= Rs[1]
# edges = seeds_bf * ~seeds_prev
# crds = np.vstack(np.where(edges))
# nwp_bf = ps.tools._insert_disk_at_points(
#     im=nwp_bf,
#     coords=crds,
#     r=Rs[1],
#     v=1,
#     smooth=smooth,
# )
# nwp_bf[seeds_bf] = True

# dt
# seeds_dt = dt >= Rs[1]
# nwp_dt = edt(~seeds_dt) < Rs[1] if smooth else edt(~seeds_dt) <= Rs[1]

# # dt_fft
# seeds_dt_fft = dt >= Rs[1]
# se = ps.tools.ps_round(Rs[1], ndim=im.ndim, smooth=smooth)
# nwp_dt_fft = ps.filters.fftmorphology(seeds_dt_fft, strel=se, mode='dilation')

# # fft
# se = ps.tools.ps_round(Rs[1], ndim=im.ndim, smooth=~smooth)
# seeds_fft = ~ps.filters.fftmorphology(~im, strel=se, mode='dilation')
# se = ps.tools.ps_round(Rs[1], ndim=im.ndim, smooth=smooth)
# nwp_fft = ps.filters.fftmorphology(seeds_fft, strel=se, mode='dilation')
