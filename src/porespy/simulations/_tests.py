import numpy as np
import matplotlib.pyplot as plt
import porespy as ps

edt = ps.tools.get_edt()

im = ps.generators.blobs([200, 200], porosity=0.75, seed=0)
dt = edt(im).astype(int)

Rs = [13, 12]

smooth = True
# bf
nwp = np.zeros_like(im)
seeds_prev = dt >= Rs[0]
seeds = dt >= Rs[1]
edges = seeds * ~seeds_prev
coords = np.vstack(np.where(edges))
nwp = ps.tools._insert_disk_at_points(
    im=nwp,
    coords=coords,
    r=Rs[1],
    v=True,
    smooth=smooth,
)
nwp[seeds] = True


plt.imshow(nwp/~edges/im)
