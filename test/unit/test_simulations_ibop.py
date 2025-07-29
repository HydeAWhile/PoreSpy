import numpy as np
from GenericTest import GenericTest
import matplotlib.pyplot as plt

import porespy as ps

ps.visualization.set_mpl_style()


ps.settings.tqdm["disable"] = True


class IBOPTest(GenericTest):
    def setup_class(self):
        self.im2D = ps.generators.blobs(
            shape=[300, 150],
            porosity=0.6,
            seed=0,
            periodic=False,
        )

    def test_ibop_w_and_wo_pc(self):
        # bins must be none to ensure they both use same bins (i.e. all of them)
        r1 = ps.simulations.drainage(im=self.im2D, steps=None)
        assert np.sum(r1.im_seq == -1) == 0
        pc = ps.filters.capillary_transform(im=self.im2D)
        r2 = ps.simulations.drainage(im=self.im2D, pc=pc, steps=None)
        assert np.all(r1.im_seq == r2.im_seq)

    def test_ibop_w_trapping(self):
        inlets = ps.generators.faces(shape=self.im2D.shape, inlet=0)
        r1 = ps.simulations.drainage(im=self.im2D, inlets=inlets, steps=None)
        outlets = ps.generators.faces(shape=self.im2D.shape, outlet=0)
        r2 = ps.simulations.drainage(
            im=self.im2D, inlets=inlets, outlets=outlets, steps=None
        )
        assert np.sum(r1.im_seq == -1) == 0
        assert np.sum(r2.im_seq == -1) == 7073
        temp = ps.filters.fill_invalid_pores(self.im2D)
        r3 = ps.simulations.drainage(im=temp, inlets=inlets, steps=None)
        assert np.sum(r3.im_seq == -1) == 0

    def test_ibop_w_residual(self):
        rs = ps.filters.local_thickness(self.im2D) > 20
        inlets = ps.generators.faces(shape=self.im2D.shape, inlet=0)
        r1 = ps.simulations.drainage(
            im=self.im2D, inlets=inlets, residual=rs, steps=None)
        # Ensure all residual voxels have a sequence of 0 (invaded before first step)
        assert np.all(r1.im_seq[rs] == 0)

    def test_drainage_implementations_no_inlets(self):
        edt = ps.tools.get_edt()
        im = ps.generators.blobs(
            shape=[100, 100],
            porosity=0.7,
            blobiness=1.5,
        )

        # All methods are equivalent IF dt is integers
        dt = edt(im).astype(int)
        steps = np.unique(dt[im])

        sizes1 = ps.simulations.drainage_dt(im=im, dt=dt, steps=steps).im_size
        sizes2 = ps.simulations.drainage_conv(im=im, dt=dt, steps=steps).im_size
        sizes3 = ps.simulations.drainage_bf(im=im, dt=dt, steps=steps).im_size
        sizes4 = ps.simulations.drainage_dt_conv(im=im, dt=dt, steps=steps).im_size
        assert np.all(sizes1 == sizes2)
        assert np.all(sizes1 == sizes3)
        assert np.all(sizes1 == sizes4)

        seq1 = ps.simulations.drainage_dt(im=im, dt=dt, steps=steps).im_seq
        seq2 = ps.simulations.drainage_conv(im=im, dt=dt, steps=steps).im_seq
        seq3 = ps.simulations.drainage_bf(im=im, dt=dt, steps=steps).im_seq
        seq4 = ps.simulations.drainage_dt_conv(im=im, dt=dt, steps=steps).im_seq
        assert np.all(seq1 == seq2)
        assert np.all(seq1 == seq3)
        assert np.all(seq1 == seq4)

        # Or we can specify discrete steps
        steps = np.arange(50, 1, -1)
        sizes1 = ps.simulations.drainage_dt(im=im, steps=steps).im_size
        sizes2 = ps.simulations.drainage_conv(im=im, steps=steps).im_size
        sizes3 = ps.simulations.drainage_bf(im=im, steps=steps).im_size
        sizes4 = ps.simulations.drainage_dt_conv(im=im, steps=steps).im_size
        assert np.all(sizes1 == sizes2)
        assert np.all(sizes1 == sizes3)
        assert np.all(sizes1 == sizes4)

        seq1 = ps.simulations.drainage_dt(im=im, steps=steps).im_seq
        seq2 = ps.simulations.drainage_conv(im=im, steps=steps).im_seq
        seq3 = ps.simulations.drainage_bf(im=im, steps=steps).im_seq
        seq4 = ps.simulations.drainage_dt_conv(im=im, steps=steps).im_seq
        assert np.all(seq1 == seq2)
        assert np.all(seq1 == seq3)
        assert np.all(seq1 == seq4)

    def test_drainage_implementations_w_inlets(self):
        edt = ps.tools.get_edt()
        im = ps.generators.blobs(
            shape=[100, 100],
            porosity=0.7,
            blobiness=1.5,
        )

        # All methods are equivalent IF dt is integers
        dt = edt(im).astype(int)
        steps = np.unique(dt[im])
        faces = ps.generators.borders(im.shape, mode='faces')

        sizes1 = ps.simulations.drainage_dt(
            im=im, dt=dt, inlets=faces, steps=steps).im_size
        sizes2 = ps.simulations.drainage_conv(
            im=im, dt=dt, inlets=faces, steps=steps).im_size
        sizes3 = ps.simulations.drainage_bf(
            im=im, dt=dt, inlets=faces, steps=steps).im_size
        sizes4 = ps.simulations.drainage_dt_conv(
            im=im, dt=dt, inlets=faces, steps=steps).im_size
        assert np.all(sizes1 == sizes2)
        assert np.all(sizes1 == sizes3)
        assert np.all(sizes1 == sizes4)

        seq1 = ps.simulations.drainage_dt(
            im=im, dt=dt, inlets=faces, steps=steps).im_seq
        seq2 = ps.simulations.drainage_conv(
            im=im, dt=dt, inlets=faces, steps=steps).im_seq
        seq3 = ps.simulations.drainage_bf(
            im=im, dt=dt, inlets=faces, steps=steps).im_seq
        seq4 = ps.simulations.drainage_dt_conv(
            im=im, dt=dt, inlets=faces, steps=steps).im_seq
        assert np.all(seq1 == seq2)
        assert np.all(seq1 == seq3)
        assert np.all(seq1 == seq4)

    def test_drainage_equals_drainage_dt(self):
        edt = ps.tools.get_edt()
        im = ps.generators.blobs(
            shape=[100, 100],
            porosity=0.7,
            blobiness=1.5,
            # seed=16,
        )
        im = ps.filters.fill_invalid_pores(im)

        # All methods are equivalent IF dt is integers
        dt = edt(im).astype(int)
        pc = 2/dt
        pc[~im] = 0
        steps = np.unique(dt[im])

        faces = ps.generators.borders(im.shape, mode='faces')

        sizes1 = ps.simulations.drainage_dt(
            im=im, dt=dt, inlets=faces, steps=steps).im_size
        sizes2 = ps.simulations.drainage(
            im=im, dt=dt, pc=pc, inlets=faces, steps=(2/steps)[-1::-1]).im_size
        # There are sometimes a few pixels that don't agree, not sure why yet.
        # Use seed=16 in blobs to see
        assert np.sum(sizes1 != sizes2) < 10

    def test_drainage_bf_w_float_dt(self):
        edt = ps.tools.get_edt()
        im = ps.generators.blobs(
            shape=[100, 100],
            porosity=0.7,
            blobiness=1.5,
            seed=16,
        )

        dt = edt(im)
        sizes1 = ps.simulations.drainage_bf(
            im=im, dt=dt, steps=None).im_size
        # plt.imshow(sizes1)

    def test_imbibition_implementations_no_inlets(self):
        edt = ps.tools.get_edt()
        im = ps.generators.blobs(
            shape=[100, 100],
            porosity=0.7,
            blobiness=1.5,
            seed=16,
        )

        # All methods are equivalent IF dt is integers
        dt = edt(im).astype(int)
        steps = np.unique(dt[im])

        sizes1 = ps.simulations.imbibition_dt(im=im, dt=dt, steps=steps).im_size
        sizes2 = ps.simulations.imbibition_conv(im=im, dt=dt, steps=steps).im_size
        sizes3 = ps.simulations.imbibition_bf(im=im, dt=dt, steps=steps).im_size
        sizes4 = ps.simulations.imbibition_dt_conv(im=im, dt=dt, steps=steps).im_size
        assert np.all(sizes1 == sizes2)
        assert np.all(sizes1 == sizes3)
        assert np.all(sizes1 == sizes4)
        plt.imshow(sizes1 - sizes2)

        seq1 = ps.simulations.imbibition_dt(im=im, dt=dt, steps=steps).im_seq
        seq2 = ps.simulations.imbibition_conv(im=im, dt=dt, steps=steps).im_seq
        seq3 = ps.simulations.imbibition_bf(im=im, dt=dt, steps=steps).im_seq
        seq4 = ps.simulations.imbibition_dt_conv(im=im, dt=dt, steps=steps).im_seq
        assert np.all(seq1 == seq2)
        assert np.all(seq1 == seq3)
        assert np.all(seq1 == seq4)

        # Of if we specify integer steps
        steps = np.arange(1, 50)
        sizes1 = ps.simulations.imbibition_dt(im=im, steps=steps).im_size
        sizes2 = ps.simulations.imbibition_conv(im=im, steps=steps).im_size
        sizes3 = ps.simulations.imbibition_bf(im=im, steps=steps).im_size
        sizes4 = ps.simulations.imbibition_dt_conv(im=im, steps=steps).im_size
        assert np.all(sizes1 == sizes2)
        assert np.all(sizes1 == sizes3)
        assert np.all(sizes1 == sizes4)

        seq1 = ps.simulations.imbibition_dt(im=im, steps=steps).im_seq
        seq2 = ps.simulations.imbibition_conv(im=im, steps=steps).im_seq
        seq3 = ps.simulations.imbibition_bf(im=im, steps=steps).im_seq
        seq4 = ps.simulations.imbibition_dt_conv(im=im, steps=steps).im_seq
        assert np.all(seq1 == seq2)
        assert np.all(seq1 == seq3)
        assert np.all(seq1 == seq4)

    def test_imbibition_implementations_w_inlets(self):
        edt = ps.tools.get_edt()
        im = ps.generators.blobs(
            shape=[100, 100],
            porosity=0.7,
            blobiness=1.5,
            seed=16,
        )

        # All methods are equivalent IF dt is integers
        dt = edt(im).astype(int)
        steps = np.unique(dt[im])

        faces = ps.generators.borders(im.shape, mode='faces')

        sizes1 = ps.simulations.imbibition_dt(
            im=im, dt=dt, inlets=faces, steps=steps).im_size
        sizes2 = ps.simulations.imbibition_conv(
            im=im, dt=dt, inlets=faces, steps=steps).im_size
        sizes3 = ps.simulations.imbibition_bf(
            im=im, dt=dt, inlets=faces, steps=steps).im_size
        sizes4 = ps.simulations.imbibition_dt_conv(
            im=im, dt=dt, inlets=faces, steps=steps).im_size
        assert np.all(sizes1 == sizes2)
        assert np.all(sizes1 == sizes3)
        assert np.all(sizes1 == sizes4)

        seq1 = ps.simulations.imbibition_dt(
            im=im, dt=dt, inlets=faces, steps=steps).im_seq
        seq2 = ps.simulations.imbibition_conv(
            im=im, dt=dt, inlets=faces, steps=steps).im_seq
        seq3 = ps.simulations.imbibition_bf(
            im=im, dt=dt, inlets=faces, steps=steps).im_seq
        seq4 = ps.simulations.imbibition_dt_conv(
            im=im, dt=dt, inlets=faces, steps=steps).im_seq
        assert np.all(seq1 == seq2)
        assert np.all(seq1 == seq3)
        assert np.all(seq1 == seq4)

    def test_imbibition_equals_imbibition_dt(self):
        edt = ps.tools.get_edt()
        im = ps.generators.blobs(
            shape=[100, 100],
            porosity=0.7,
            blobiness=1.5,
            seed=16,
        )
        im = ps.filters.fill_invalid_pores(im)

        # All methods are equivalent IF dt is integers
        dt = edt(im).astype(int)
        steps = np.unique(dt[im])

        faces = ps.generators.borders(im.shape, mode='faces')

        seq1 = ps.simulations.imbibition_dt(
            im=im, dt=dt, inlets=faces, steps=steps).im_size
        seq2 = ps.simulations.imbibition(
            im=im, dt=dt, inlets=faces, steps=(2/steps)).im_size

        plt.imshow(seq1 - seq2)
        assert np.all(seq1 == seq2)


if __name__ == "__main__":
    self = IBOPTest()
    self.run_all()
