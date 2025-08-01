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
            im=self.im2D, inlets=inlets, outlets=outlets, steps=None, smooth=False,
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
            seed=0,
        )

        for smooth in [True, False]:
            dt = edt(im)
            # All methods are equivalent IF steps are integers
            steps = np.unique(dt.astype(int)[im])

            sizes1 = ps.simulations.drainage_dt(
                im=im, dt=dt, steps=steps, smooth=smooth).im_size
            sizes2 = ps.simulations.drainage_conv(
                im=im, dt=dt, steps=steps, smooth=smooth).im_size
            sizes3 = ps.simulations.drainage_bf(
                im=im, dt=dt, steps=steps, smooth=smooth).im_size
            sizes4 = ps.simulations.drainage_dt_conv(
                im=im, dt=dt, steps=steps, smooth=smooth).im_size

            assert np.all(sizes1 == sizes2)
            assert np.all(sizes1 == sizes3)
            assert np.all(sizes1 == sizes4)

            seq1 = ps.simulations.drainage_dt(
                im=im, dt=dt, steps=steps, smooth=smooth).im_seq
            seq2 = ps.simulations.drainage_conv(
                im=im, dt=dt, steps=steps, smooth=smooth).im_seq
            seq3 = ps.simulations.drainage_bf(
                im=im, dt=dt, steps=steps, smooth=smooth).im_seq
            seq4 = ps.simulations.drainage_dt_conv(
                im=im, dt=dt, steps=steps, smooth=smooth).im_seq

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

        dt = edt(im)
        # All methods are equivalent IF steps are integers
        steps = np.unique(dt.astype(int)[im])
        faces = ps.generators.borders(im.shape, mode='faces')

        for smooth in [True, False]:

            sizes1 = ps.simulations.drainage_dt(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_size
            sizes2 = ps.simulations.drainage_conv(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_size
            sizes3 = ps.simulations.drainage_bf(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_size
            sizes4 = ps.simulations.drainage_dt_conv(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_size
            assert np.all(sizes1 == sizes2)
            assert np.all(sizes1 == sizes3)
            assert np.all(sizes1 == sizes4)

            seq1 = ps.simulations.drainage_dt(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_seq
            seq2 = ps.simulations.drainage_conv(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_seq
            seq3 = ps.simulations.drainage_bf(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_seq
            seq4 = ps.simulations.drainage_dt_conv(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_seq
            assert np.all(seq1 == seq2)
            assert np.all(seq1 == seq3)
            assert np.all(seq1 == seq4)

    def test_drainage_equals_drainage_dt_smooth(self):
        edt = ps.tools.get_edt()
        im = ps.generators.blobs(
            shape=[100, 100],
            porosity=0.7,
            blobiness=1.5,
            seed=16,
        )
        im = ps.filters.fill_invalid_pores(im)

        dt = edt(im)
        pc = 2/dt
        pc[~im] = 0
        steps = ps.tools.parse_steps(steps=13, vals=dt.astype(int), pad=(1, 0))
        steps[-1] = 0.5

        faces = ps.generators.borders(im.shape, mode='faces')

        sizes1 = ps.simulations.drainage_dt(
            im=im, dt=dt, inlets=faces, steps=steps, smooth=True).im_size
        sizes2 = ps.simulations.drainage(
            im=im, dt=dt, pc=pc, inlets=faces, steps=(2/steps), smooth=True).im_size
        assert np.sum(sizes1 != sizes2) == 0

        seq1 = ps.simulations.drainage_dt(
            im=im, dt=dt, inlets=faces, steps=steps, smooth=True).im_seq
        seq2 = ps.simulations.drainage(
            im=im, dt=dt, pc=pc, inlets=faces, steps=(2/steps), smooth=True).im_seq
        assert np.sum(seq1 != seq2) == 0

    def test_drainage_equals_drainage_dt_not_smooth(self):
        edt = ps.tools.get_edt()
        im = ps.generators.blobs(
            shape=[100, 100],
            porosity=0.7,
            blobiness=1.5,
            seed=16,
        )
        im = ps.filters.fill_invalid_pores(im)

        dt = edt(im)
        pc = 2/dt.astype(int)
        pc[~im] = 0
        steps = ps.tools.parse_steps(steps=13, vals=dt.astype(int), pad=(1, 0))
        # steps[-1] = 0.1

        faces = ps.generators.borders(im.shape, mode='faces')

        sizes1 = ps.simulations.drainage_dt(
            im=im,
            dt=dt.astype(int),
            inlets=faces,
            steps=steps,
            smooth=False,
        ).im_size
        pc_steps = 2/steps
        pc_steps[-1] = pc_steps[-2] * 10
        sizes2 = ps.simulations.drainage(
            im=im,
            dt=dt.astype(int),
            pc=pc,
            inlets=faces,
            steps=pc_steps,
            smooth=False,
        ).im_size
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(sizes1/im)
        ax[1].imshow(sizes2/im)
        assert np.sum(sizes1 != sizes2) == 0

        seq1 = ps.simulations.drainage_dt(
            im=im,
            dt=dt.astype(int),
            inlets=faces,
            steps=steps,
            smooth=False,
        ).im_seq
        seq2 = ps.simulations.drainage(
            im=im,
            dt=dt.astype(int),
            pc=pc,
            inlets=faces,
            steps=(2/steps),
            smooth=False,
        ).im_seq
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(seq1)
        ax[1].imshow(seq2)
        assert np.sum(seq1 != seq2) == 0

    def test_imbibition_implementations_no_inlets(self):
        edt = ps.tools.get_edt()
        im = ps.generators.blobs(
            shape=[100, 100],
            porosity=0.7,
            blobiness=1.5,
            seed=16,
        )

        # All methods are equivalent IF dt is integers
        dt = edt(im)
        steps = np.unique(dt[im].astype(int))
        for smooth in [True, False]:
            sizes1 = ps.simulations.imbibition_dt(
                im=im, dt=dt, steps=steps, smooth=smooth).im_size
            sizes2 = ps.simulations.imbibition_conv(
                im=im, dt=dt, steps=steps, smooth=smooth).im_size
            sizes3 = ps.simulations.imbibition_bf(
                im=im, dt=dt, steps=steps, smooth=smooth).im_size
            sizes4 = ps.simulations.imbibition_dt_conv(
                im=im, dt=dt, steps=steps, smooth=smooth).im_size
            assert np.all(sizes1 == sizes2)
            assert np.all(sizes1 == sizes3)
            assert np.all(sizes1 == sizes4)
            assert np.all(sizes2 == sizes3)
            assert np.all(sizes2 == sizes4)
            assert np.all(sizes3 == sizes4)


            seq1 = ps.simulations.imbibition_dt(
                im=im, dt=dt, steps=steps, smooth=smooth).im_seq
            seq2 = ps.simulations.imbibition_conv(
                im=im, dt=dt, steps=steps, smooth=smooth).im_seq
            seq3 = ps.simulations.imbibition_bf(
                im=im, dt=dt, steps=steps, smooth=smooth).im_seq
            seq4 = ps.simulations.imbibition_dt_conv(
                im=im, dt=dt, steps=steps, smooth=smooth).im_seq
            assert np.all(seq1 == seq2)
            assert np.all(seq1 == seq3)
            assert np.all(seq1 == seq4)
            assert np.all(seq2 == seq3)
            assert np.all(seq2 == seq4)
            assert np.all(seq3 == seq4)


    def test_imbibition_implementations_w_inlets(self):
        edt = ps.tools.get_edt()
        im = ps.generators.blobs(
            shape=[100, 100],
            porosity=0.7,
            blobiness=1.5,
            seed=16,
        )

        # All methods are equivalent IF dt is integers
        dt = edt(im)
        steps = np.unique(dt[im].astype(int))
        faces = ps.generators.borders(im.shape, mode='faces')

        for smooth in [True, False]:
            sizes1 = ps.simulations.imbibition_dt(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_size
            sizes2 = ps.simulations.imbibition_conv(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_size
            sizes3 = ps.simulations.imbibition_bf(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_size
            sizes4 = ps.simulations.imbibition_dt_conv(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_size
            assert np.all(sizes1 == sizes2)
            assert np.all(sizes1 == sizes3)
            assert np.all(sizes1 == sizes4)
            assert np.all(sizes2 == sizes3)
            assert np.all(sizes2 == sizes4)
            assert np.all(sizes3 == sizes4)

            seq1 = ps.simulations.imbibition_dt(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_seq
            seq2 = ps.simulations.imbibition_conv(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_seq
            seq3 = ps.simulations.imbibition_bf(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_seq
            seq4 = ps.simulations.imbibition_dt_conv(
                im=im, dt=dt, inlets=faces, steps=steps, smooth=smooth).im_seq
            assert np.all(seq1 == seq2)
            assert np.all(seq1 == seq3)
            assert np.all(seq1 == seq4)
            assert np.all(seq2 == seq3)
            assert np.all(seq2 == seq4)
            assert np.all(seq3 == seq4)

    def test_imbibition_equals_imbibition_dt(self):
        edt = ps.tools.get_edt()
        im = ps.generators.blobs(
            shape=[100, 100],
            porosity=0.7,
            blobiness=1.5,
            seed=16,
        )
        im = ps.filters.fill_invalid_pores(im)

        # All methods are equivalent IF steps integers
        dt = edt(im)
        steps = np.unique(dt[im].astype(int))

        faces = ps.generators.borders(im.shape, mode='faces')

        size1 = ps.simulations.imbibition_dt(
            im=im, dt=dt, inlets=faces, steps=steps, smooth=True).im_size
        size2 = ps.simulations.imbibition(
            im=im, dt=dt, inlets=faces, steps=(2/steps)).im_size
        # assert np.sum(size1 != size2) == 0

        seq1 = ps.simulations.imbibition_dt(
            im=im, dt=dt, inlets=faces, steps=steps, smooth=True).im_seq
        seq2 = ps.simulations.imbibition(
            im=im, dt=dt, inlets=faces, steps=(2/steps)).im_seq
        # assert np.sum(seq1 != seq2) == 0


if __name__ == "__main__":
    self = IBOPTest()
    self.run_all()
