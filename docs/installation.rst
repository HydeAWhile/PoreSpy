.. _installation:

############
Installation
############

PoreSpy depends Numpy, Scipy, Matplotlib, Scikit-Image, and their dependencies. There are several ways to get a working installation.  

Anaconda
========

The easiest way to get started is to install the `Anaconda
distribution <https://www.anaconda.com/products/individual#Downloads>`__. Be sure to get the **Python 3.10+ version**. Anaconda includes most of the dependencies that PoreSpy needs.

Once you've installed *Anaconda* you can then install ``porespy``. It is
available on `conda-forge <https://anaconda.org/conda-forge/porespy>`__
and can be installed by typing the following at the *conda* prompt:

   $ conda install -c conda-forge porespy $

uv
===

`uv` has rapidly become the most popular package manager for Python. You'll be able to find lots of blog articles extolling the virtues of `uv`, but its main feature is speed. Because of this, it encourages a different way to work.  With Anaconda, one might have just a few environments that get used with all projects. With `uv` the preferred way is to have a unique environment for each project you have because creating new environments and switching between them is essentially instantaneous. 

First you need to install `uv`.  This can be downloaded from `here <https://docs.astral.sh/uv/getting-started/installation>`_.  This will install `uv` on your system so it will be available from the terminal. Next you navigate to the folder where the project files will be stored, like scripts and tomograms. Then you create a virtual environment using `uv venv`. This adds a `.venv` folder to the project, where `uv` will store all its information. Finally, you run `uv pip install porespy`.  The first time you do this `uv` will download and compile a few things, which may take some time, but it will store all of this so subsequent usage will be much faster. 

Installing the dev version
##########################
If you want to use the latest features available on the `dev` branch which have not yet been officially released, you have two options:

The easy way
------------
If you're looking for an easy way to install the development version of
``porespy`` and use the latest features, you can install it using::

   $ pip install git+https://github.com/PMEAL/porespy.git@dev

.. warning::
   This approach is not recommended if you are a porespy contributor or
   want to frequently get new updates as they roll in. If you insist on
   using this approach, to get the latest version at any point, you
   need to first uninstall your porespy and then rerun the command above.

The hard (but correct) way
--------------------------
If you are a porespy contributor or want to easily get the new updates as
they roll in, you need to properly clone our repo and install it locally.
It's not as difficult as it sounds, just follow these steps:

Open up the terminal/cmd and ``cd`` to the directory you want to clone ``porespy``.

Clone the repo somewhere in your disk using::

   $ git clone https://github.com/PMEAL/porespy

``cd`` to the root folder of ``porespy``::

   $ cd porespy

Install ``porespy`` dependencies::

   $ conda install --file=requirements/conda.txt
   $ pip install -r requirements.txt

Install ``porespy`` in "editable" mode::

   $ pip install --no-deps -e .

Voila! You can now use the latest features available on the ``dev`` branch. To
keep your "local" ``porespy`` installation up to date, every now and then, ``cd``
to the root folder of ``porespy`` and pull the latest changes::

   $ git pull

.. warning::
   For the development version of ``porespy`` to work, you need to first remove
   the ``porespy`` that you've previously installed using ``pip`` or ``conda``.

Where's my ``conda`` prompt?
###################################
All the commands in this page need to be typed in the ``conda`` prompt.


.. tab-set::

   .. tab-item:: Windows

      On Windows you should have a shortcut to the "Anaconda prompt" in the
      Anaconda program group in the start menu. This will open a Windows
      command console with access to the Python features added by *conda*,
      such as installing things via ``conda``.

   .. tab-item:: Mac and Linux

      On Mac or Linux, you need to open a normal terminal window, then type
      ``source activate env`` where you replace ``env`` with the name of
      the environment you want to install PoreSpy. If you don't know what this
      means, then use ``source activate base``, which will install PoreSpy in
      the base environment which is the default.
