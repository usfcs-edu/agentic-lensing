GIGA-Lens
========================

.. image:: https://img.shields.io/pypi/v/gigalens.svg
    :target: https://pypi.python.org/pypi/gigalens
    :alt: Latest PyPI version

Gradient Informed, GPU Accelerated Lens modelling (GIGA-Lens) is a package for fast Bayesian inference on strong gravitational lenses, with support for multi-device and multi-node GPU acceleration for large-scale inference workloads. For details, please see `our paper <https://arxiv.org/abs/2202.07663>`__. See
`here <https://giga-lens.github.io/gigalens/>`__ for our documentation.

Usage
-----
Note: Some ``.ipynb`` files cannot be run directly in the GitHub preview and must be downloaded to be opened in Jupyter Notebook or JupyterLab.

Installation
------------
``GIGA-Lens`` can be installed via pip: ::

    pip install gigalens[cuda]

If pip notes an error after installation about conflicting dependencies, these can usually be safely ignored.
If you wish to test the installation, tests can be run simply by running ``tox`` in the root directory.

If you don’t have access to institutional GPUs, one easy way is to use GPU on Google Colab.  Please remember the
very first cell should have ``!pip install gigalens[cuda]``. If you do have access to institutional GPUs, you can set up a
notebook to run on GPU.  For example, at `NESRC <https://jupyter.nersc.gov/hub/>`__, you can choose the kernel
``tensorflow-2.6.0``, and include in the first cell: ``!pip install gigalens[cuda]``.

Requirements
------------
Python Version >= 3.12

The following packages are requirements for GIGA-Lens. However, ``!pip install gigalens[cuda]`` is all you need to do. In fact,
separately installing other packages can cause issues with subpackage dependencies. Some users may find it necessary
to install PyYAML.

- ``jax==0.6.2``
- ``tensorflow-probability==0.25.0``
- ``lenstronomy>=1.13.2,<2.0.0``
- ``optax>=0.2.6,<0.3.0``
- ``objax>=1.8.0,<2.0.0``
- ``numpy==2.1.3``
- ``tqdm>=4.67.1,<5.0.0``

Authors
-------

`GIGALens` was written by `Andi Gu <andi.gu@berkeley.edu>`_.
