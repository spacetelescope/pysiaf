************
pysiaf
************
Handling of Science Instrument Aperture Files (SIAF) for space telescopes

Functionalities
==========================
* Captures current PRD content, i.e. pysiaf includes a copy of the SIAF XML files. These are maintained to be synchronized with the PRD.
* Transformations between the SIAF frames (Detector, Science, Ideal, Telelescope/V) are pre-loaded and easily accessible.
* Tools for plotting, validation, and comparison of SIAF apertures and files.
* Support for implementing transformations between celestial (RA, Dec) and telescope/V (V2, V3) coordinate systems is provided.
* Input/output: reading SIAF XML, writing XML/Excel/csv etc.
* Captures SI source data and code to generate the SIAF apertures
* Standard python package with installation script, unit tests, documentation.
* Supports working with HST SIAF (read-only).


Where to Find pysiaf
==========================
pysiaf is hosted and developed at https://github.com/spacetelescope/pysiaf


Installation
==================
This package is being developed in a python 3.6 environment.

How to install
**************
pysiaf is available on `PyPI <https://pypi.org/project/pysiaf/>`_ and is included in astroconda.


pip install pysiaf

Clone the repository:
git clone https://github.com/spacetelescope/pysiaf
Install pysiaf:
cd pysiaf
python setup.py install or
pip install .

Known installation issue
************************

If you get an error upon ``import pysiaf`` that traces back to ``import lxml.etree as ET`` and states

``ImportError [...] Library not loaded: libxml2.2.dylib Reason: Incompatible library version: etree.[...] requires version 12.0.0 or later, but libxml2.2.dylib provides version 10.0.0``,

this can probably be fixed by downgrading the version of lxml, e.g.

``pip uninstall lxml``
``pip install lxml==3.6.4``


User Documentation
==================
Example usage:

Check which PRD version is in use:
``print(pysiaf.JWST_PRD_VERSION)``

Frame transformations (``det``, ``sci``, ``idl``, ``tel`` are supported frames)::

    import pysiaf
    instrument = 'NIRISS'

    # read SIAFXML
    siaf = pysiaf.Siaf(instrument)

    # select single aperture by name
    nis_cen = siaf['NIS_CEN']

    # access SIAF parameters
    print('{} V2Ref = {}'.format(nis_cen.AperName, nis_cen.V2Ref))
    print('{} V3Ref = {}'.format(nis_cen.AperName, nis_cen.V3Ref))

    for attribute in ['InstrName', 'AperShape']:
        print('{} {} = {}'.format(nis_cen.AperName, attribute, getattr(nis_cen, attribute)))


    # coordinates in Science frame
    sci_x = np.array([0, 2047, 2047, 0])
    sci_y = np.array([0, 0, 2047, 2047])

    # transform from Science frame to Ideal frame
    idl_x, idl_y = nis_cen.sci_to_idl(sci_x, sci_y)


Using sky transforms
********************
Transformations to/from ``sky`` coordinates (RA, Dec) are also supported, but you have to first define and set an
attitude matrix that represents the observatory orientation with respect to the celestial sphere. This can be done with
``pysiaf.utils.rotations.attitude_matrix``::

    # find attitude with some coordinates (v2,v3) pointed at (ra, dec) with a given pa
    attmat = pysiaf.utils.rotations.attitude_matrix(v2, v3, ra, dec, pa)

    # set that attitude for the transforms
    nis_cen.set_attitude_matrix(attmat)

    # transform from Science frame to Sky frame
    sky_ra, sky_dec = nis_cen.sci_to_sky(sci_x, sci_y)

Sky coordinates are given in units of degrees RA and Dec.

Reporting Issues / Contributing
===============================
Do you have feedback and feature requests? Is there something missing you would like to see? Please open a new issue or new pull request at https://github.com/spacetelescope/pysiaf for bugs, feedback, or new features you would like to see. If there is an issue you would like to work on, please leave a comment and we will be happy to assist. New contributions and contributors are very welcome! This package follows the STScI `Code of Conduct <https://github.com/spacetelescope/pysiaf/blob/master/CODE_OF_CONDUCT.md>`_ strives to provide a welcoming community to all of our users and contributors.

Coding and other guidelines
***************************
We strive to adhere to the `STScI Style Guides <https://github.com/spacetelescope/style-guides>`_.

How to make a code contribution
*******************************
The following describes the typical work flow for contributing to the pysiaf project (adapted from `<https://github.com/spacetelescope/jwql>`_):

#. Do not commit any sensitive information (e.g. STScI-internal path structures, machine names, user names, passwords, etc.) to this public repository. Git history cannot be erased.
#. Create a fork off of the ``spacetelescope`` ``pysiaf`` repository on your personal github space.
#. Make a local clone of your fork.
#. Ensure your personal fork is pointing ``upstream`` to https://github.com/spacetelescope/pysiaf
#. Open an issue on ``spacetelescope`` ``pysiaf`` that describes the need for and nature of the changes you plan to make. This is not necessary for minor changes and fixes.
#. Create a branch on that personal fork.
#. Make your software changes.
#. Push that branch to your personal GitHub repository, i.e. to ``origin``.
#. On the ``spacetelescope`` ``pysiaf`` repository, create a pull request that merges the branch into ``spacetelescope:master``.
#. Assign a reviewer from the team for the pull request, if you are allowed to do so.
#. Iterate with the reviewer over any needed changes until the reviewer accepts and merges your branch.
#. Delete your local copy of your branch.

Disclaimer
============
All parameter values in pysiaf are subject to change. JWST values are preliminary until the JWST observatory commissioning has concluded.

Distortion and other transformations in pysiaf are of sufficient accuracy for operations, but do not necessarily have science-grade quality. For instance, generally only one filter solution is carried per aperture.
For science-grade transformations, please consult the science pipelines and their reference files (see https://jwst-docs.stsci.edu/display/JDAT/JWST+Data+Reduction+Pipeline).

For science observation planning, the focal plane geometry implemented in the latest APT (http://www.stsci.edu/hst/proposing/apt) takes precedence.

The STScI Telescopes Branch provides full support of pysiaf for S&OC operational systems only.



Reference API
=============
.. toctree::
   :maxdepth: 1

   aperture.rst
   compare.rst
   polynomial.rst
   projection.rst
   read.rst
   rotations.rst
   siaf.rst
   tools.rst
   write.rst
