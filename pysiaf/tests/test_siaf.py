#!/usr/bin/env python
"""Tests for the pysiaf siaf classes."""

import pytest

from pysiaf import Siaf


def test_delete_aperture_single_name():
    """A single aperture name given as a string removes that aperture."""
    siaf = Siaf('nircam')
    n_apertures = len(siaf)
    assert 'NRCA1_FULL' in siaf.apernames

    siaf.delete_aperture('NRCA1_FULL')

    assert 'NRCA1_FULL' not in siaf.apernames
    assert len(siaf) == n_apertures - 1
    assert list(siaf.toc['AperName']) == list(siaf.apernames)


@pytest.mark.parametrize('container', [list, tuple])
def test_delete_aperture_several_names(container):
    """Several aperture names are removed in one call."""
    siaf = Siaf('nircam')
    n_apertures = len(siaf)
    names = container(['NRCA1_FULL', 'NRCA2_FULL'])

    siaf.delete_aperture(names)

    assert not set(names) & set(siaf.apernames)
    assert len(siaf) == n_apertures - 2
    assert list(siaf.toc['AperName']) == list(siaf.apernames)


def test_delete_aperture_unknown_name():
    """An unknown aperture name is reported with the full name."""
    siaf = Siaf('nircam')

    with pytest.raises(KeyError, match='NOT_AN_APERTURE'):
        siaf.delete_aperture('NOT_AN_APERTURE')
