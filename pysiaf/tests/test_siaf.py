#!/usr/bin/env python
"""Tests for the pysiaf siaf classes."""

import pytest

from pysiaf import Siaf


@pytest.mark.parametrize('instrument', ['nircam', 'hst', 'roman'])
def test_list_apertures(instrument, capsys):
    """A Siaf read from disk has a table of contents and list_apertures prints it."""
    siaf = Siaf(instrument)

    assert list(siaf.toc['AperName']) == list(siaf.apernames)

    siaf.list_apertures()

    assert 'AperName' in capsys.readouterr().out


def test_list_apertures_selection(capsys):
    """The instrument and shape arguments restrict the printed list."""
    siaf = Siaf('hst')
    expected = {name for name, aperture in siaf.apertures.items()
                if aperture.InstrName == 'COS' and aperture.AperShape == 'CIRC'}
    assert 0 < len(expected) < len(siaf)

    siaf.list_apertures(instrument='COS', shape='CIRC')

    # astropy may shorten a long table, so check the rows that were printed
    printed = set(capsys.readouterr().out.split()) & set(siaf.apernames)
    assert len(printed) > 0
    assert printed <= expected
