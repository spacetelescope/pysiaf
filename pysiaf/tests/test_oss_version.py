#!/usr/bin/env python
"""Tests for the pysiaf utils.tools functions.

Authors
-------

    Johannes Sahlmann

"""

from astropy.table import Table
import pytest

from ..utils.tools import select_oss_version
from ..iando.read import read_siaf_oss_version

INSTRUMENT_DATA = ["FGS", "MIRI", "NIRCAM", "NIRISS", "NIRSPEC"]

@pytest.mark.parametrize("instrument", INSTRUMENT_DATA)
def test_oss_instrument(instrument):
    oss_table = read_siaf_oss_version(instrument)
    for row in oss_table:
        assert row['InstrName'] == instrument

APERTURE_DATA = [
    ("FGS", "FGS2_SUB128DIAG", "8.4"),
    ("FGS", "J-FRAME", None),
    ("MIRI", "MIRIM_MASK1065", "8.4"),
    ("MIRI", "MIRIM_TA1550_LL", "8.4"),
    ("NIRCAM", "NRCA3_FP1", "8.4"),
    ("NIRCAM", "NRCA4_SUB64P", "11.1"),
    ("NIRCAM", "NRCA5_SUB160P", None),
    ("NIRISS", "NIS_AMI2", "8.4"),
    ("NIRISS", "NIS_SUBAMPCAL", "8.4"),
    ("NIRSPEC", "NRS_IFU_SLICE12", "8.4"),
    ("NIRSPEC", "NRS_VIGNETTED_MSA2", "8.4"),
]

@pytest.mark.parametrize("InstrName,AperName,OSS_Version", APERTURE_DATA)
def test_oss_instrument(InstrName, AperName, OSS_Version):
    oss_table = read_siaf_oss_version(InstrName)
    oss_version = select_oss_version(AperName, oss_table)
    if OSS_Version is not None:
        assert oss_version == OSS_Version
    else:
        assert isinstance(oss_version, Table)
