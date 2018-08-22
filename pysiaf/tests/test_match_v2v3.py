from pysiaf import iando
from pysiaf.utils import tools


def test_match_v2v3(instrument='NIRCam', apName_1='NRCA1_FULL', apName_2='NRCA5_FULL', verbose=False):
    siaf = iando.read.read_jwst_siaf(instrument=instrument)
    aperture_1 = siaf[apName_1]
    aperture_2 = siaf[apName_2]
    new_aperture_2 = tools.match_v2v3(aperture_1, aperture_2, verbose=verbose)
    if verbose:
        print('Old VRef {:10.3f} {:10.3f}'.format(aperture_2.XDetRef, aperture_2.YDetRef))
        print('New VRef {:10.3f} {:10.3f}'.format(new_aperture_2.XDetRef, new_aperture_2.YDetRef))

    # Test that new VRef position has been installed
    assert new_aperture_2.V2Ref == aperture_1.V2Ref, 'V2Ref not in place'
    assert new_aperture_2.V3Ref == aperture_1.V3Ref, 'V3Ref not in place'

    # Test that detector to telescope frame relationship is unchanged
    xd = 100
    yd = 100
    (v2, v3) = aperture_2.convert(xd, yd, 'det', 'tel')
    (newv2, newv3) = new_aperture_2.convert(xd, yd, 'det', 'tel')
    assert abs(v2 - newv2) < 0.1, 'V2 positions do not agree'
    assert abs(v3 - newv3) < 0.1, 'V3 positions do not agree'
    return None

