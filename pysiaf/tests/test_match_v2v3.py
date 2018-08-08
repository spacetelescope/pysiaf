from pysiaf import iando
from pysiaf.utils import tools


print('\n\n ************************************* BEGIN TEST **************************')
instrument = 'NIRCam'
apName_1 = 'NRCA1_FULL'
apName_2 = 'NRCA5_FULL'

siaf = iando.read.read_jwst_siaf(instrument=instrument)
aperture_1 = siaf[apName_1]
aperture_2 = siaf[apName_2]

new_aperture_2 = tools.match_v2v3(aperture_1, aperture_2, verbose=True)
assert new_aperture_2.V2Ref == aperture_1.V2Ref
assert new_aperture_2.V3Ref == aperture_1.V3Ref