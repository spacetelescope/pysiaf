#
"""
Reference data for testing the MIRI imager FLT-1 distortion solution.
This is for SIAF testing, and therefore includes only F770W reference data
using SIAF-convention x,y pixel conventions.

Note that since this uses 1-indexed SIAF convention, it will NOT give correct
results if passed into the 0-indexed JWST calibration pipeline or the miricoord package.

"""

import numpy as np

def siaf_testdata():
    # F770W tests, xy are 0-indexed detector pixels, add 1 to convert to the 1-indexed
    # pixels used by SIAF
    x_770=np.array([692.5 , 511.5 , 948.18, 676.75, 404.81, 132.65, 923.52, 653.11,
       382.37, 111.34, 899.64, 629.88, 360.  ,  89.77])+1
    y_770=np.array([511.5 , 511.5 , 724.94, 745.67, 767.77, 791.34, 455.4 , 476.54,
       498.57, 521.66, 184.81, 206.95, 229.12, 251.55])+1
    v2_770=np.array([-453.34829012, -433.41691881, -479.35702157, -449.34943073,
        -419.33092578, -389.32606367, -479.3617599 , -449.34596406,
        -419.32677586, -389.34044485, -479.37280538, -449.35190468,
        -419.34256534, -389.38958253])
    v3_770=np.array([-373.6679493 , -375.27358152, -347.87742698, -347.86857266,
        -347.86671455, -347.85671539, -377.89957418, -377.89105908,
        -377.88356193, -377.85860274, -407.91748198, -407.90462395,
        -407.88815064, -407.84818552])
    
    return x_770,y_770,v2_770,v3_770
