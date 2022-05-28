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
    v2_770=np.array([-453.37849012, -433.44711881, -479.38722157, -449.37963073,
       -419.36112578, -389.35626367, -479.3919599 , -449.37616406,
       -419.35697586, -389.37064485, -479.40300538, -449.38210468,
       -419.37276534, -389.41978253])
    v3_770=np.array([-373.8105493 , -375.41618152, -348.02002698, -348.01117266,
       -348.00931455, -347.99931539, -378.04217418, -378.03365908,
       -378.02616193, -378.00120274, -408.06008198, -408.04722395,
       -408.03075064, -407.99078552])
    
    return x_770,y_770,v2_770,v3_770
