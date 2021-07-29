#
"""
Reference data for testing the MIRI imager CDP-7 distortion solution.
This is for SIAF testing, and therefore includes only F770W reference data
using SIAF-convention x,y pixel conventions.

Note that since this uses 1-indexed SIAF convention, it will NOT give correct
results if passed into the 0-indexed JWST calibration pipeline or the miricoord package.

"""

import numpy as np

def siaf_testdata():
    # F770W tests
    v2v3_770=np.array([[-386.54862921,-430.28480771]])
    xy_770=np.array([[51,51]],dtype=np.float)
    # Note that we had to add 5,1 to Alistair's x,y locations because he uses 0-indexed science pixels,
    # not 1-indexed detector pixels like the SIAF does.


    x=xy_770[:,0]
    y=xy_770[:,1]
    v2=v2v3_770[:,0]
    v3=v2v3_770[:,1]

    return x,y,v2,v3
