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
    v2v3_770=np.array([[-415.069,-400.576],[-453.559,-373.814],[-434.083,-375.388],[-480.,-348.],[-450.,-348.],[-420.,-348.],[-390.,-348.],[-480.,-378.],[-450.,-378.],[-420,-378.],[-390.,-378.],[-480.,-408.],[-450.,-408.],[-420.,-408.],[-390.,-408.]])
    xy_770=np.array([[321.13,299.7],[688.5,511.5],[511.5,511.5],[948.18,724.94],[676.75,745.67],[404.81,767.77],[132.65,791.34],[923.52,455.40],[653.11,476.53],[382.37,498.57],[111.34,521.66],[899.64,184.81],[629.88,206.95],[360.00,229.12],[89.77,251.55]],dtype=np.float) + [5,1]
    # Note that we had to add 5,1 to Alistair's x,y locations because he uses 0-indexed science pixels,
    # not 1-indexed detector pixels like the SIAF does.


    x=xy_770[:,0]
    y=xy_770[:,1]
    v2=v2v3_770[:,0]
    v3=v2v3_770[:,1]

    return x,y,v2,v3
