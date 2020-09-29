"""

Tools to support the creation of an enhanced nircam siaf aperture definition file that can be used to verify subarray corners as well as reference point V2/V3 values.

Authors
-------
    Mario Gennaro

"""

#from __future__ import absolute_import, print_function, division
import os
from ..constants import JWST_SOURCE_DATA_ROOT, JWST_TEMPORARY_DATA_ROOT, JWST_DELIVERY_DATA_ROOT

def create_enhanced_aperture_file(aperture_dict,verbose=False):
    """
    Read the 'nircam_siaf_aperture_definition.txt' file and add extra columns to it (corner positions and V2/V3 of the reference point).
    
    Parameters
    ----------
    aperture_dict: dictionary of apertures created by generate_nircam starting from the 'nircam_siaf_aperture_definition.txt' file
    verbose: optional, boolean (default = False) to allow more diagnostic messages to be printed on the standard output
    
    Returns
    ------- 
    Nothing 
    
    """
    instrument = 'NIRCam'
    
    in_file = os.path.join(JWST_SOURCE_DATA_ROOT,instrument,'nircam_siaf_aperture_definition.txt')
    out_file = os.path.join(JWST_DELIVERY_DATA_ROOT, instrument,'nircam_enhanced_siaf_aperture_definition.txt')

    print('Writing enhanced SIAF aperture definition file including subarrays and V2/V3 of reference point')
    with open(in_file) as fp:
        lines = fp.readlines()
    
    with open(out_file, 'w') as the_file:
        
        for line in lines:
            
            newline = line[:]
    
            # If the original siaf file line starts with a comment or with a newline command (i.e. empty line), do not change it
            if (line[0] in ['#','\n']) == False:
                llist = line.split(',')
                aper_name = llist[0].strip()
    
                # The header line does not start with "#" neither it is empty. Yet we must add two header columns. 
                # The try statement fails because the first element oin the header row is not a valid aperture name, the exception is captured,
                # the header modified an the continue statment breaks this iteration of the for loop on the lines
                
                try:
                    ap = aperture_dict[aper_name]
                    
                except:
                    llist.insert(8,' {:>11} '.format('ColCorner'))
                    llist.insert(9,' {:>11} '.format('RowCorner'))
                    llist.insert(10,' {:>20} '.format('V2_ref'))
                    llist.insert(11,' {:>20} '.format('V3_ref'))
                    
                    newline = ','.join(llist)
                    the_file.write(newline)
                    continue
                
                else:
                    # If instead the try statement does not fail, execute this block
                    # This second try/except statement is necessary to capture the exception thrown by calling the corner method on apertures that do not have detector corners,      
                    # like the compoud and  grism_wfss ones. If the try fails, we add "None" for the OSS corner, otherwise we use the new DMS_corner method of the aperture object
                    try:
                        col_corner, row_corner = ap.dms_corner()
                           
                    except:
                        llist.insert(8,' {:>11} '.format('None'))
                        llist.insert(9,' {:>11} '.format('None'))
                    else:            
                        llist.insert(8,' {:>11} '.format(col_corner))
                        llist.insert(9,' {:>11} '.format(row_corner))
    
                    # This third try/except statement is necessary to capture the exception thrown by calling the reference point method on apertures that do not have one.      
                    # If the try fails, we add "None" for the V2/V3 ref, otherwise we use the reference_point('idl') method of the aperture object
                    try:
                        V2_ref, V3_ref = ap.reference_point('tel')
                           
                    except:
                        llist.insert(10,' {:>20} '.format('None'))
                        llist.insert(11,' {:>20} '.format('None'))
                    else:            
                        llist.insert(10,' {:20.14f} '.format(V2_ref))
                        llist.insert(11,' {:20.14f} '.format(V3_ref))
    
                newline = ','.join(llist)
                if verbose:
                    print(newline)
    
            the_file.write(newline)

