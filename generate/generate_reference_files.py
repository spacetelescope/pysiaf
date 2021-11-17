"""Functions to generate SIAF reference files

These reference files are expected to be updated as a consequence of
commissioning and calibration activities. They have to provided pysiaf
with all necessary information to regenerate the complete information
contained in the SIAF files and with the correct formatting for
ingestion in the PRD.

Functions to read these reference are provided by iando.read


Authors
-------

    Johannes Sahlmann
    Mario Gennaro

References
----------

    Parts of generate_siaf_pre_flight_reference_files_niriss were
    adapted from Colin Cox' makeSIAF.py

    Parts of generate_siaf_pre_flight_reference_files_nircam were
    adapted from Colin Cox' nircamtrans.py

    nircam_get_polynomial_forward, nircam_get_polynomial_inverse,
    nircam_get_polynomial_both were adapted from from Colin Cox'
    nircamtrans.py

"""

import copy
import math
import os

import numpy as np
from astropy.table import Table, Column
from astropy.time import Time
import lxml.etree as ET

import pysiaf
from pysiaf.constants import JWST_SOURCE_DATA_ROOT, JWST_PRD_VERSION, JWST_DELIVERY_DATA_ROOT, JWST_TEMPORARY_DATA_ROOT
from pysiaf.utils import polynomial, tools
from pysiaf import iando

username = os.getlogin()
timestamp = Time.now()

def generate_fgs_fsw_coefficients(siaf=None, verbose=False, scale=0.06738281367):
    """Write the FGS distortion coefficients for the flight software to file.

    The FGS flight software (FSW) necessitates the distortion coefficients in a format that differs
    from the SIAF. In particular, the FSW coefficients are scale-corrected, i.e. they transform
    between ideal and distorted pixel coordinates. Second, the origin of the coordinate system
    used for transformation is at an aperture corner and not at the center like for the SIAF
    transformation.

    Parameters
    ----------
    siaf : pysiaf.Siaf instance
        The Siaf object to extract the coefficients from.
    verbose : bool
        verbosity
    scale : float
        Global pixel scale in arcsec/pixel

    """
    if siaf is None:
        siaf = pysiaf.Siaf('fgs')

    instrument = 'FGS'

    pre_delivery_dir = os.path.join(JWST_DELIVERY_DATA_ROOT, instrument)
    if not os.path.isdir(pre_delivery_dir):
        os.makedirs(pre_delivery_dir)

    for aperture_name in ['FGS1_FULL_OSS', 'FGS2_FULL_OSS']:

        aperture = siaf[aperture_name]

        # center_offset_x = 1023.5
        # center_offset_y = 1023.5
        center_offset_x = aperture.XSciRef - 1.
        center_offset_y = aperture.YSciRef - 1.

        if verbose:
            print('External scale {}'.format(scale))
            print(aperture.get_polynomial_scales())

        # get SIAF coefficients
        coefficients = aperture.get_polynomial_coefficients()

        ar = coefficients['Sci2IdlX']
        br = coefficients['Sci2IdlY']
        cr = coefficients['Idl2SciX']
        dr = coefficients['Idl2SciY']

        a_fsw, b_fsw, c_fsw, d_fsw = polynomial.rescale(ar, br, cr, dr, 1. / scale)
        factor = -1.

        if 'FGS1' in aperture_name:
            b_fsw *= -1
            c_fsw = polynomial.flip_y(c_fsw)
            d_fsw = polynomial.flip_y(d_fsw)

        a_fsw = polynomial.shift_coefficients(a_fsw, factor * center_offset_x,
                                              factor * center_offset_y)
        b_fsw = polynomial.shift_coefficients(b_fsw, factor * center_offset_x,
                                              factor * center_offset_y)
        c_fsw = polynomial.shift_coefficients(c_fsw, factor * center_offset_x,
                                              factor * center_offset_y)
        d_fsw = polynomial.shift_coefficients(d_fsw, factor * center_offset_x,
                                              factor * center_offset_y)

        a_fsw[0] += center_offset_x
        b_fsw[0] += center_offset_y
        c_fsw[0] += center_offset_x
        d_fsw[0] += center_offset_y

        # print FSW coefficients to screen
        fsw_coefficients = Table((c_fsw, d_fsw, a_fsw, b_fsw), names=(
         'IDEALPTOREALPXCOE', 'IDEALPTOREALPYCOE', 'REALPTOIDEALPXCOE', 'REALPTOIDEALPYCOE'))
        if verbose:
            fsw_coefficients.pprint()

        table = Table(names=('parameter_name', 'value'), dtype=(object, float))
        table.add_row(['XOFFSET', center_offset_x])
        table.add_row(['YOFFSET', center_offset_y])
        table.add_row(['PLATESCALE', scale])
        for colname in fsw_coefficients.colnames:
            for i in range(len(fsw_coefficients[colname])):
                table.add_row(['{}_{}'.format(colname, i), fsw_coefficients[colname][i]])
        table['parameter_name'] = np.array(table['parameter_name']).astype(str)

        # write to file
        fsw_distortion_file = os.path.join(pre_delivery_dir, 'ifgs{}_distortion_tbl.txt'.format(aperture_name[3]))
        comments = []
        comments.append('FGS distortion coefficients for FSW')
        comments.append('')
        comments.append('Derived from SIAF distortion coefficients.')
        comments.append('')
        comments.append('Generated {} {}'.format(timestamp.isot, timestamp.scale))
        comments.append('by {}'.format(username))
        comments.append('')
        table.meta['comments'] = comments
        formats={'parameter_name': '%-20s', 'value': '%+2.6e'}
        table.write(fsw_distortion_file, format='ascii.fixed_width',
                                         delimiter=',', delimiter_pad=' ', bookend=False,
                                         overwrite=True, formats=formats)


def generate_initial_siaf_aperture_definitions(instrument):
    """Write text file that contains all the necessary aperture information to generate the full SIAF given the
    necessary reference files (focal plane alignment, distortion) and auxiliary information (DDC mapping, wedge offsets, ...)

    This file also defines the order in which the apertures are presented.

    :param instrument:
    :return:
    """
    siaf_detector_layout = iando.read.read_siaf_detector_layout()

    prd_siaf = pysiaf.Siaf(instrument)
    siaf_definitions = Table()


    for attribute_name in 'AperName AperType XDetRef YDetRef XSciSize YSciSize XSciRef YSciRef'.split():
        siaf_definitions[attribute_name] = [getattr(prd_siaf[aperture_name], attribute_name) for aperture_name in prd_siaf.apertures]

    parent_apertures = [None]*len(siaf_definitions)
    dependency_type = [None]*len(siaf_definitions)

    if instrument == 'NIRISS':
        for i,aperture_name in enumerate(siaf_definitions['AperName']):
            if aperture_name != 'NIS_CEN':
                parent_apertures[i] = 'NIS_CEN'
            if '_OSS' in aperture_name:
                dependency_type[i] = 'oss_default'
            elif aperture_name != 'NIS_CEN':
                dependency_type[i] = 'default'

    elif instrument == 'FGS':
        for i,aperture_name in enumerate(siaf_definitions['AperName']):
            if aperture_name not in ['FGS1_FULL', 'FGS2_FULL']:
                parent_apertures[i] = '{}_FULL'.format(aperture_name.split('_')[0])
            else:
                dependency_type[i] = 'master'
            if '_OSS' in aperture_name:
                dependency_type[i] = 'oss_default'
            elif aperture_name not in ['FGS1_FULL', 'FGS2_FULL']:
                dependency_type[i] = 'default'

    elif instrument == 'NIRSpec':
        for i,aperture_name in enumerate(siaf_definitions['AperName']):
            if siaf_definitions['AperType'][i] == 'TRANSFORM':
                continue

            if '_OSS' in aperture_name:
                dependency_type[i] = 'oss_default'
                parent_apertures[i] = aperture_name.split('_OSS')[0]
            elif 'NRS_IFU' in aperture_name:
                parent_apertures[i] = 'NRS1_FULL'
                dependency_type[i] = 'default'
            elif aperture_name in ['NRS_S200B1_SLIT']:
                parent_apertures[i] = 'NRS2_FULL'
                dependency_type[i] = 'default'
            elif aperture_name in 'NRS_S200A1_SLIT NRS_S200A2_SLIT NRS_S400A1_SLIT NRS_S1600A1_SLIT NRS_FULL_IFU'.split():
                parent_apertures[i] = 'NRS1_FULL'
                dependency_type[i] = 'default'
            elif ('_MSA1' in aperture_name) or ('_MSA2' in aperture_name):
                parent_apertures[i] = 'NRS2_FULL'
                dependency_type[i] = 'default'
            elif ('_MSA3' in aperture_name) or ('_MSA4' in aperture_name):
                parent_apertures[i] = 'NRS1_FULL'
                dependency_type[i] = 'default'
            elif aperture_name in ['NRS1_FP1MIMF']:
                parent_apertures[i] = 'NRS_S1600A1_SLIT'
                dependency_type[i] = 'FP1MIMF'
            elif 'MIMF' in aperture_name:
                parent_apertures[i] = aperture_name.split('_')[0]+'_FULL'
                dependency_type[i] = 'default'


                # if aperture_name not 'NIS_CEN':
            #     parent_apertures[i] = 'NIS_CEN'
            # elif aperture_name != 'NIS_CEN':
            #     dependency_type[i] = 'default'

    elif instrument == 'NIRCam':
        for i,aperture_name in enumerate(siaf_definitions['AperName']):

            # Master apertures
            if aperture_name in siaf_detector_layout['AperName']:
                dependency_type[i] = 'master'

            elif siaf_definitions['AperType'][i] in ['SUBARRAY', 'FULLSCA', 'ROI']:
                if 'MASK' in aperture_name:
                    # Coronagraphic apertures with wedge offset
                    dependency_type[i] = 'wedge'
                elif 'DHSPIL_WEDGES' in aperture_name:
                    dependency_type[i] = 'dhspil_wedge'
                else:
                    dependency_type[i] = 'default'
                sca_name = aperture_name[0:5]
                parent_apertures[i] = '{}_FULL'.format(sca_name)

            # OSS apertures
            elif siaf_definitions['AperType'][i] in ['OSS']:
                dependency_type[i] = 'default'
                parent_apertures[i] = aperture_name.split('_OSS')[0]

            elif (siaf_definitions['AperType'][i] in ['SLIT', 'COMPOUND']) and ('GRISM' in aperture_name) and ('WFSS' in aperture_name):
                dependency_type[i] = 'grism_wfss'
                sca_name = aperture_name.split('_')[0]
                # sca_name = aperture_name[0:5]
                parent_apertures[i] = '{}_FULL'.format(sca_name)

            # elif 'MASK' in aperture_name:
            #     dependency_type[i] = 'wedge'
            #     sca_name = aperture_name[0:5]
            #     parent_apertures[i] = '{}_FULL'.format(sca_name)



            elif aperture_name in 'NRCALL_FULL NRCAS_FULL NRCBS_FULL'.split():
                dependency_type[i] = 'nircam_compound'
                if aperture_name == 'NRCALL_FULL':
                    parent_apertures[i] = '; '.join(['NRCA1_FULL', 'NRCB2_FULL', 'NRCB1_FULL', 'NRCA2_FULL'])
                elif aperture_name == 'NRCAS_FULL':
                    parent_apertures[i] = '; '.join(['NRCA1_FULL', 'NRCA3_FULL', 'NRCA4_FULL', 'NRCA2_FULL'])
                elif aperture_name == 'NRCBS_FULL':
                    parent_apertures[i] = '; '.join(['NRCB4_FULL', 'NRCB2_FULL', 'NRCB1_FULL', 'NRCB3_FULL'])



    siaf_definitions['parent_apertures'] = parent_apertures
    siaf_definitions['dependency_type'] = dependency_type

    siaf_definitions.pprint()


    siaf_definitions_file_name = os.path.join(JWST_SOURCE_DATA_ROOT, instrument,
                                              '{}_siaf_aperture_definition.txt'.format(instrument.lower()))

    comments = []
    comments.append('{} aperture definition file for SIAF'.format(instrument))
    comments.append('')
    comments.append('This file contains all the necessary aperture information to generate the full SIAF given the necessary reference files (focal plane alignment, distortion) and auxiliary information (DDC mapping, wedge offsets, ...)')
    comments.append('This file also defines the order in which the apertures are presented.')
    comments.append('')
    comments.append('Originally based on {}.'.format(JWST_PRD_VERSION))
    comments.append('')
    comments.append('Generated {} {}'.format(timestamp.isot, timestamp.scale))
    comments.append('{}'.format(username))
    comments.append('')
    siaf_definitions.meta['comments'] = comments
    siaf_definitions.write(siaf_definitions_file_name, format='ascii.fixed_width', delimiter=',',
                                     delimiter_pad=' ', bookend=False)


def generate_siaf_ddc_mapping_reference_file(instrument):
    """
        # DDC mapping

    :param instrument:
    :return:
    """
    ddc_mapping_file = os.path.join(JWST_SOURCE_DATA_ROOT, instrument,
                                    '{}_siaf_ddc_apername_mapping.txt'.format(instrument.lower()))
    if instrument == 'NIRISS':
        _ddc_apername_mapping = {
            'NIS_CEN': 'NIS_CNTR',
            'NIS_AMI1': 'NIS_AMI'}

    elif instrument == 'FGS':
        _ddc_apername_mapping = {
            'FGS1_FULL': 'GUIDER1_CNTR',
            'FGS2_FULL': 'GUIDER2_CNTR'}

    elif instrument == 'NIRSpec':
        _ddc_apername_mapping = {
            'NRS_FULL_MSA': 'NRS_CNTR',
            'NRS1_FULL': 'NRS1_CNTR',
            'NRS2_FULL': 'NRS2_CNTR'}

    elif instrument == 'MIRI':
        _ddc_apername_mapping = {
            'MIRIM_ILLUM': 'MIRIMAGE_ILLCNTR',
            'MIRIM_MASK1550': 'MIRIMAGE_MASK1550',
            'MIRIM_MASK1140': 'MIRIMAGE_MASK1140',
            'MIRIM_MASK1065': 'MIRIMAGE_MASK1065',
            'MIRIM_MASKLYOT': 'MIRIMAGE_MASKLYOT',
            'MIRIFU_CHANNEL2B': 'MIRIFU_CNTR'}

    elif instrument == 'NIRCam':
        # dictionary that maps SIAF aperture names to DDC aperture names
        _ddc_apername_mapping = {
            'NRCAS_FULL': 'NRCA_CNTR',
            'NRCBS_FULL': 'NRCB_CNTR',
            'NRCALL_FULL': 'NRCALL_CNTR',
            'NRCA5_MASK430R': 'NRC_MASK430R',
            'NRCA5_MASKLWB': 'NRC_MASKLWB',
            'NRCA4_MASKSWB': 'NRC_MASKSWB',
            'NRCA5_MASK335R': 'NRC_MASK335R',
            'NRCA2_MASK210R': 'NRC_MASK210R'
        }

    ddc_mapping_table = Table()
    ddc_mapping_table['SIAF_NAME'] = [key for key,value in _ddc_apername_mapping.items()]
    ddc_mapping_table['DDC_NAME'] = [value for key,value in _ddc_apername_mapping.items()]

    comments = []
    comments.append('{} DDC mapping definition file for SIAF'.format(instrument))
    comments.append('')
    comments.append('This file contains the DDC aperture mapping.')
    comments.append('')
    comments.append('Generated {} {}'.format(timestamp.isot, timestamp.scale))
    comments.append('by {}'.format(username))
    comments.append('')
    ddc_mapping_table.meta['comments'] = comments
    ddc_mapping_table.write(ddc_mapping_file, format='ascii.fixed_width', delimiter=',',
                                     delimiter_pad=' ', bookend=False, overwrite=True)


def generate_siaf_detector_layout():
    """Write reference file that specifies the orientations between SIAF frames for the full frame standard apertures,
    e.g between the SIAF Detector and SIAF Science frame. These apertures act as parent apertures of all other SI
    apertures and their parameters are thus inherited.

    Note that the DMS Detector frame differs in its definition from the SIAF Detector frame. What commonly is referred
    to as 'DMS coordinate system' corresponds to the SIAF Science frame.

    Updated by M.Gennaro to allow for coronographic apertures support in NIRCam

    :return:
    """

    VIdlParity = -1
    layout = Table(dtype=['S100', 'S100', 'f4', 'i4', 'i4'] ,names=('InstrName', 'AperName', 'DetSciYAngle', 'DetSciParity', 'VIdlParity'))
    for instrument in 'NIRCam FGS NIRISS NIRSpec MIRI'.split():
        if instrument == 'NIRCam':
            for sca_name in 'A1 A3 A5 B2 B4'.split():
                layout.add_row([instrument.upper(), 'NRC{}_FULL'.format(sca_name), 0, -1, VIdlParity])
            for sca_name in 'A2 A4 B1 B3 B5'.split():
                layout.add_row([instrument.upper(), 'NRC{}_FULL'.format(sca_name), 180, -1, VIdlParity])
            for sca_name in ['NRCA2_FULL_WEDGE_RND','NRCA2_FULL_WEDGE_BAR','NRCA4_FULL_WEDGE_RND','NRCA4_FULL_WEDGE_BAR']:
                layout.add_row([instrument.upper(), '{}'.format(sca_name), 180, -1, VIdlParity])
            for sca_name in ['NRCA1_FULL_WEDGE_RND','NRCA1_FULL_WEDGE_BAR','NRCA3_FULL_WEDGE_RND','NRCA3_FULL_WEDGE_BAR','NRCA5_FULL_WEDGE_RND','NRCA5_FULL_WEDGE_BAR']:
                layout.add_row([instrument.upper(), '{}'.format(sca_name), 0, -1, VIdlParity])
        elif instrument == 'NIRISS':
            for sca_name in ['NIS_CEN']:
                layout.add_row([instrument, sca_name, 180, 1, VIdlParity])
        elif instrument == 'MIRI':
            for sca_name in ['MIRIM_FULL']:
                layout.add_row([instrument, sca_name, 0, 1, VIdlParity])
        elif instrument == 'NIRSpec':
            for sca_name in ['NRS1_FULL']:
                layout.add_row([instrument.upper(), sca_name, 0, 1, VIdlParity])
            for sca_name in ['NRS2_FULL']:
                layout.add_row([instrument.upper(), sca_name, 180, 1, VIdlParity])
        elif instrument == 'FGS':
            for sca_name in ['FGS1_FULL']:
                layout.add_row([instrument, sca_name, 180, 1, VIdlParity])
            for sca_name in ['FGS2_FULL']:
                layout.add_row([instrument, sca_name, 0, -1, VIdlParity])

    layout_file = os.path.join(JWST_SOURCE_DATA_ROOT, 'siaf_detector_layout.txt')

    layout.pprint()

    comments = []
    comments.append('SIAF detector layout definition file.'.format(instrument))
    comments.append('')
    comments.append('These apertures act as parent apertures of all other SI apertures and their parameters are thus inherited.')
    comments.append('')
    comments.append('Generated {} {}'.format(timestamp.isot, timestamp.scale))
    comments.append('by {}'.format(username))
    comments.append('')
    layout.meta['comments'] = comments
    layout.write(layout_file, format='ascii.fixed_width', delimiter=',',
                                     delimiter_pad=' ', bookend=False)


def generate_siaf_detector_reference_file(instrument):
    """

    :param instrument:
    :return:
    """

    configuration = Table()
    configuration['InstrName'] = [instrument.upper()] # to conform with SIAFXML convention = all uppercase
    configuration['AperShape'] = ['QUAD']

    if instrument in ['NIRISS', 'FGS', 'NIRCam', 'NIRSpec']:
        configuration['XDetSize'] = 2048
        configuration['YDetSize'] = 2048
    if instrument == 'MIRI':
        configuration['XDetSize'] = 1032
        configuration['YDetSize'] = 1024

    configuration_file = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, '{}_siaf_detector_parameters.txt'.format(instrument.lower()))

    comments = []
    comments.append('{} detector parameter definition file for SIAF'.format(instrument))
    comments.append('')
    comments.append('This file contains the basic detector characteristics.')
    comments.append('')
    comments.append('Generated {} {}'.format(timestamp.isot, timestamp.scale))
    comments.append('by {}'.format(username))
    comments.append('')
    configuration.meta['comments'] = comments
    print('Writing to {}'.format(configuration_file))
    configuration.write(configuration_file, format='ascii.fixed_width', delimiter=',',
                                     delimiter_pad=' ', bookend=False)


# NIRCam reference files
def generate_siaf_pre_flight_reference_files_nircam():
    """Generate NIRCam distortion and alignment reference files.

    Written by Johannes Sahlmann 2018-02-18

    This function was written on the basis of nircamtrans.py:
        Created by Colin Cox on 2015-02-02.
        Read in transformation coefficients from Randal Telfer
        Combine shift, add and distortion steps into single set of coefficients
        trans does the forward   detector(x,y) to V2V3
        inverse does V2V3 to detector(x,y)
     
    Updated by M.Gennaro to allow for coronographic master apertures support
    
    :return:
    """

    instrument = 'NIRCam'
    overwrite_wedge_file = False
    overwrite_grism_file = False


    # wedge definitions
    wedge_file = os.path.join(JWST_SOURCE_DATA_ROOT, instrument,
                              '{}_siaf_wedge_offsets.txt'.format(instrument.lower()))

    if (not os.path.isfile(wedge_file) or (overwrite_wedge_file)):

        wedge_offsets = Table.read(os.path.join(JWST_SOURCE_DATA_ROOT, instrument, 'wedge_offsets.txt'), format='ascii.basic', delimiter=' ', guess=False)

        comments = []
        comments.append('{} detector parameter definition file for SIAF'.format(instrument))
        comments.append('')
        comments.append('This file contains the wedge offsets.')
        comments.append('')
        comments.append('Generated {} {}'.format(timestamp.isot, timestamp.scale))
        comments.append('by {}'.format(username))
        comments.append('')
        wedge_offsets.meta['comments'] = comments
        wedge_offsets.write(wedge_file, format='ascii.fixed_width', delimiter=',',
                                         delimiter_pad=' ', bookend=False)

    # grism definitions
    grism_file = os.path.join(JWST_SOURCE_DATA_ROOT, instrument,
                              '{}_siaf_grism_parameters.txt'.format(instrument.lower()))

    if (not os.path.isfile(wedge_file) or (overwrite_grism_file)):
        # grism parameters,     see WFSS worksheet in EXCEL SIAF
        grism_parameters = Table.read(grism_file, format='ascii.basic', delimiter=',', guess=False)

        # Save a backup copy of the grism file
        cmd = 'cp {} {}'.format(grism_file,os.path.join(JWST_TEMPORARY_DATA_ROOT, instrument, 'nircam_siaf_grism_parameters_backup.txt'))
        os.system(cmd)

        # different sign in Y for NRCB apertures
        factor = np.array(
            [1. if 'NRCA' in grism_parameters['aperture_name'][i] else -1. for i in range(len(grism_parameters))])

        for col in grism_parameters.colnames[1:]:
            # these are Sci coordinates
            if col[0] != 'D':
                if 'X' in col:
                    grism_parameters['D{}'.format(col)] = grism_parameters[col].data - 1024.5
                elif 'Y' in col:
                    grism_parameters['D{}'.format(col)] = factor * (grism_parameters[col].data - 1024.5)



        comments = []
        comments.append('{} grism parameter definition file for SIAF'.format(instrument))
        comments.append('')
        comments.append('This file contains the grism parameters.')
        comments.append('')
        comments.append('Generated {} {}'.format(timestamp.isot, timestamp.scale))
        comments.append('by {}'.format(username))
        comments.append('')
        grism_parameters.meta['comments'] = comments
        grism_parameters.write(grism_file, format='ascii.fixed_width', delimiter=',',
                                         delimiter_pad=' ', bookend=False)

    # Transformation parameters, mapping used to select rows in cold_fit_[] file
    coldfit_name_mapping = {
                             'NRCA1_FULL': {'degrees_to_mm':'OTESKYToNIRCAMASW_201609161431',
                                             'mm_to_pixels':'NIRCAMASWToNIRCAMASW_1_20161025081540',
                                             'pixels_to_mm':'NIRCAMASW_1ToNIRCAMASW_20161025081540',
                                            'mm_to_degrees':'NIRCAMASWoOTESKY_201609161431',
                                             },
                             'NRCA2_FULL': {'degrees_to_mm':'OTESKYToNIRCAMASW_201609161431',
                                             'mm_to_pixels':'NIRCAMASWToNIRCAMASW_2_20161025081547',
                                             'pixels_to_mm':'NIRCAMASW_2ToNIRCAMASW_20161025081547',
                                            'mm_to_degrees':'NIRCAMASWoOTESKY_201609161431',
                                             },
                             'NRCA3_FULL': {'degrees_to_mm':'OTESKYToNIRCAMASW_201609161431',
                                             'mm_to_pixels':'NIRCAMASWToNIRCAMASW_3_20161025081552',
                                             'pixels_to_mm':'NIRCAMASW_3ToNIRCAMASW_20161025081552',
                                            'mm_to_degrees':'NIRCAMASWoOTESKY_201609161431',
                                             },
                             'NRCA4_FULL': {'degrees_to_mm':'OTESKYToNIRCAMASW_201609161431',
                                             'mm_to_pixels':'NIRCAMASWToNIRCAMASW_4_20161025081557',
                                             'pixels_to_mm':'NIRCAMASW_4ToNIRCAMASW_20161025081557',
                                            'mm_to_degrees':'NIRCAMASWoOTESKY_201609161431',
                                             },
                             'NRCA5_FULL' :{'degrees_to_mm':'OTESKYToNIRCAMALW_RT_20170307121022',
                                             'mm_to_pixels':'NIRCAMALWToNIRCAMALW_1_20161227162042',
                                             'pixels_to_mm':'NIRCAMALW_1ToNIRCAMALW_20161227162042',
                                            'mm_to_degrees':'NIRCAMALWToOTESKY_RT_20170307121022',
                                             },
                             'NRCB1_FULL': {'degrees_to_mm':'OTESKYToNIRCAMBSW_RT_20170307121023',
                                             'mm_to_pixels':'NIRCAMBSWToNIRCAMBSW_1_20161025081604',
                                             'pixels_to_mm':'NIRCAMBSW_1ToNIRCAMBSW_20161025081604',
                                            'mm_to_degrees':'NIRCAMBSWToOTESKY_RT_20170307121024',
                                             },
                             'NRCB2_FULL': {'degrees_to_mm':'OTESKYToNIRCAMBSW_RT_20170307121023',
                                             'mm_to_pixels':'NIRCAMBSWToNIRCAMBSW_2_20161025081912',
                                             'pixels_to_mm':'NIRCAMBSW_2ToNIRCAMBSW_20161025081912',
                                            'mm_to_degrees':'NIRCAMBSWToOTESKY_RT_20170307121024',
                                             },
                             'NRCB3_FULL': {'degrees_to_mm':'OTESKYToNIRCAMBSW_RT_20170307121023',
                                             'mm_to_pixels':'NIRCAMBSWToNIRCAMBSW_3_20161025082300',
                                             'pixels_to_mm':'NIRCAMBSW_3ToNIRCAMBSW_20161025082300',
                                            'mm_to_degrees':'NIRCAMBSWToOTESKY_RT_20170307121024',
                                             },
                             'NRCB4_FULL': {'degrees_to_mm':'OTESKYToNIRCAMBSW_RT_20170307121023',
                                             'mm_to_pixels':'NIRCAMBSWToNIRCAMBSW_4_20161025082647',
                                             'pixels_to_mm':'NIRCAMBSW_4ToNIRCAMBSW_20161025082647',
                                            'mm_to_degrees':'NIRCAMBSWToOTESKY_RT_20170307121024',
                                             },
                             'NRCB5_FULL' :{'degrees_to_mm':'OTESKYToNIRCAMBLW_RT_20170307121023',
                                             'mm_to_pixels':'NIRCAMBLWToNIRCAMBLW_1_20161227162336',
                                             'pixels_to_mm':'NIRCAMBLW_1ToNIRCAMBLW_20161227162336',
                                            'mm_to_degrees':'NIRCAMBLWToOTESKY_RT_20170307121023',
                                             },
                   'NRCA1_FULL_WEDGE_RND' :{'degrees_to_mm':'OTESKYToNIRCAMASW_RNDNC_202110261138',
                                             'mm_to_pixels':'NIRCAMASWToNIRCAMASW_1_20161025081540',
                                             'pixels_to_mm':'NIRCAMASW_1ToNIRCAMASW_20161025081540',
                                            'mm_to_degrees':'NIRCAMASW_RNDNCToOTESKY_202110261138',
                                             },
                   'NRCA2_FULL_WEDGE_RND' :{'degrees_to_mm':'OTESKYToNIRCAMASW_RND_202005150434',
                                             'mm_to_pixels':'NIRCAMASWToNIRCAMASW_2_20161025081547',
                                             'pixels_to_mm':'NIRCAMASW_2ToNIRCAMASW_20161025081547',
                                            'mm_to_degrees':'NIRCAMASW_RNDToOTESKY_202005150434',
                                             },
                   'NRCA3_FULL_WEDGE_RND' :{'degrees_to_mm':'OTESKYToNIRCAMASW_RNDNC_202110261138',
                                             'mm_to_pixels':'NIRCAMASWToNIRCAMASW_3_20161025081552',
                                             'pixels_to_mm':'NIRCAMASW_3ToNIRCAMASW_20161025081552',
                                            'mm_to_degrees':'NIRCAMASW_RNDNCToOTESKY_202110261138',
                                             },
                   'NRCA4_FULL_WEDGE_RND' :{'degrees_to_mm':'OTESKYToNIRCAMASW_RND_202005150434',
                                             'mm_to_pixels':'NIRCAMASWToNIRCAMASW_4_20161025081557',
                                             'pixels_to_mm':'NIRCAMASW_4ToNIRCAMASW_20161025081557',
                                            'mm_to_degrees':'NIRCAMASW_RNDToOTESKY_202005150434',
                                             },
                   'NRCA1_FULL_WEDGE_BAR' :{'degrees_to_mm':'OTESKYToNIRCAMASW_BARNC_202110261138',
                                             'mm_to_pixels':'NIRCAMASWToNIRCAMASW_1_20161025081540',
                                             'pixels_to_mm':'NIRCAMASW_1ToNIRCAMASW_20161025081540',
                                            'mm_to_degrees':'NIRCAMASW_BARNCToOTESKY_202110261138',
                                             },
                   'NRCA2_FULL_WEDGE_BAR' :{'degrees_to_mm':'OTESKYToNIRCAMASW_BAR_202005150434',
                                             'mm_to_pixels':'NIRCAMASWToNIRCAMASW_2_20161025081547',
                                             'pixels_to_mm':'NIRCAMASW_2ToNIRCAMASW_20161025081547',
                                            'mm_to_degrees':'NIRCAMASW_BARToOTESKY_202005150434',
                                             },
                   'NRCA3_FULL_WEDGE_BAR' :{'degrees_to_mm':'OTESKYToNIRCAMASW_BARNC_202110261138',
                                             'mm_to_pixels':'NIRCAMASWToNIRCAMASW_3_20161025081552',
                                             'pixels_to_mm':'NIRCAMASW_3ToNIRCAMASW_20161025081552',
                                            'mm_to_degrees':'NIRCAMASW_BARNCToOTESKY_202110261138',
                                             },
                   'NRCA4_FULL_WEDGE_BAR' :{'degrees_to_mm':'OTESKYToNIRCAMASW_BAR_202005150434',
                                             'mm_to_pixels':'NIRCAMASWToNIRCAMASW_4_20161025081557',
                                             'pixels_to_mm':'NIRCAMASW_4ToNIRCAMASW_20161025081557',
                                            'mm_to_degrees':'NIRCAMASW_BARToOTESKY_202005150434',
                                             },
                   'NRCA5_FULL_WEDGE_RND' :{'degrees_to_mm':'OTESKYToNIRCAMALW_RND_202005150434',
                                             'mm_to_pixels':'NIRCAMALWToNIRCAMALW_1_20161227162042',
                                             'pixels_to_mm':'NIRCAMALW_1ToNIRCAMALW_20161227162042',
                                            'mm_to_degrees':'NIRCAMALW_RNDToOTESKY_202005150434',
                                             },
                   'NRCA5_FULL_WEDGE_BAR' :{'degrees_to_mm':'OTESKYToNIRCAMALW_BAR_202005150434',
                                             'mm_to_pixels':'NIRCAMALWToNIRCAMALW_1_20161227162042',
                                             'pixels_to_mm':'NIRCAMALW_1ToNIRCAMALW_20161227162042',
                                            'mm_to_degrees':'NIRCAMALW_BARToOTESKY_202005150434',
                                             }
    }

    # coldfit_source_data_file = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, '{}'.format('cold_fit_201703071210.csv'))
    coldfit_source_data_file = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, '{}'.format('nircam_cold_fit.txt'))
    print('NIRCam coldfit data from', coldfit_source_data_file)
    t = open(coldfit_source_data_file)
    coldfit_source_data = t.readlines()
    t.close()
    # remove comments from read content
    coldfit_source_data = [line for line in coldfit_source_data if line[0] != '#']

    siaf_detector_layout = iando.read.read_siaf_detector_layout()
    # siaf_alignment_parameters = iando.read.read_siaf_alignment_parameters(instrument)
    siaf_aperture_definitions = iando.read.read_siaf_aperture_definitions(instrument)
    # aperture_dict = {}
    aperture_name_list = siaf_aperture_definitions['AperName'].tolist()

    # generate alignment reference file, one file for all master apertures
    outfile = os.path.join(JWST_SOURCE_DATA_ROOT, instrument,
                           '{}_siaf_alignment.txt'.format(instrument.lower()))
    siaf_alignment = Table()

    for AperName in aperture_name_list:

        # process the master apertures of NIRCam
        if AperName in siaf_detector_layout['AperName']:
            (A, B, C, D, betaX, betaY, V2Ref, V3Ref) = nircam_get_polynomial_both(AperName, siaf_aperture_definitions, coldfit_name_mapping, coldfit_source_data)

            #generate distortion reference file
            number_of_coefficients = len(A)
            polynomial_degree = np.int((np.sqrt(8 * number_of_coefficients + 1) - 3) / 2)
            siaf_index = []
            exponent_x = []
            exponent_y = []
            for i in range(polynomial_degree + 1):
                for j in np.arange(i + 1):
                    siaf_index.append('{:d}{:d}'.format(i, j))
                    exponent_x.append(i - j)
                    exponent_y.append(j)

            distortion_reference_table = Table((siaf_index, exponent_x, exponent_y, A, B, C, D), names=(
            'siaf_index', 'exponent_x', 'exponent_y', 'Sci2IdlX', 'Sci2IdlY', 'Idl2SciX', 'Idl2SciY'))
            distortion_reference_table.add_column(
                Column([AperName] * len(distortion_reference_table), name='AperName'), index=0)
            distortion_reference_file_name = os.path.join(JWST_SOURCE_DATA_ROOT, instrument,
                                                          '{}_siaf_distortion_{}.txt'.format(instrument.lower(),
                                                                                             AperName.lower()))
            # distortion_reference_table.pprint()
            comments = []
            comments.append('{} distortion reference file for SIAF\n'.format(instrument))
            comments.append('Aperture: {}\n'.format(AperName))
            comments.append('Based on coefficients given in {},'.format(os.path.basename(coldfit_source_data_file)))
            # comments.append('that were rescaled, shifted for a different reference pixel location, and rotated:')
            # comments.append('Rotation of {:2.3f} deg was removed and is carried separately in V3IdlYangle.'.format(
            #     np.rad2deg(V3angle)))  # *units.deg.to(units.arcsecond)
            # if 'may_2015' in distortion_file_name:
            #     comments.append(
            #         'These parameters are stored in the currently (January 2018) active SIAF (PRDOPSSOC-G-012). ')
            comments.append('')
            comments.append('Generated {} {}'.format(timestamp.isot, timestamp.scale))
            comments.append('by {}'.format(username))
            comments.append('')
            distortion_reference_table.meta['comments'] = comments
            distortion_reference_table.write(distortion_reference_file_name, format='ascii.fixed_width',
                                             delimiter=',', delimiter_pad=' ', bookend=False, overwrite=True)

            V3SciYAngle = betaY
            V3SciXAngle = betaX
            if np.abs(V3SciYAngle) < 90.:
                V3IdlYAngle = V3SciYAngle
            else:
                V3IdlYAngle = V3SciYAngle - np.sign(V3SciYAngle) * 180.

            if len(siaf_alignment) == 0: # first entry
                siaf_alignment['AperName'] = ['{:>30}'.format(AperName)]
                siaf_alignment['V3IdlYAngle'] = [V3IdlYAngle]
                siaf_alignment['V3SciXAngle'] = V3SciXAngle #[np.rad2deg(betaX)]
                siaf_alignment['V3SciYAngle'] = V3SciYAngle #[np.rad2deg(betaY)]
                siaf_alignment['V2Ref'] = [V2Ref]
                siaf_alignment['V3Ref'] = [V3Ref]
            else:
                siaf_alignment.add_row(['{:>30}'.format(AperName), V3IdlYAngle, V3SciXAngle, V3SciYAngle, V2Ref,V3Ref])
    comments = []
    comments.append('{} alignment parameter reference file for SIAF'.format(instrument))
    comments.append('')
    comments.append('This file contains the focal plane alignment parameters of master apertures calibrated')
    comments.append('during FGS-SI alignment.')
    comments.append('')
    comments.append('Generated {} {}'.format(timestamp.isot, timestamp.scale))
    comments.append('by {}'.format(username))
    comments.append('')
    siaf_alignment.meta['comments'] = comments
    siaf_alignment.write(outfile, format='ascii.fixed_width', delimiter=',',
                                     delimiter_pad=' ', bookend=False, overwrite=True)


# NIRISS reference files
def generate_siaf_pre_flight_reference_files_niriss(distortion_file_name, verbose=False):

    instrument = 'NIRISS'

    # hardcoded pixelscale, reference?
    nscale = 0.064746 # arcsec/pixel

    # polynomial degree
    order = 4

    if 0:
        # write prvious distortion coeffs to file
        # May 2015 NIRISS delivery from Alex Fullerton

        A3f = np.array([-2.0388254166E+01,
                           1.0098516941E+00, 1.0353224352E-02,
                           -1.9696277320E-07, -6.8952192578E-07, -2.4045394298E-07,
                           1.9735679757E-10, 5.3064397321E-10, 2.7406343861E-10, 1.2852645337E-10,
                           -2.5872994068E-14, -1.2855664341E-13, -2.2660453903E-14, -1.1668982024E-13, -7.6194308787E-16])

        B3f = np.array([-1.2692330360E+01,
                           -4.2610596865E-03, 1.0151660442E+00,
                           -2.4012283006E-07, 6.4668375899E-07, 3.1266804399E-06,
                           -9.6323615750E-11, -5.6579524399E-10, -2.7661065105E-10, -1.5520402741E-09,
                           2.4306057655E-14, 1.5179872265E-14, 1.4515561604E-13, 1.5168905306E-15, 2.7000358329E-13])

        C3f = np.array([2.0045064926E+01,
                           9.9019122124E-01, -1.0050848126E-02,
                           1.7905556149E-07, 6.3058558908E-07, 2.0085650476E-07,
                           -1.8718689099E-10, -4.9044374117E-10, -2.3546886663E-10, -1.0646614840E-10,
                           2.5364267774E-14, 1.1937670362E-13, 1.7891919296E-14, 1.0486258400E-13, -2.8008798233E-15])

        D3f = np.array([1.2594367027E+01,
                           4.1380361654E-03, 9.8487943411E-01,
                           2.4823151534E-07, -5.9602325564E-07, -2.8410295272E-06,
                           9.2453565637E-11, 5.1690540737E-10, 2.5494825695E-10, 1.4019794214E-09,
                           -2.2563438138E-14, -1.5354696478E-14, -1.2849642953E-13, -3.8877020804E-15, -2.4232504024E-13])

        fullerton_table = Table((A3f, B3f, C3f, D3f), names=(
        'A_real_to_ideal', 'B_real_to_ideal', 'C_ideal_to_real', 'D_ideal_to_real'))

        distortion_file_name = os.path.join(JWST_SOURCE_DATA_ROOT, instrument,
                                            'niriss_astrometric_coefficients_may_2015_with_header.txt')

        fullerton_table.write(distortion_file_name, format='ascii.fixed_width', delimiter=',', delimiter_pad=' ', bookend=False)
        1/0

    distortion_coefficients = Table.read(distortion_file_name, format='ascii.basic')

    A3m = distortion_coefficients['A_real_to_ideal'].data
    B3m = distortion_coefficients['B_real_to_ideal'].data
    C3m = distortion_coefficients['C_ideal_to_real'].data
    D3m = distortion_coefficients['D_ideal_to_real'].data



    # write intermediate csv file, JSA: converted to focal plane alignment reference file
    # outfile = os.path.join(test_dir, 'SIAFcoeffs_pysiaf.csv')
    outfile = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, '{}_siaf_alignment.txt'.format(instrument.lower()))
    if os.path.isfile(outfile):
        os.remove(outfile)
    oss_flags = [False, True]

    siaf_alignment = None

    # verbose = True


    if 0:
        A = A3m
        B = B3m
        C = C3m
        D = D3m

        if verbose:
            print('Original values')
            print('A')
            polynomial.print_triangle(A)
            print('B')
            polynomial.print_triangle(B)
        print('C')
        polynomial.print_triangle(C)
            # print('D')
            # polynomial.print_triangle(D)
        # Scale to arcsec
        (AX, BX, CX, DX) = polynomial.rescale(A, B, C, D, nscale)

        if 0:
            print
            print('Scaled values')
            if verbose:
                print('AX')
                polynomial.print_triangle(AX)
                print('BX')
                polynomial.print_triangle(BX)
            print('CX')
            polynomial.print_triangle(CX)
            print('DX')
            polynomial.print_triangle(DX)

            V2c = polynomial.poly(AX, 1023.5, 1023.5)
            V3c = polynomial.poly(BX, 1023.5, 1023.5)
            print('V2V3 center %8.3f %8.3f' % (V2c, V3c))
            # Corners
            V2c1 = polynomial.poly(AX, 0, 0)
            V3c1 = polynomial.poly(BX, 0, 0)
            V2c2 = polynomial.poly(AX, 2047, 0)
            V3c2 = polynomial.poly(BX, 2047, 0)
            V2c3 = polynomial.poly(AX, 2047, 2047)
            V3c3 = polynomial.poly(BX, 2047, 2047)
            V2c4 = polynomial.poly(AX, 0, 2047)
            V3c4 = polynomial.poly(BX, 0, 2047)

            AS = polynomial.shift_coefficients(AX, 1023.5, 1023.5)
            AS[0] = 0.0
            BS = polynomial.shift_coefficients(BX, 1023.5, 1023.5)
            BS[0] = 0.0
            CS = polynomial.shift_coefficients(CX, V2c, V3c)
            CS[0] = 0.0
            DS = polynomial.shift_coefficients(DX, V2c, V3c)
            DS[0] = 0.0

            print('\nAS')
            polynomial.print_triangle(AS)
            print('BS')
            polynomial.print_triangle(BS)

            xScale = np.hypot(AS[1], BS[1])
            yScale = np.hypot(AS[2], BS[2])
            print('Scales X %10.6f  Y %10.6f' % (xScale, yScale))

            xd = 300
            yd = 700
            u = polynomial.poly(AS, xd, yd)
            v = polynomial.poly(BS, xd, yd)
            print('Solution   %8.3f %8.3f' % (xd, yd))
            print('Raw values %8.3f %8.3f' % (u, v))
            xs = xd  # For OSS detector and science are identical
            ys = yd
            AF = AS
            BF = -BS

            print('\nAF')
            polynomial.print_triangle(AF)
            print('BF')
            polynomial.print_triangle(BF)

            xi = polynomial.poly(AF, xs, ys)
            yi = polynomial.poly(BF, xs, ys)
            print('Ideal values')
            print(' %8.3f %8.3f' % (xs, ys))
            print(' %8.3f %8.3f' % (xi, yi))

            CF = polynomial.flip_y(CS)
            DF = polynomial.flip_y(DS)

            print('\nCS')
            polynomial.print_triangle(CS)
            print('DS')
            polynomial.print_triangle(DS)
            print('CF')
            polynomial.print_triangle(CF)
            print('DF')
            polynomial.print_triangle(DF)

            xdr = polynomial.poly(CF, xi, yi)
            ydr = polynomial.poly(DF, xi, yi)
            print('Regained Detector  %8.3f %8.3f' % (xdr, ydr))

            betaX = math.atan2(AF[1], BF[1])
            betaY = math.atan2(AF[2], BF[2])
            print('\nAngles')
            print('betaX   %10.4f' % math.degrees(betaX))
            print('betaY   %10.4f' % math.degrees(betaY))
            V3angle = betaY
            if abs(V3angle) > math.pi / 2:
                V3angle = V3angle - math.copysign(math.pi, V3angle)
            print('V3angle %10.4f' % math.degrees(V3angle))
            (AR, BR) = polynomial.add_rotation(AF, BF, np.rad2deg(-V3angle))
            print('AR')
            polynomial.print_triangle(AR)
            print('BR')
            polynomial.print_triangle(BR)

            xii = polynomial.poly(AR, xs, ys)
            yii = polynomial.poly(BR, xs, ys)
            print('Ideal Rotated  %8.3f %8.3f' % (xii, yii))
            (u, v) = polynomial.add_rotation(xii, yii, np.rad2deg(-betaY))
            print('Ideal again %8.3f %8.3f' % (u, v))

            # (CR,DR) = add_rotation(CF,DF, -betaY)
            CR = polynomial.prepend_rotation_to_polynomial(CF, math.degrees(V3angle))
            DR = polynomial.prepend_rotation_to_polynomial(DF, math.degrees(V3angle))
            print('\nCR')
            polynomial.print_triangle(CR)
            print('DR')
            polynomial.print_triangle(DR)
            xs3 = polynomial.poly(CR, xii, yii)
            ys3 = polynomial.poly(DR, xii, yii)
            print('Regained rotated Science %8.3f %8.3f' % (xs3, ys3))




    print('*'*100)
    for oss in oss_flags:

        if oss:
            # aperture_name = 'NIRISSOSS'
            aperture_name = 'NIS_CEN_OSS'
            oss_factor =  1.
        else:
            # aperture_name = 'NIRISS'
            aperture_name = 'NIS_CEN'
            oss_factor = -1.

        print('{}'.format(aperture_name))
        A = A3m
        B = B3m
        C = C3m
        D = D3m

        if verbose:
            print('Original values')
            print('A')
            polynomial.print_triangle(A)
            print('B')
            polynomial.print_triangle(B)
            print('C')
            polynomial.print_triangle(C)
            print('D')
            polynomial.print_triangle(D)

        # Scale to arcsec
        (AX, BX, CX, DX) = polynomial.rescale(A, B, C, D, nscale)
        print('Scaled values')
        if verbose:
            print('AX')
            polynomial.print_triangle(AX)
            print('BX')
            polynomial.print_triangle(BX)
            print('CX')
            polynomial.print_triangle(CX)
            print('DX')
            polynomial.print_triangle(DX)

        V2c = polynomial.poly(AX, 1023.5, 1023.5)
        V3c = polynomial.poly(BX, 1023.5, 1023.5)

        if verbose:
            print('V2V3 center %8.3f %8.3f' %(V2c, V3c))
        AS = polynomial.shift_coefficients(AX, 1023.5, 1023.5)
        AS[0] = 0.0
        BS = polynomial.shift_coefficients(BX, 1023.5, 1023.5)
        BS[0] = 0.0
        CS = polynomial.shift_coefficients(CX, V2c, V3c)
        CS[0] = 0.0
        DS = polynomial.shift_coefficients(DX, V2c, V3c)
        DS[0] = 0.0

        if verbose:
            print('\nAS')
            polynomial.print_triangle(AS)
            print('BS')
            polynomial.print_triangle(BS)

        x0 = 300
        y0 = 700
        if oss is False:
            xd = x0
            yd = y0
            u = polynomial.poly(AS, x0, y0)
            v = polynomial.poly(BS, x0, y0)

            xs = -xd
            ys = -yd
            AF = -polynomial.flip_x(polynomial.flip_y(AS))
            BF = -polynomial.flip_x(polynomial.flip_y(BS))
        else:
            xd = 300
            yd = 700
            u = polynomial.poly(AS, xd, yd)
            v = polynomial.poly(BS, xd, yd)

            xs = xd  # For OSS detector and science are identical
            ys = yd
            AF = AS
            BF = -BS

        if verbose:
            print('\nRaw Results')
            print('%8.1f %8.1f'%(x0, y0))
            print('%8.3f %8.3f' %(u,v))
            print('\nAF')
            polynomial.print_triangle(AF)
            print('BF')
            polynomial.print_triangle(BF)

        xi = polynomial.poly(AF,xs,ys)
        yi = polynomial.poly(BF,xs,ys)
        if verbose:
            print('\nIdeal')
            print('%8.1f %8.1f' %(xs,ys))
            print('%8.3f %8.3f' %(xi,yi))

        xdr = polynomial.poly(CS, xi, yi)
        ydr = polynomial.poly(DS, xi, yi)

        if verbose:
            print('Detector   %8.3f %8.3f' %(xd, yd))
            print('Ideal      %8.3f %8.3f' %(xi, yi))
            print('Detector 2 %8.3f %8.3f' %(xdr, ydr))
            print()

        if oss is False:
            CF = -polynomial.flip_x(polynomial.flip_y(CS))
            DF = -polynomial.flip_x(polynomial.flip_y(DS))
        else:
            CF = polynomial.flip_y(CS)
            DF = polynomial.flip_y(DS)

        xsr = polynomial.poly(CF,xi,yi)
        ysr = polynomial.poly(DF, xi, yi)
        if verbose:
            print('Regained Science  %8.3f %8.3f' %(xsr, ysr))
            print()
            print('AF')
            polynomial.print_triangle(AF)
            print('BF')
            polynomial.print_triangle(BF)

        betaX = np.arctan2(oss_factor * AF[1], BF[1])
        betaY = np.arctan2(oss_factor * AF[2], BF[2])
        if verbose:
            print('\nAngles')
            print('betaX   %10.4f'% np.rad2deg(betaX))
            print('betaY   %10.4f'% np.rad2deg(betaY))
        V3angle = betaY
        if abs(V3angle) > np.pi/2:
            V3angle = V3angle - np.copysign(np.pi, V3angle)
        if verbose:
            print('V3angle %10.4f' % np.rad2deg(V3angle))

        (AR,BR) = polynomial.add_rotation(AF, BF, -1 * oss_factor * np.rad2deg(V3angle))

        if verbose:
            print('AR')
            polynomial.print_triangle(AR)
            print('BR')
            polynomial.print_triangle(BR)

            xii = polynomial.poly(AR, xs,ys)
            yii = polynomial.poly(BR, xs,ys)
            print('Ideal Rotated  %8.3f %8.3f' %(xii,yii))
            (u,v) = polynomial.add_rotation(xii, yii, np.rad2deg(-betaY))
            print('Ideal again %8.3f %8.3f' %(u,v))

        # take out the rotation, carried separately in V3IdlYangle
        CR = polynomial.prepend_rotation_to_polynomial(CF, oss_factor * np.rad2deg(V3angle))
        DR = polynomial.prepend_rotation_to_polynomial(DF, oss_factor * np.rad2deg(V3angle))

        if verbose:
            print('CR')
            polynomial.print_triangle(CR)
            print('DR')
            polynomial.print_triangle(DR)
            xsr = polynomial.poly(CR,xii,yii)
            ysr = polynomial.poly(DR, xii, yii)
            # if verbose:
            print('Science  %8.3f %8.3f' %(xsr, ysr))
            pl.figure()
            pl.clf()
            pl.title(aperture_name)
            pl.grid(True)
            x0 = -polynomial.poly(AS, 0.0, 0.0)
            y0 = -polynomial.poly(BS, 0.0, 0.0)
            x1 = -polynomial.poly(AS, 0.0, 1023.5)
            y1 = -polynomial.poly(BS, 0.0, 1023.5)
            x2 = -polynomial.poly(AS, 1023.5, 0.0)
            y2 = -polynomial.poly(BS, 1023.5, 0.0)
            pl.plot([x1,x0,x2], [y1,y0,y2])
            pl.show()

            print('Append to csv file')
        # makecsv(outfile, aperture_name, V3angle, betaX, betaY, AR, BR, CR, DR)

        if len(aperture_name) > 30:
            raise RuntimeError('aperture_name will be truncated')
        # if siaf_alignment is None:
        V2Ref = 60 * (-4.835)
        V3Ref = -60 * (3.825 + 7.8)

        if aperture_name == 'NIS_CEN':
            siaf_alignment = Table()
            siaf_alignment['AperName'] = ['{:>30}'.format(aperture_name)]
            siaf_alignment['V3IdlYAngle'] = [np.rad2deg(V3angle)]
            siaf_alignment['V3SciXAngle'] = [np.rad2deg(betaX)]
            siaf_alignment['V3SciYAngle'] = [np.rad2deg(betaY)]
            siaf_alignment['V2Ref'] = [V2Ref]
            siaf_alignment['V3Ref'] = [V3Ref]

        else:
            siaf_alignment.add_row(['{:>30}'.format(aperture_name), np.rad2deg(V3angle), np.rad2deg(betaX), np.rad2deg(betaY), V2Ref, V3Ref])



        number_of_coefficients = len(AR)
        polynomial_degree = np.int((np.sqrt(8 * number_of_coefficients + 1) - 3) / 2)

        # if oss is False:
        if 1:
            # aperture_name = 'NIS_CEN'
            siaf_index = []
            exponent_x = []
            exponent_y = []
            for i in range(polynomial_degree + 1):
                for j in np.arange(i + 1):
                    siaf_index.append('{:d}{:d}'.format(i, j))
                    exponent_x.append(i-j)
                    exponent_y.append(j)

            distortion_reference_table = Table((siaf_index, exponent_x, exponent_y, AR, BR, CR, DR), names=('siaf_index', 'exponent_x', 'exponent_y', 'Sci2IdlX', 'Sci2IdlY', 'Idl2SciX', 'Idl2SciY'))
            distortion_reference_table.add_column(Column([aperture_name] * len(distortion_reference_table), name='AperName'), index=0)
            distortion_reference_file_name = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, 'niriss_siaf_distortion_{}.txt'.format(aperture_name.lower()))
            distortion_reference_table.pprint()
            comments = []
            comments.append('NIRISS distortion reference file for SIAF\n')
            comments.append('Aperture: {}'.format(aperture_name))
            if 'august_2016' in distortion_file_name:
                comments.append('Based on coefficients from Martel & Fullerton "The Geometric Distortion of NIRISS" (JWST-STScI-003524 Rev A),')
            elif 'may_2015' in distortion_file_name:
                comments.append('Based on coefficients from Fullerton delivery in May 2015,')
            comments.append('that were rescaled, shifted for a different reference pixel location, and rotated:')
            comments.append('Rotation of {:2.3f} deg was removed and is carried separately in V3IdlYangle.'.format(np.rad2deg(V3angle))) #*units.deg.to(units.arcsecond)
            if 'may_2015' in distortion_file_name:
                comments.append('These parameters are stored in the currently (January 2018) active SIAF (PRDOPSSOC-G-012). ')
            elif 'august_2016' in distortion_file_name:
                comments.append('These parameters are stored in PRDOPSSOC-H-014.')
            comments.append('')
            comments.append('Generated {} {}'.format(timestamp.isot, timestamp.scale))
            comments.append('by {}'.format(username))
            comments.append('')
            distortion_reference_table.meta['comments'] = comments
            distortion_reference_table.write(distortion_reference_file_name, format='ascii.fixed_width', delimiter=',', delimiter_pad=' ', bookend=False, overwrite=True)


    comments = []
    comments.append('{} alignment parameter reference file for SIAF'.format(instrument))
    comments.append('')
    comments.append('This file contains the focal plane alignment parameters calibrated during FGS-SI alignment.')
    comments.append('')
    comments.append('Generated {} {}'.format(timestamp.isot, timestamp.scale))
    comments.append('by {}'.format(username))
    comments.append('')
    siaf_alignment.meta['comments'] = comments
    siaf_alignment.write(outfile, format='ascii.fixed_width', delimiter=',',
                                     delimiter_pad=' ', bookend=False)


# FGS reference files
def generate_siaf_pre_flight_reference_files_fgs(verbose=False, mode='siaf'):
    """Write the FGS alignment and distortion source data files to file.

    The inputs are hardcoded reference positions and distortion coefficients obtained from
    Honeywell.

    Parameters
    ----------
    verbose : bool
        verbosity
    mode : str
        Either `siaf` or `fsw`. This specifies in which convention/format the distortion
        coefficients are written.

    """
    instrument = 'FGS'

    center_offset_x = 1023.5
    center_offset_y = 1023.5

    # hardcoded pixelscale, reference?
    scale = 0.06738281367  # arcsec/pixel

    if mode == 'siaf':
        # write focal plane alignment reference file
        outfile = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, '{}_siaf_alignment.txt'.format(instrument.lower()))
        oss_flags = [False, True]
    elif mode == 'fsw':
        outfile = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, '{}_fsw_coefficients.txt'.format(instrument.lower()))
        oss_flags = [True]

    if os.path.isfile(outfile):
        os.remove(outfile)

    siaf_alignment = None
    counter = 0

    for aperture_id in 'FGS1 FGS2'.split():

        if aperture_id == 'FGS1':
            V2Ref = 207.1900
            V3Ref = -697.5000

            # coefficients copied from Cox' makeSIAF.py to reproduce PRDOPSSOC-H-015
            # February 2015 FGS delivery
            # these numbers match the `To be Updated for CV3` column in the Tables on page 6ff
            # of an unpublished word document entitled `FGS Transformation for CV3.docx` by
            # Julia Zhou, e.g. C = IDEALPTOREALPXCOE_N

            # Initialize the parameters
            A = np.array(
                [-2.33369320E+01, 9.98690490E-01, 1.05024970E-02, 2.69889020E-06, 6.74362640E-06,
                 9.91415010E-07, 1.21090320E-09, -2.84802930E-11, 1.27892930E-09, -1.91322470E-11,
                 5.34567520E-14, 9.29791010E-14, 8.27060020E-14, 9.70576590E-14, 1.94203870E-14])

            B = np.array(
                [-2.70337440E+01, -2.54596080E-03, 1.01166810E+00, 2.46371870E-06, 2.08880620E-06,
                 9.32489680E-06, -4.11885660E-11, 1.26383770E-09, -7.60173360E-11, 1.36525900E-09,
                 2.70499280E-14, 5.70198270E-14, 1.43943080E-13, 7.02321790E-14, 1.21579450E-13])

            C = np.array(
                [2.31013520E+01, 1.00091800E+00, -1.06389620E-02, -2.65680980E-06, -6.51704610E-06,
                 -7.45631440E-07, -1.29600400E-09, -4.27453220E-12, -1.27808870E-09, 5.01165140E-12,
                 2.72622090E-15, 5.42715750E-15, 3.46979980E-15, 2.49124350E-15, 1.22848570E-15])

            D = np.array(
                [2.67853100E+01, 2.26545910E-03, 9.87816850E-01, -2.35598140E-06, -1.91455620E-06,
                 -8.92779540E-06, -3.24201520E-11, -1.30056630E-09, -1.73730700E-11,
                 -1.27341590E-09, 1.84205730E-15, 3.13647160E-15, -2.99705840E-16, 1.98589690E-15,
                 -1.26523200E-15])

        elif aperture_id == 'FGS2':
            V2Ref = 24.4300
            V3Ref = -697.5000

            A = np.array(
                [-3.28410900E+01, 1.03455010E+00, 2.11920160E-02, -9.08746430E-06, -1.43516480E-05,
                 -3.93814140E-06, 1.60956450E-09, 5.82814640E-10, 2.02870570E-09, 2.08582470E-10,
                 -2.79748590E-14, -8.11622820E-14, -4.76943000E-14, -9.01937740E-14,
                 -8.76203780E-15])

            B = np.array(
                [-7.76806220E+01, 2.92234710E-02, 1.07790000E+00, -6.31144890E-06, -7.87266390E-06,
                 -2.14170580E-05, 2.13293560E-10, 2.03376270E-09, 6.74607790E-10, 2.41463060E-09,
                 -2.30267730E-14, -3.63681270E-14, -1.35117660E-13, -4.22207660E-14,
                 -1.16201020E-13])

            C = np.array(
                [3.03390890E+01, 9.68539030E-01, -1.82288450E-02, 7.72758330E-06, 1.17536430E-05,
                 2.71516870E-06, -1.28167820E-09, -6.34376120E-12, -1.24563160E-09, -9.26192040E-12,
                 8.14604260E-16, -5.93798790E-16, -2.69247540E-15, -4.05196100E-15, 2.14529600E-15])

            D = np.array(
                [7.13783150E+01, -2.55191710E-02, 9.30941560E-01, 5.01322910E-06, 5.10548510E-06,
                 1.68083960E-05, 9.41565630E-12, -1.29749490E-09, -1.89194230E-11, -1.29425530E-09,
                 -2.81501600E-15, -1.73025000E-15, 2.57732600E-15, 1.75268080E-15, 2.95238320E-15])

        number_of_coefficients = len(A)
        polynomial_degree = np.int((np.sqrt(8 * number_of_coefficients + 1) - 3) / 2)

        # generate distortion coefficient files
        siaf_index = []
        exponent_x = []
        exponent_y = []
        for i in range(polynomial_degree + 1):
            for j in np.arange(i + 1):
                siaf_index.append('{:d}{:d}'.format(i, j))
                exponent_x.append(i-j)
                exponent_y.append(j)


        print('*'*100)
        aperture_name = '{}_FULL'.format(aperture_id)
        for oss in oss_flags:

            if oss:
                aperture_name = aperture_name + '_OSS'
                oss_factor =  1.
            else:
                oss_factor = -1.

            print('{}'.format(aperture_name))

            if mode == 'fsw':
                (AX, BX, CX, DX) = (A, B, C, D)

                AS = polynomial.shift_coefficients(AX, center_offset_x, center_offset_y)
                BS = polynomial.shift_coefficients(BX, center_offset_x, center_offset_y)

                AS0 = copy.deepcopy(AS[0])
                BS0 = copy.deepcopy(BS[0])
                AS[0] = 0.0
                BS[0] = 0.0

                betaY = np.arctan2(AS[2], BS[2])
                print('Beta Y', np.degrees(betaY))
                print('Shift zeros', AS0, BS0)

                AR = AS * np.cos(betaY) - BS * np.sin(betaY)
                BR = AS * np.sin(betaY) + BS * np.cos(betaY)


                AR[0] = center_offset_x
                BR[0] = center_offset_y

                AF = polynomial.shift_coefficients(AR, -center_offset_x, -center_offset_y)
                BF = polynomial.shift_coefficients(BR, -center_offset_x, -center_offset_y)

                # Inverse matrices
                xc = polynomial.poly(AX, center_offset_x, center_offset_y)
                yc = polynomial.poly(BX, center_offset_x, center_offset_y)
                # CS1 = 1.0*C1 # Force a real copy
                CS = polynomial.shift_coefficients(CX, xc, yc)
                DS = polynomial.shift_coefficients(DX, xc, yc)
                CS0 = copy.deepcopy(CS[0])
                DS0 = copy.deepcopy(DS[0])

                CS[0] = 0.0
                DS[0] = 0.0
                CR = polynomial.prepend_rotation_to_polynomial(CS, np.degrees(betaY))
                DR = polynomial.prepend_rotation_to_polynomial(DS, np.degrees(betaY))
                CR[0] = CS0
                DR[0] = DS0
                CF = polynomial.shift_coefficients(CR, -center_offset_x, -center_offset_y)
                DF = polynomial.shift_coefficients(DR, -center_offset_x, -center_offset_y)
                distortion_reference_table = Table((siaf_index, exponent_x, exponent_y, AF, BF, CF, DF),
                                                   names=('siaf_index', 'exponent_x', 'exponent_y', 'Sci2IdlX', 'Sci2IdlY', 'Idl2SciX','Idl2SciY'))

                V3angle = 0
                betaX = 0


            else:
                # Scale to arcsec
                (AX, BX, CX, DX) = polynomial.rescale(A, B, C, D, scale)


                V2c = polynomial.poly(AX, center_offset_x, center_offset_y)
                V3c = polynomial.poly(BX, center_offset_x, center_offset_y)

                AS = polynomial.shift_coefficients(AX, center_offset_x, center_offset_y)
                AS[0] = 0.0
                BS = polynomial.shift_coefficients(BX, center_offset_x, center_offset_y)
                BS[0] = 0.0
                CS = polynomial.shift_coefficients(CX, V2c, V3c)
                CS[0] = 0.0
                DS = polynomial.shift_coefficients(DX, V2c, V3c)
                DS[0] = 0.0

                if aperture_id == 'FGS1':
                    if oss is False:
                        AF = -polynomial.flip_x(polynomial.flip_y(AS))
                        BF = -polynomial.flip_x(polynomial.flip_y(BS))
                        CF = -polynomial.flip_x(polynomial.flip_y(CS))
                        DF = -polynomial.flip_x(polynomial.flip_y(DS))
                    else:
                        AF = AS # For OSS detector and science are identical
                        BF = -BS
                        CF = polynomial.flip_y(CS)
                        DF = polynomial.flip_y(DS)
                elif aperture_id == 'FGS2':
                    if oss is False:
                        AF = -polynomial.flip_x(AS)
                        BF =  polynomial.flip_x(BS)
                        CF = -polynomial.flip_x(CS)
                        DF =  polynomial.flip_x(DS)
                    else:
                        AF = AS # For OSS detector and science are identical
                        BF = BS
                        CF = CS
                        DF = DS

                betaX = np.arctan2(oss_factor * AF[1], BF[1])
                betaY = np.arctan2(oss_factor * AF[2], BF[2])

                V3angle = copy.deepcopy(betaY)
                if (abs(V3angle) > np.pi/2):
                    V3angle = V3angle - np.copysign(np.pi, V3angle)

                (AR,BR) = polynomial.add_rotation(AF, BF, -1 * oss_factor * np.rad2deg(V3angle))

                # take out the rotation, carried separately in V3IdlYangle
                CR = polynomial.prepend_rotation_to_polynomial(CF, oss_factor * np.rad2deg(V3angle))
                DR = polynomial.prepend_rotation_to_polynomial(DF, oss_factor * np.rad2deg(V3angle))
                distortion_reference_table = Table((siaf_index, exponent_x, exponent_y, AR, BR, CR, DR),
                                                   names=('siaf_index', 'exponent_x', 'exponent_y', 'Sci2IdlX', 'Sci2IdlY', 'Idl2SciX','Idl2SciY'))

            print('{} {}'.format(aperture_name, np.rad2deg(betaY)))
            # if aperture_name == 'FGS1_FULL':  # first in loop
            if counter == 0:  # first in loop
                siaf_alignment = Table()
                siaf_alignment['AperName'] = ['{:>30}'.format(aperture_name)]
                siaf_alignment['V3IdlYAngle'] = [np.rad2deg(V3angle)]
                siaf_alignment['V3SciXAngle'] = [np.rad2deg(betaX)]
                siaf_alignment['V3SciYAngle'] = [np.rad2deg(betaY)]
                siaf_alignment['V2Ref'] = [V2Ref]
                siaf_alignment['V3Ref'] = [V3Ref]
            else:
                siaf_alignment.add_row(['{:>30}'.format(aperture_name), np.rad2deg(V3angle), np.rad2deg(betaX), np.rad2deg(betaY), V2Ref, V3Ref])

            counter += 1


            distortion_reference_table.add_column(Column([aperture_name] * len(distortion_reference_table), name='AperName'), index=0)
            if mode == 'fsw':
                distortion_reference_file_name = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, 'fgs_fsw_distortion_{}.txt'.format(aperture_name.lower()))
            else:
                distortion_reference_file_name = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, 'fgs_siaf_distortion_{}.txt'.format(aperture_name.lower()))

            comments = []
            comments.append('FGS distortion reference file for SIAF\n')
            comments.append('')
            comments.append('Based on coefficients delivered to STScI in February 2015.')
            comments.append('These parameters are stored in PRDOPSSOC-H-014.')
            comments.append('')
            comments.append('Generated {} {}'.format(timestamp.isot, timestamp.scale))
            comments.append('by {}'.format(username))
            comments.append('')
            distortion_reference_table.meta['comments'] = comments
            distortion_reference_table.write(distortion_reference_file_name, format='ascii.fixed_width', delimiter=',', delimiter_pad=' ', bookend=False, overwrite=True)

    comments = []
    comments.append('{} alignment parameter reference file for SIAF'.format(instrument))
    comments.append('')
    comments.append('This file contains the focal plane alignment parameters calibrated during FGS-SI alignment.')
    comments.append('')
    comments.append('Generated {} {}'.format(timestamp.isot, timestamp.scale))
    comments.append('by {}'.format(username))
    comments.append('')
    siaf_alignment.meta['comments'] = comments
    siaf_alignment.write(outfile, format='ascii.fixed_width', delimiter=',',
                                     delimiter_pad=' ', bookend=False, overwrite=True)


def generate_siaf_xml_field_format_reference_files(verbose=False):

    # from constants import JWST_PRD_DATA_ROOT
    basepath = pysiaf.constants.JWST_PRD_DATA_ROOT

    xml_formats = {}
    siaf_detector_layout = pysiaf.iando.read.read_siaf_detector_layout()
    for instrument in 'NIRCam NIRISS MIRI NIRSPec FGS'.split():
        T = Table()
        print('*'*100)
        print('{}'.format(instrument))
        filename = os.path.join(basepath, instrument + '_SIAF.xml')
        primary_master_aperture_name = siaf_detector_layout['AperName'][siaf_detector_layout['InstrName']==instrument.upper()][0]

        tree = ET.parse(filename)

        # generate Aperture objects from SIAF XML file, parse the XML
        for entry in tree.getroot().iter('SiafEntry'):
            show = False
            field_index = 1
            for node in entry.iterchildren():
                field_number = '1.{:d}'.format(field_index)
                # print('{}   {}: {}'.format(primary_master_aperture_name, node.tag, node.text))
                if (node.tag == 'AperName') and (node.text == primary_master_aperture_name):
                    show=True
                if (node.tag in pysiaf.aperture._attributes_that_can_be_none) and (node.text is None):
                    value = node.text
                    prd_data_class = 'None'
                    python_format = 'None'
                elif node.tag in pysiaf.aperture._integer_attributes:
                    prd_data_class = 'integer'
                    python_format = 'd'
                    try:
                        value = int(node.text)
                    except (TypeError):
                        print('{}: {}'.format(node.tag, node.text))
                        raise TypeError
                elif (node.tag in pysiaf.aperture._string_attributes):
                    prd_data_class = 'string'
                    python_format = ''
                    value = node.text
                else:
                    prd_data_class = 'float'
                    # if show:
                    #     print('{}   {}: {}'.format(primary_master_aperture_name, node.tag, node.text))

                    try:
                        decimal_part = node.text.split('.')[1]


                        if 'E' in decimal_part:
                            decimal_precision = len(decimal_part.split('E')[0])
                            python_format = '.{}e'.format(decimal_precision)
                        else:
                            decimal_precision = len(decimal_part)
                            python_format = '.{}f'.format(decimal_precision)

                        try:
                            value = float(node.text)
                        except (TypeError):
                            print('{}: {}'.format(node.tag, node.text))
                            raise TypeError

                    except IndexError: # e.g. MIRI 'DetSciYAngle'
                        prd_data_class = 'int'
                        python_format = 'd'

                if show:
                    if node.tag == 'Comment':
                        prd_data_class = 'string'
                        python_format = ''
                        node.text = ''

                    if verbose:
                        print('{:5} {} {:10} {:10} {:30}'.format(field_number, node.tag, prd_data_class, python_format, str(node.text)))
                    if len(T) == 0:
                        T['field_nr'] = ['{:>5}'.format('1.1')]
                        T['field_name'] = ['{:>20}'.format('InstrName')]
                        T['format'] = ['{:>10}'.format('string')]
                        T['pyformat'] = ['{:>10}'.format('')]
                        T['example'] = ['{:>30}'.format(instrument.upper())]
                        T.add_row((field_number, node.tag, prd_data_class, python_format, node.text))
                    else:
                        T.add_row((field_number, node.tag, prd_data_class, python_format, node.text))

                field_index += 1

        xml_formats[instrument] = T
                # except (ValueError, TypeError):
                #     value = node.text

                # self.__dict__[tag] = value
                # setattr(a, node.tag, value)

        outfile = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, '{}_siaf_xml_field_format.txt'.format(instrument.lower()))
        comments = []
        comments.append('{} xml field format reference file for SIAF'.format(instrument))
        comments.append('')
        comments.append('This file describes the format and order of the XML fields present in the SIAF.')
        comments.append('This is compliant with the IRCD (JWST-STScI-000949) and complementary because the decimal')
        comments.append('precision of float fields is specified.')
        comments.append('')
        comments.append('Generated {} {}'.format(timestamp.isot, timestamp.scale))
        comments.append('by {}'.format(username))
        comments.append('')
        T.meta['comments'] = comments
        T.write(outfile, format='ascii.fixed_width', delimiter=',',
                             delimiter_pad=' ', bookend=False)
        print('Wrote {}'.format(outfile))

    if verbose:
        nircam_table = xml_formats['NIRCam']
        for col in ['format', 'pyformat', 'example']:
            nircam_table.rename_column(col, 'NIRCam_{}'.format(col).lower())
        for key in xml_formats.keys():
            if key != 'NIRCam':
                for col in ['format', 'pyformat', 'example']:
                    nircam_table['{}_{}'.format(key, col).lower()] = xml_formats[key][col]


        columns_to_show = ['field_nr', 'field_name']
        for col in nircam_table.colnames:
            if ('pyformat' in col) or ('format' in col):#col.split('_')[1] in ['Format', 'pyformat']:
                columns_to_show.append(col)
        columns_to_show.append('nircam_example')

        nircam_table[columns_to_show].pprint()
        nircam_table[columns_to_show].write('siaf_field_formats.txt', format='ascii.fixed_width', delimiter=',', bookend=False)


def nircam_get_polynomial_both(apName, siaf_aperture_definitions, coldfit_name_mapping, coldfit_source_data, makeplot=False, verbose=False):
    """Get forward and inverse polynomials between detector and V2V3
    Generate polynomials for Ideal coordinates.

    Originally written by Colin Cox, adapted by Johannes Sahlmann

    :param apName:
    :param makeplot:
    :param verbose:
    :return:
    """
    OSS = 'OSS' in apName  # Different axes from OSS and non-OSS apertures

    (A, B) = nircam_get_polynomial_forward(apName, siaf_aperture_definitions, coldfit_name_mapping, coldfit_source_data, makeplot)
    (C, D) = nircam_get_polynomial_inverse(apName, siaf_aperture_definitions, coldfit_name_mapping, coldfit_source_data, verbose=False)

    AR, BR, CR, DR, betaX, V3Angle, v2Ref, v3Ref = tools.convert_polynomial_coefficients(A, B, C, D, oss=OSS)

    return (AR, BR, CR, DR, betaX, V3Angle, v2Ref, v3Ref)


def nircam_get_polynomial_forward(apName, siaf_aperture_definitions, coldfit_name_mapping, coldfit_source_data, makeplot=False, test=False, verbose=False):
    """return poly coefficients transforming to OTESKY

    Originally written by Colin Cox, adapted by Johannes Sahlmann


    :param apName: SIAF Aperture name
    :param makeplot:
    :param test:
    :param verbose:
    :return: AFS, BFS

    v2ref = AFS[0]
    v3ref = BFS[0]

    aperture_name_list is the list of aperture names read from current SIAF

    """
    
    aperture_name_list = siaf_aperture_definitions['AperName'].tolist()

    for row, name in enumerate(aperture_name_list):
        if name == apName:
            r = row

    # read in hardcoded dictionary
    apSys = coldfit_name_mapping[apName]
    xref = siaf_aperture_definitions['XDetRef'][r]
    yref = siaf_aperture_definitions['YDetRef'][r]
    if xref == '':
        xref = 1024.5  # Allow for COMPOUND types
    else:
        xref = float(xref)  # with no pixel information
    if yref == '':
        yref = 1024.5
    else:
        yref = float(yref)

    print(apName, xref, yref)

    order = 5 # polynomial order
    terms = (order + 1) * (order + 2) // 2 # number of poly coeffs
    A = np.zeros((terms))
    B = np.zeros((terms))

    part1 = False
    part2 = False

    # read parameters from cold_fit_[] file
    for line in coldfit_source_data:
        column = line.split(',')
        modelname = column[0].strip()
        if modelname==apSys['pixels_to_mm']:
            a0 = float(column[7])
            a1 = float(column[9])
            a2 = float(column[8])
            b0 = float(column[28])
            b1 = float(column[30])
            b2 = float(column[29])
            part1 = True
            if verbose:
                print('a', a0, a1, a2)
                print('b', b0, b1, b2)

        # find transformation to OTESKY
        if modelname==apSys['mm_to_degrees']:
            for i in range(terms):
                A[i] = float(column[i + 7])
                B[i] = float(column[i + 28])
            (A1, B1) = polynomial.reorder(A, B)
            part2 = True
            if verbose:
                print(' Before combining')
                print('A1')
                polynomial.print_triangle(A1)
                print('B1')
                polynomial.print_triangle(B1)

    if not (part1 and part2):
        print('Incomplete Transform')
        return

    # Combine transformations
    delta = a1 * b2 - a2 * b1
    alpha = (b2 * a0 - a2 * b0) / delta
    beta = (-b1 * a0 + a1 * b0) / delta
    AT = polynomial.transform_coefficients(A1, a1, a2, b1, b2)
    BT = polynomial.transform_coefficients(B1, a1, a2, b1, b2)
    ATS = polynomial.shift_coefficients(AT, alpha, beta)
    BTS = polynomial.shift_coefficients(BT, alpha, beta)

    # Generate polynomials in terms of V2V3 in arcsec
    AF = 3600 * ATS
    BF = -3600 * BTS
    BF[0] = BF[0] - 468.0

    if makeplot:
        # Plot axes in V2V3 coords
        (V20, V30) = (AF[0], BF[0])
        print('Bottom Left Corner', V20, V30)
        V2x = polynomial.poly(AF, 2048.0, 0.0)
        V3x = polynomial.poly(BF, 2048.0, 0.0)
        V2y = polynomial.poly(AF, 0.0, 2048.0)
        V3y = polynomial.poly(BF, 0.0, 2048.0)
        V2opp = polynomial.poly(AF, 2048.0, 2048.0)
        V3opp = polynomial.poly(BF, 2048.0, 2048.0)
        V2c = polynomial.poly(AF, 1024.0, 1024.0)
        V3c = polynomial.poly(BF, 1024.0, 1024.0)
        print('Center', V2c, V3c)

        P.figure(1)
        # P.clf()
        P.plot((V2x, V20, V2y), (V3x, V30, V3y))
        P.plot((V2x, V2opp, V2y), (V3x, V3opp, V3y), linestyle='dashed')
        P.plot((V2c), (V3c), 'rx', ms=10.0)
        P.grid(True)
        P.axis('equal')
        P.text(V2x, V3x, 'X')
        P.text(V2y, V3y, 'Y')
        P.text(V2c, V3c, apName)
        P.title('NIRCam')
        P.xlabel('<---V2')
        P.ylabel('V3 --->')
        # V2 to the left
        (l, r) = P.xlim()
        P.xlim(r, l)

    # Shift to reference point (xref=1024.5, yref=1024.5 hardcoded for COMPOUND apertures. OK?)
    AFS = polynomial.shift_coefficients(AF, xref, yref)
    BFS = polynomial.shift_coefficients(BF, xref, yref)

    if test:
        xy = input('x y positions ')
        x = float(xy.split(',')[0])
        y = float(xy.split(',')[1])
        # Two step calculation
        xm = a0 + a1 * x + a2 * y
        ym = b0 + b1 * x + b2 * y
        xan = polynomial.poly(A1, xm, ym, 5)
        yan = polynomial.poly(B1, xm, ym, 5)
        v2 = 3600 * xan
        v3 = -3600 * (yan + 0.13)
        print('\n Two step forward calculation')
        print(x, y, xm, ym, xan, yan)
        print(v2, v3)

        v21 = polynomial.poly(AF, x, y, 5)
        v31 = polynomial.poly(BF, x, y, 5)
        print('One step')
        print(v21, v31)

        xr = x - 1024.5
        yr = y - 1024.5
        v2r = polynomial.poly(AFS, xr, yr, 5)
        v3r = polynomial.poly(BFS, xr, yr, 5)
        print('Shifted')
        print(v2r, v3r)

    return (AFS, BFS)


def nircam_get_polynomial_inverse(apName, siaf_aperture_definitions, coldfit_name_mapping, coldfit_source_data, verbose=False):
    """return poly coefficients transforming from OTESKY

    Originally written by Colin Cox, adapted by Johannes Sahlmann

    :param apName:
    :param verbose:
    :return:
    """

    aperture_name_list = siaf_aperture_definitions['AperName'].tolist()

    if verbose:
        print('Running inverse ...')
    for row, name in enumerate(aperture_name_list):
        if name == apName:
            if verbose:
                print('Found aperture {}'.format(apName))
            r = row
    apSys = coldfit_name_mapping[apName]
    xref = siaf_aperture_definitions['XDetRef'][r]
    yref = siaf_aperture_definitions['YDetRef'][r]
    if xref == '':
        xref = 1024.5  # Allow for COMPOUND types
    else:
        xref = float(xref)  # with no pixel information
    if yref == '':
        yref = 1024.5
    else:
        yref = float(yref)

    order = 5
    terms = (order + 1) * (order + 2) // 2
    C = np.zeros((terms))
    D = np.zeros((terms))

    part1 = False
    part2 = False
    for line in coldfit_source_data:  # coeffs read in during initialization
        column = line.split(',')
        modelname = column[0].strip()
        if modelname==apSys['mm_to_pixels']: # linear transformation
            c0 = float(column[7])
            c1 = float(column[9])
            c2 = float(column[8])
            d0 = float(column[28])
            d1 = float(column[30])
            d2 = float(column[29])
            if verbose:
                print('c', c0, c1, c2)
                print('d', d0, d1, d2)
            part1 = True

        if modelname==apSys['degrees_to_mm']:  # Polynomial
            for i in range(terms):
                C[i] = float(column[i + 7])
                D[i] = float(column[i + 28])
            (C1, D1) = polynomial.reorder(C, D)

            if verbose:
                print('C1')
                polynomial.print_triangle(C1)
                print('D1')
                polynomial.print_triangle(D1)

            # Combination polynomials CF and DF transform
            # from XAN,YAN directly to x,y pixels.

            CS = polynomial.shift_coefficients(C1, 0.0, -0.13)
            DS = polynomial.shift_coefficients(D1, 0.0, -0.13)
            CV = np.zeros((terms))
            DV = np.zeros((terms))

            k = 0
            for i in range(order + 1):
                for j in range(i + 1):
                    CV[k] = (-1) ** j * CS[k] / 3600.0 ** i
                    DV[k] = (-1) ** j * DS[k] / 3600.0 ** i
                    k += 1
            part2 = True


    if part1 and part2:
        CVF = c1 * CV + c2 * DV
        CVF[0] = CVF[0] + c0
        DVF = d1 * CV + d2 * DV
        DVF[0] = DVF[0] + d0
        # Shift to reference position
        CVF[0] = CVF[0] - xref
        DVF[0] = DVF[0] - yref
    else:
        print('Incomplete transform')

    return (CVF, DVF)



