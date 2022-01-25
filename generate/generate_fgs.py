#!/usr/bin/env python
"""Script to generate FGS SIAF content and files using pysiaf and flight-like SIAF reference files.


Authors
-------

    Johannes Sahlmann

References
----------

    This code was partially adapted from Colin Cox' FGS2017.py, and
    FGS2017.py

    Created by Colin Cox on 2016-11-23.
    Fit using data from Julia Zhou's worksheeet


"""


from collections import OrderedDict
import os

import numpy as np
from astropy.time import Time

import pysiaf
from pysiaf.utils import tools, compare
from pysiaf.constants import JWST_SOURCE_DATA_ROOT, JWST_TEMPORARY_DATA_ROOT, JWST_DELIVERY_DATA_ROOT
from pysiaf import iando
from pysiaf.tests import test_aperture

import generate_reference_files

import importlib
importlib.reload(pysiaf.tools)
importlib.reload(generate_reference_files)

username = os.getlogin()
timestamp = Time.now()


#############################
instrument = 'FGS'
test_dir = os.path.join(JWST_TEMPORARY_DATA_ROOT, instrument, 'generate_test')

# regenerate SIAF reference files if needed
generate_basic_reference_files = False
if generate_basic_reference_files:
    generate_reference_files.generate_initial_siaf_aperture_definitions(instrument)
    generate_reference_files.generate_siaf_detector_layout()
    generate_reference_files.generate_siaf_detector_reference_file(instrument)
    generate_reference_files.generate_siaf_ddc_mapping_reference_file(instrument)

# DDC name mapping
ddc_apername_mapping = iando.read.read_siaf_ddc_mapping_reference_file(instrument)

# NIRSpec detected parameters, e.g. XDetSize
siaf_detector_parameters = iando.read.read_siaf_detector_reference_file(instrument)

# Fundamental aperture definitions: names, types, reference positions, dependencies
siaf_aperture_definitions = iando.read.read_siaf_aperture_definitions(instrument)

# definition of the master apertures, the 16 SCAs
detector_layout = iando.read.read_siaf_detector_layout()
master_aperture_names = detector_layout['AperName'].data

# directory containing reference files delivered by IDT
source_data_dir = os.path.join(JWST_SOURCE_DATA_ROOT, instrument)
print('Loading reference files from directory: {}'.format(source_data_dir))

generate_preflight_alignment_and_distortion_reference_files = False
if generate_preflight_alignment_and_distortion_reference_files:
    generate_reference_files.generate_siaf_pre_flight_reference_files_fgs()

if 0:
    generate_reference_files.generate_siaf_pre_flight_reference_files_fgs(mode='fsw')
    1/0

siaf_alignment_parameters = iando.read.read_siaf_alignment_parameters(instrument)

aperture_dict = OrderedDict()
aperture_name_list = siaf_aperture_definitions['AperName'].tolist()

for AperName in aperture_name_list:
    # child aperture to be constructed
    aperture = pysiaf.JwstAperture()
    aperture.AperName = AperName
    aperture.InstrName = siaf_detector_parameters['InstrName'][0].upper()

    aperture.XDetSize = siaf_detector_parameters['XDetSize'][0]
    aperture.YDetSize = siaf_detector_parameters['YDetSize'][0]
    aperture.AperShape = siaf_detector_parameters['AperShape'][0]

    aperture.DDCName = 'not set'
    aperture.Comment = None
    aperture.UseAfterDate = '2014-01-01'

    if AperName == 'J-FRAME':
        aperture_dict[AperName] = aperture
        continue

    aperture_definitions_index = siaf_aperture_definitions['AperName'].tolist().index(AperName)
    # Retrieve basic aperture parameters from definition files
    for attribute in 'XDetRef YDetRef AperType XSciSize YSciSize XSciRef YSciRef'.split():
        setattr(aperture, attribute, siaf_aperture_definitions[attribute][aperture_definitions_index])

    if aperture.AperType == 'OSS':
        aperture.DetSciYAngle = 0
        aperture.DetSciParity = 1
        aperture.VIdlParity = 1

    if AperName in ['FGS1_FULL', 'FGS1_FULL_OSS', 'FGS2_FULL', 'FGS2_FULL_OSS']:
        if AperName in detector_layout['AperName']:
            detector_layout_index = detector_layout['AperName'].tolist().index(AperName)
            for attribute in 'DetSciYAngle DetSciParity VIdlParity'.split():
                setattr(aperture, attribute, detector_layout[attribute][detector_layout_index])

        index = siaf_alignment_parameters['AperName'].tolist().index(AperName)
        aperture.V3SciYAngle = float(siaf_alignment_parameters['V3SciYAngle'][index])
        aperture.V3SciXAngle = float(siaf_alignment_parameters['V3SciXAngle'][index])
        aperture.V3IdlYAngle = float(siaf_alignment_parameters['V3IdlYAngle'][index])

        for attribute_name in 'V2Ref V3Ref'.split():
            setattr(aperture, attribute_name, siaf_alignment_parameters[attribute_name][index])

        polynomial_coefficients = iando.read.read_siaf_distortion_coefficients(instrument, AperName)

        number_of_coefficients = len(polynomial_coefficients)
        polynomial_degree = np.int((np.sqrt(8 * number_of_coefficients + 1) - 3) / 2)

        # set polynomial coefficients
        siaf_indices = ['{:02d}'.format(d) for d in polynomial_coefficients['siaf_index'].tolist()]
        for i in range(polynomial_degree + 1):
            for j in np.arange(i + 1):
                row_index = siaf_indices.index('{:d}{:d}'.format(i, j))
                for colname in 'Sci2IdlX Sci2IdlY Idl2SciX Idl2SciY'.split():
                    setattr(aperture, '{}{:d}{:d}'.format(colname, i, j), polynomial_coefficients[colname][row_index])

        aperture.Sci2IdlDeg = polynomial_degree
        aperture.complement()

    aperture_dict[AperName] = aperture

# Second pass to set parameters for apertures that depend on other apertures
for AperName in aperture_name_list:
    if AperName == 'J-FRAME':
        continue
    index = siaf_aperture_definitions['AperName'].tolist().index(AperName)
    aperture = aperture_dict[AperName]

    if (siaf_aperture_definitions['parent_apertures'][index] is not None) and (siaf_aperture_definitions['dependency_type'][index] == 'default'):

        aperture._parent_apertures = siaf_aperture_definitions['parent_apertures'][index]
        parent_aperture = aperture_dict[aperture._parent_apertures]

        for attribute in 'DetSciYAngle Sci2IdlDeg DetSciParity ' \
                         'VIdlParity V3IdlYAngle'.split(): # V3SciYAngle V3SciXAngle
            setattr(aperture, attribute, getattr(parent_aperture, attribute))

        aperture = tools.set_reference_point_and_distortion(instrument, aperture, parent_aperture)

        # see Excel spreadsheet -> Calc -> N85
        phi_y = np.rad2deg(np.arctan2(aperture.Sci2IdlX11*aperture.VIdlParity, aperture.Sci2IdlY11))
        aperture.V3SciYAngle = aperture.V3IdlYAngle + phi_y

        phi_x = np.rad2deg(np.arctan2(aperture.Sci2IdlX10*aperture.VIdlParity, aperture.Sci2IdlY10))
        aperture.V3SciXAngle = aperture.V3IdlYAngle + phi_x

        if 'MIMF' in AperName:
            for attribute in 'V3SciYAngle V3SciXAngle'.split():
                setattr(aperture, attribute, 0.)

    aperture.complement()

    # set Sci2IdlX11 to zero if it is very small
    coefficient_threshold = 1e-15
    if np.abs(aperture.Sci2IdlX11) < coefficient_threshold:
        aperture.Sci2IdlX11 = 0.

    aperture_dict[AperName] = aperture

# Set attributes for the special case of the J-FRAME aperture
aperture = aperture_dict['J-FRAME']
definition_index = siaf_aperture_definitions['AperName'].tolist().index(AperName)
for attributes, value in [('VIdlParity', 1),
                          ('AperType', siaf_aperture_definitions['AperType'][definition_index]),
                          ('XDetSize YDetSize DetSciYAngle DetSciParity', None),
                          ('XIdlVert2 XIdlVert3 YIdlVert1 YIdlVert2', 1000.),
                          ('XIdlVert1 XIdlVert4 YIdlVert3 YIdlVert4', -1000.)]:
    [setattr(aperture, attribute_name, value) for attribute_name in attributes.split()]
alignment_index = siaf_alignment_parameters['AperName'].tolist().index('J-FRAME')
for attribute_name in 'V3IdlYAngle V2Ref V3Ref'.split():
    setattr(aperture, attribute_name, siaf_alignment_parameters[attribute_name][alignment_index])
aperture_dict[AperName] = aperture

# sort SIAF entries in the order of the aperture definition file
aperture_dict = OrderedDict(sorted(aperture_dict.items(), key=lambda t: aperture_name_list.index(t[0])))

# third pass to set DDCNames apertures, which depend on other apertures
ddc_siaf_aperture_names = np.array([key for key in ddc_apername_mapping.keys()])
ddc_v2 = np.array([aperture_dict[aperture_name].V2Ref for aperture_name in ddc_siaf_aperture_names])
ddc_v3 = np.array([aperture_dict[aperture_name].V3Ref for aperture_name in ddc_siaf_aperture_names])
for AperName in aperture_name_list:
    if aperture_dict[AperName].AperType not in ['TRANSFORM']:
        separation_tel_from_ddc_aperture = np.sqrt((aperture_dict[AperName].V2Ref - ddc_v2)**2 + (aperture_dict[AperName].V3Ref - ddc_v3)**2)
        aperture_dict[AperName].DDCName = ddc_apername_mapping[ddc_siaf_aperture_names[np.argmin(separation_tel_from_ddc_aperture)]]

aperture_collection = pysiaf.ApertureCollection(aperture_dict)

emulate_delivery = True

if emulate_delivery:
    pre_delivery_dir = os.path.join(JWST_DELIVERY_DATA_ROOT, instrument)
    if not os.path.isdir(pre_delivery_dir):
        os.makedirs(pre_delivery_dir)

    # write the SIAF files to disk
    filenames = pysiaf.iando.write.write_jwst_siaf(aperture_collection, basepath=pre_delivery_dir,
                                                   file_format=['xml', 'xlsx']) #, label='update'

    pre_delivery_siaf = pysiaf.Siaf(instrument, basepath=pre_delivery_dir)

    for compare_to in [pysiaf.JWST_PRD_VERSION]:  # 'FGS_SIAF_2019-04-15']:

        if compare_to == 'PRDOPSSOC-M-024':
            prd_data_dir = pysiaf.constants.JWST_PRD_DATA_ROOT.rsplit('PRD', 1)[0]
            ref_siaf = pysiaf.Siaf(instrument,
                                   filename=os.path.join(prd_data_dir,
                                                         'PRDOPSSOC-M-024/SIAFXML/SIAFXML/FGS_SIAF.xml'))
        elif compare_to == 'FGS_SIAF_2019-04-15':
            prd_data_dir = pysiaf.constants.JWST_DELIVERY_DATA_ROOT.rsplit('pre', 1)[0]
            ref_siaf = pysiaf.Siaf(instrument,
                                   filename=os.path.join(prd_data_dir,
                                                         'temporary_data/FGS/FGS_SIAF_2019-04-15.xml'))
        elif compare_to == 'FGS_SIAF_bugfix-only':
            ref_siaf = pysiaf.Siaf(instrument,
                                   filename=os.path.join(pre_delivery_dir,
                                                         'FGS_SIAF_bugfix-only.xml'))
        else:
            # compare new SIAF with PRD version
            ref_siaf = pysiaf.Siaf(instrument)

        tags = {'reference': compare_to, 'comparison': 'pre_delivery'}

        compare.compare_siaf(pre_delivery_siaf, reference_siaf_input=ref_siaf,
                             fractional_tolerance=1e-6, report_dir=pre_delivery_dir,
                             tags=tags)

        compare.compare_transformation_roundtrip(pre_delivery_siaf, reference_siaf_input=ref_siaf,
                                                 tags=tags,
                                                 report_dir=pre_delivery_dir)

        compare.compare_inspection_figures(pre_delivery_siaf, reference_siaf_input=ref_siaf,
                                           report_dir=pre_delivery_dir, tags=tags)

    # run some tests on the new SIAF
    print('\nRunning aperture_transforms test for pre_delivery_siaf')
    test_aperture.test_jwst_aperture_transforms([pre_delivery_siaf])
    print('\nRunning aperture_vertices test for pre_delivery_siaf')
    test_aperture.test_jwst_aperture_vertices([pre_delivery_siaf])
