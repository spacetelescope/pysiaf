#!/usr/bin/env python
"""Script to generate NIRISS SIAF content and files using pysiaf and flight-like SIAF reference files


Authors
-------

    Johannes Sahlmann

References
----------

    Parts of the code were adapted from Colin Cox' makeSIAF.py

    For a detailed description of the NIRISS SIAF, the underlying reference files, and the
    transformations, see Goudfrooij & Cox, 2018: The Pre-Flight SI Aperture File, Part 5: NIRISS
    (JWST-STScI-006317).
"""

from collections import OrderedDict
import os

import numpy as np

import pysiaf
from pysiaf.utils import tools, compare
from pysiaf.constants import JWST_SOURCE_DATA_ROOT, JWST_TEMPORARY_DATA_ROOT, \
    JWST_DELIVERY_DATA_ROOT
from pysiaf import iando

import generate_reference_files

#############################
instrument = 'NIRISS'

test_dir = os.path.join(JWST_TEMPORARY_DATA_ROOT, instrument, 'generate_test')

# regenerate SIAF reference files if needed
regenerate_basic_reference_files = False
if regenerate_basic_reference_files:
    # generate_siaf_detector_layout()
    # generate_reference_files.generate_initial_siaf_aperture_definitions(instrument)
    # generate_siaf_detector_reference_file(instrument)
    # generate_siaf_ddc_mapping_reference_file(instrument)
    distortion_file_name = os.path.join(JWST_SOURCE_DATA_ROOT, instrument,
                                        'niriss_astrometric_coefficients_august_2016_with_header.txt')
    generate_reference_files.generate_siaf_pre_flight_reference_files_niriss(distortion_file_name)

# DDC name mapping
_ddc_apername_mapping = iando.read.read_siaf_ddc_mapping_reference_file(instrument)

# NIRISS detector parameters, e.g. XDetSize
siaf_detector_parameters = iando.read.read_siaf_detector_reference_file(instrument)

detector_layout = iando.read.read_siaf_detector_layout()
siaf_alignment_parameters = iando.read.read_siaf_alignment_parameters(instrument)

# Fundamental aperture definitions: names, types, reference positions, dependencies
siaf_aperture_definitions = iando.read.read_siaf_aperture_definitions(instrument)

aperture_dict = {}
aperture_name_list = siaf_aperture_definitions['AperName'].tolist()

for AperName in aperture_name_list:
    # child aperture to be constructed
    aperture = pysiaf.JwstAperture()
    aperture.AperName = AperName
    aperture.InstrName = siaf_detector_parameters['InstrName'][0].upper()  # all capitals. OK

    aperture.XDetSize = siaf_detector_parameters['XDetSize'][0]
    aperture.YDetSize = siaf_detector_parameters['YDetSize'][0]
    aperture.AperShape = siaf_detector_parameters['AperShape'][0]
    aperture.DetSciParity = 1

    aperture_definitions_index = siaf_aperture_definitions['AperName'].tolist().index(AperName)

    # Retrieve basic aperture parameters from definition files
    for attribute in 'XDetRef YDetRef AperType XSciSize YSciSize XSciRef YSciRef'.split():
        setattr(aperture, attribute, siaf_aperture_definitions[attribute][aperture_definitions_index])

    if siaf_aperture_definitions['AperType'][aperture_definitions_index] == 'OSS':
        aperture.DetSciYAngle = 0
        aperture.DetSciParity = 1
        aperture.VIdlParity = 1  # -> move to NIS_CEN aperture

    if AperName in ['NIS_CEN', 'NIS_CEN_OSS']:
        if AperName in detector_layout['AperName']:
            detector_layout_index = detector_layout['AperName'].tolist().index(AperName)
            for attribute in 'DetSciYAngle DetSciParity VIdlParity'.split():
                setattr(aperture, attribute, detector_layout[attribute][detector_layout_index])

        index = siaf_alignment_parameters['AperName'].tolist().index(AperName)
        aperture.V3SciYAngle = siaf_alignment_parameters['V3SciYAngle'][index]
        aperture.V3SciXAngle = siaf_alignment_parameters['V3SciXAngle'][index]
        aperture.V3IdlYAngle = siaf_alignment_parameters['V3IdlYAngle'][index]
        # aperture.V3IdlYAngle = tools.v3sciyangle_to_v3idlyangle(aperture.V3SciYAngle)

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

    else:
        aperture.DetSciYAngle = 180
        aperture.VIdlParity = -1

    aperture.Sci2IdlDeg = polynomial_degree
    aperture_dict[AperName] = aperture

# second pass to set parameters for apertures that depend on other apertures
# calculations emulate the Cox' Excel worksheets as described in JWST-01550
# NIRISS is the same as FGS
for AperName in aperture_name_list:
    index = siaf_aperture_definitions['AperName'].tolist().index(AperName)
    aperture = aperture_dict[AperName]

    if (siaf_aperture_definitions['parent_apertures'][index] is not None) and (siaf_aperture_definitions['dependency_type'][index] == 'default'):
        aperture._parent_apertures = siaf_aperture_definitions['parent_apertures'][index]
        parent_aperture = aperture_dict[aperture._parent_apertures]

        aperture.V3SciYAngle = parent_aperture.V3SciYAngle
        aperture.V3SciXAngle = parent_aperture.V3SciXAngle

        aperture.V3IdlYAngle = tools.v3sciyangle_to_v3idlyangle(aperture.V3SciYAngle)

        aperture = tools.set_reference_point_and_distortion(instrument, aperture, parent_aperture)

    aperture.complement()
    aperture.Comment = None
    aperture.UseAfterDate = '2014-01-01'
    aperture_dict[AperName] = aperture

# sort SIAF entries in the order of the aperture definition file
aperture_dict = OrderedDict(sorted(aperture_dict.items(), key=lambda t: aperture_name_list.index(t[0])))

# third pass to set DDCNames apertures, which depend on other apertures
ddc_siaf_aperture_names = np.array([key for key in _ddc_apername_mapping.keys()])
ddc_v2 = np.array([aperture_dict[aperture_name].V2Ref for aperture_name in ddc_siaf_aperture_names])
ddc_v3 = np.array([aperture_dict[aperture_name].V3Ref for aperture_name in ddc_siaf_aperture_names])
for AperName in aperture_name_list:
    separation_tel_from_ddc_aperture = np.sqrt((aperture_dict[AperName].V2Ref - ddc_v2)**2 +
                                               (aperture_dict[AperName].V3Ref - ddc_v3)**2)
    aperture_dict[AperName].DDCName = _ddc_apername_mapping[ddc_siaf_aperture_names[np.argmin(
        separation_tel_from_ddc_aperture)]]

######################################
# SIAF content generation finished
######################################

aperture_collection = pysiaf.ApertureCollection(aperture_dict)

emulate_delivery = True

if emulate_delivery:
    pre_delivery_dir = os.path.join(JWST_DELIVERY_DATA_ROOT, instrument)
    if not os.path.isdir(pre_delivery_dir):
        os.makedirs(pre_delivery_dir)

    # write the SIAF files to disk
    filenames = pysiaf.iando.write.write_jwst_siaf(aperture_collection, basepath=pre_delivery_dir,
                                                   file_format=['xml', 'xlsx'])

    pre_delivery_siaf = pysiaf.Siaf(instrument, basepath=pre_delivery_dir)

    compare_against_prd = True
    compare_against_cdp7b = True

    # print('\nRunning regression test of pre_delivery_siaf against test_data:')
    # None yet

    for compare_to in [pysiaf.JWST_PRD_VERSION]:
        if compare_to == 'cdp7b':
            ref_siaf = pysiaf.Siaf(instrument,
                                   filename=os.path.join(pre_delivery_dir, 'NIRISS_SIAF_cdp7b.xml'))
        else:
            # compare new SIAF with PRD version
            ref_siaf = pysiaf.Siaf(instrument)

        tags = {'reference': compare_to, 'comparison': 'pre_delivery'}

        compare.compare_siaf(pre_delivery_siaf, reference_siaf_input=ref_siaf,
                             fractional_tolerance=1e-6, report_dir=pre_delivery_dir, tags=tags)

        compare.compare_transformation_roundtrip(pre_delivery_siaf,
                                                 reference_siaf_input=ref_siaf, tags=tags,
                                                 report_dir=pre_delivery_dir, )

        compare.compare_inspection_figures(pre_delivery_siaf, reference_siaf_input=ref_siaf,
                                           report_dir=pre_delivery_dir, tags=tags,
                                           xlimits=(-210, -370), ylimits=(-780, -615))

    # run some tests on the new SIAF
    from pysiaf.tests import test_aperture
    print('\nRunning aperture_transforms test for pre_delivery_siaf')
    test_aperture.test_jwst_aperture_transforms([pre_delivery_siaf], verbose=True, threshold=0.04)
    print('\nRunning aperture_vertices test for pre_delivery_siaf')
    test_aperture.test_jwst_aperture_vertices([pre_delivery_siaf])

else:

    test_dir = os.path.join(JWST_TEMPORARY_DATA_ROOT, instrument, 'generate_test')
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)

    # write the SIAFXML to disk
    [filename] = pysiaf.iando.write.write_jwst_siaf(aperture_collection, basepath=test_dir,
                                                    file_format=['xml'])
    print('SIAFXML written in {}'.format(filename))

