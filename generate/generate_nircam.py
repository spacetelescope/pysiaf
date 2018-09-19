#!/usr/bin/env python
"""Script to generate NIRCam SIAF content and files using pysiaf and flight-like SIAF reference files.


Authors
-------

    Johannes Sahlmann

References
----------
    Parts of the code were adapted from Colin Cox' nircamtrans.py


"""

from collections import OrderedDict
import os

import numpy as np
from astropy.table import Table

from pysiaf import iando
from pysiaf.utils import tools, compare
import generate_reference_files
from pysiaf.constants import JWST_SOURCE_DATA_ROOT, JWST_TEMPORARY_DATA_ROOT, JWST_DELIVERY_DATA_ROOT
import pysiaf.aperture

instrument = 'NIRCam'

# generate_reference_files.generate_siaf_xml_field_format_reference_files()

if 0:
    generate_reference_files.generate_siaf_detector_reference_file(instrument)
    generate_reference_files.generate_siaf_ddc_mapping_reference_file(instrument)

_ddc_apername_mapping = iando.read.read_siaf_ddc_mapping_reference_file(instrument)

siaf_xml_field_format = iando.read.read_siaf_xml_field_format_reference_file(instrument)
test_dir = os.path.join(JWST_TEMPORARY_DATA_ROOT, instrument, 'generate_test')
if not os.path.isdir(test_dir):
    os.makedirs(test_dir)

if 0:
    generate_reference_files.generate_initial_siaf_aperture_definitions(instrument)
    generate_reference_files.generate_siaf_pre_flight_reference_files_nircam()
    1/0
if 0:
    generate_reference_files.generate_siaf_pre_flight_reference_files_nircam()

wedge_file = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, '{}_siaf_wedge_offsets.txt'.format(instrument.lower()))
wedge_offsets = Table.read(wedge_file, format='ascii.basic', delimiter=',')

grism_file = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, '{}_siaf_grism_parameters.txt'.format(instrument.lower()))
grism_parameters = Table.read(grism_file, format='ascii.basic', delimiter=',')



siaf_detector_layout = iando.read.read_siaf_detector_layout()
siaf_alignment_parameters = iando.read.read_siaf_alignment_parameters(instrument)
siaf_aperture_definitions = iando.read.read_siaf_aperture_definitions(instrument)
# print(siaf_aperture_definitions)
siaf_detector_parameters = iando.read.read_siaf_detector_reference_file(instrument)

aperture_dict = OrderedDict()
aperture_name_list = siaf_aperture_definitions['AperName'].tolist()

master_aperture_names = siaf_detector_layout['AperName'].data

# First pass:
for AperName in aperture_name_list:
    # child aperture to be constructed
    aperture = pysiaf.JwstAperture()

    aperture.AperName = AperName
    aperture.InstrName = siaf_detector_parameters['InstrName'][
        0].upper()  # all capitals to conform with SIAFXML convention

    aperture.AperShape = siaf_detector_parameters['AperShape'][0]

    # Retrieve basic aperture parameters from definition files
    aperture_definitions_index = siaf_aperture_definitions['AperName'].tolist().index(AperName)

    for attribute in ['AperType']:
        setattr(aperture, attribute, siaf_aperture_definitions[attribute][aperture_definitions_index])

    if aperture.AperType not in ['COMPOUND', 'SLIT']:
        for attribute in 'XDetRef YDetRef AperType XSciSize YSciSize XSciRef YSciRef'.split():
            setattr(aperture, attribute, siaf_aperture_definitions[attribute][aperture_definitions_index])
        for attribute in 'XDetSize YDetSize'.split():
            setattr(aperture, attribute, siaf_detector_parameters[attribute][0])

    # process the 10 master apertures of NIRCam
    if AperName in master_aperture_names:

        detector_layout_index = siaf_detector_layout['AperName'].tolist().index(AperName)
        for attribute in 'DetSciYAngle DetSciParity VIdlParity'.split():
            setattr(aperture, attribute, siaf_detector_layout[attribute][detector_layout_index])

        index = siaf_alignment_parameters['AperName'].tolist().index(AperName)
        for attribute_name in 'V2Ref V3Ref V3SciYAngle V3SciXAngle V3IdlYAngle'.split():
            setattr(aperture, attribute_name, siaf_alignment_parameters[attribute_name][index])

        polynomial_coefficients = iando.read.read_siaf_distortion_coefficients(instrument, AperName)

        number_of_coefficients = len(polynomial_coefficients)
        polynomial_degree = np.int((np.sqrt(8 * number_of_coefficients + 1) - 3) / 2)
        aperture.Sci2IdlDeg = polynomial_degree

        # set polynomial coefficients
        siaf_indices = ['{:02d}'.format(d) for d in polynomial_coefficients['siaf_index'].tolist()]
        for i in range(polynomial_degree + 1):
            for j in np.arange(i + 1):
                row_index = siaf_indices.index('{:d}{:d}'.format(i, j))
                for colname in 'Sci2IdlX Sci2IdlY Idl2SciX Idl2SciY'.split():
                    setattr(aperture, '{}{:d}{:d}'.format(colname, i, j), polynomial_coefficients[colname][row_index])

        aperture.complement()

    aperture.DDCName = 'NOT_SET'
    aperture.UseAfterDate = '2014-01-01'


    aperture_dict[AperName] = aperture

#second pass to set parameters for apertures that depend on other apertures
# calculations emulate the Cox' Excel worksheets as described in JWST-
for AperName in aperture_name_list:
    index = siaf_aperture_definitions['AperName'].tolist().index(AperName)
    aperture = aperture_dict[AperName]

    parent_apertures = siaf_aperture_definitions['parent_apertures'][index]
    dependency_type = siaf_aperture_definitions['dependency_type'][index]


    if (parent_apertures is not None):

        if (dependency_type in ['default', 'wedge', 'dhspil_wedge']):
            aperture._parent_apertures = parent_apertures
            parent_aperture = aperture_dict[aperture._parent_apertures]

            # for attribute in 'V3SciXAngle V3SciYAngle DetSciYAngle Sci2IdlDeg VIdlParity V3IdlYAngle'.split():
            for attribute in 'DetSciYAngle Sci2IdlDeg DetSciParity VIdlParity'.split():
                setattr(aperture, attribute, getattr(parent_aperture, attribute))

            # set coefficients for OSS apertures
            if aperture.AperType == 'OSS':
                aperture.VIdlParity = 1
                aperture.DetSciParity = 1
                aperture.DetSciYAngle = 0.
                # compute V2Ref, V3Ref, distortion from XDetRef and YDetRef of aperture, based on the parent_aperture
                aperture = tools.set_reference_point_and_distortion(instrument, aperture, parent_aperture)
            else:
                # compute V2Ref, V3Ref, distortion from XDetRef and YDetRef of aperture, based on the parent_aperture
                aperture = tools.set_reference_point_and_distortion(instrument, aperture, parent_aperture)

            if dependency_type == 'wedge':
                sca_name = aperture.AperName[0:5]
                if (sca_name == 'NRCA5') and (('MASK335R' in aperture.AperName) or ('MASK430R' in aperture.AperName)):
                    # see https://jira.stsci.edu/browse/JWSTSIAF-77
                    sca_name += '335R430R'
                v2_offset = np.float(wedge_offsets['v2_offset'][wedge_offsets['name'] == sca_name])
                v3_offset = np.float(wedge_offsets['v3_offset'][wedge_offsets['name'] == sca_name])
                aperture.V2Ref += v2_offset
                aperture.V3Ref += v3_offset
            elif dependency_type == 'dhspil_wedge':
                aperture.V3Ref += 43.

            aperture.complement()

        elif dependency_type == 'nircam_compound':
            # COMPOUND apertures: V2,V3 reference points are linked to specific det points of one individual aperture
            # the order of the parent_apertures is defined by the sequence of corners

            aperture._parent_apertures = [s.strip() for s in parent_apertures.split(';')]

            for attribute in 'VIdlParity'.split():  # DetSciYAngle Sci2IdlDeg  DetSciParity
                setattr(aperture, attribute, getattr(aperture_dict[aperture._parent_apertures[0]], attribute))

            if AperName in ['NRCAS_FULL']:
                defining_aperture = aperture_dict['NRCA3_FULL']
            elif AperName in ['NRCBS_FULL', 'NRCALL_FULL']:
                defining_aperture = aperture_dict['NRCB4_FULL']

            aperture_definitions_index = siaf_aperture_definitions['AperName'].tolist().index(AperName)
            XDetRef = siaf_aperture_definitions['XDetRef'][aperture_definitions_index]
            YDetRef = siaf_aperture_definitions['YDetRef'][aperture_definitions_index]

            # set V2/V3 reference point corresponding to a detector pixel in one SCA
            aperture.V2Ref, aperture.V3Ref = defining_aperture.det_to_tel(XDetRef, YDetRef)

            # aperture.V2Ref = np.mean([aperture_dict[aperture_name].V2Ref for aperture_name in
            #                                          aperture._parent_apertures])
            # aperture.V3Ref = np.mean([aperture_dict[aperture_name].V3Ref for aperture_name in
            #                                          aperture._parent_apertures])

            # compute IdlCorners from V2V3 corners of individual apertures (which themselves are derived from the respective idl corners)
            compound_corners_Tel_x = np.zeros(4)
            compound_corners_Tel_y = np.zeros(4)
            for j, tmp_aperture in enumerate(
                    [aperture_dict[aper_name] for aper_name in aperture._parent_apertures]):
                corners_Tel_x, corners_Tel_y = tmp_aperture.corners('tel')
                compound_corners_Tel_x[j] = corners_Tel_x[j]
                compound_corners_Tel_y[j] = corners_Tel_y[j]

            # V3_IdlYAngle from compound corners in tel frame (see Intermediate worsksheet in excel SIAF)
            # argument order in np.arctan2 is opposite to excel's atan2
            aperture.V3IdlYAngle = np.rad2deg(np.arctan2(
                (compound_corners_Tel_x[2] + compound_corners_Tel_x[3])
                - (compound_corners_Tel_x[0] + compound_corners_Tel_x[1]),
                (compound_corners_Tel_y[2] + compound_corners_Tel_y[3])
                - (compound_corners_Tel_y[0] + compound_corners_Tel_y[1])
            ))

            # now we can compute the corners of the compound aperture in the tel frame
            compound_corners_Idl_x, compound_corners_Idl_y = aperture.convert(
                compound_corners_Tel_x, compound_corners_Tel_y, 'tel', 'idl')

            for j in range(4):
                setattr(aperture, 'XIdlVert{:d}'.format(j + 1), compound_corners_Idl_x[j])
                setattr(aperture, 'YIdlVert{:d}'.format(j + 1), compound_corners_Idl_y[j])

        elif dependency_type == 'grism_wfss':
            aperture._parent_apertures = parent_apertures
            parent_aperture = aperture_dict[aperture._parent_apertures]
            sca_name = AperName.split('_')[0]
            for attribute in 'VIdlParity'.split():  # DetSciYAngle Sci2IdlDeg  DetSciParity
                setattr(aperture, attribute, getattr(parent_aperture, attribute))

            if 'NRCA5' == sca_name:
                parent_aperture_name_for_distortion = 'NRCA5_FULL_OSS'
                parent_aperture_for_distortion = aperture_dict[parent_aperture_name_for_distortion]
            elif 'NRCB5' == sca_name:
                parent_aperture_name_for_distortion = 'NRCB5_FULL_OSS'
                parent_aperture_for_distortion = aperture_dict[parent_aperture_name_for_distortion]
            elif 'NRCALL' == sca_name:
                # those apertures depend on two different apertures for distortion and V2/3Ref
                # first/second corner depend on NRCA5 and third/fourth corner depend on NRCB5
                parent_aperture_name_for_distortion_12 = 'NRCA5_FULL_OSS'
                parent_aperture_name_for_distortion_34 = 'NRCB5_FULL_OSS'
                parent_aperture_for_distortion_12 = aperture_dict[parent_aperture_name_for_distortion_12]
                parent_aperture_for_distortion_34 = aperture_dict[parent_aperture_name_for_distortion_34]

            aperture.V2Ref = parent_aperture.V2Ref
            aperture.V3Ref = parent_aperture.V3Ref
            aperture.V3IdlYAngle = 0.

            # see WFSS worksheet is EXCEL SIAF
            grism_index = np.where(grism_parameters['aperture_name'] == aperture.AperName)[0][0]
            corners_Sci_x = np.array(
                [grism_parameters['DX{}'.format(j)][grism_index] for j in np.arange(4) + 1])
            corners_Sci_y = np.array(
                [grism_parameters['DY{}'.format(j)][grism_index] for j in np.arange(4) + 1])

            if sca_name in ['NRCA5', 'NRCB5']:
                # have to add parent_aperture_for_distortion.XSciRef because of special treatment in  grism_parameters of NRCB Y parameters
                tmp_corners_Idl_x, tmp_corners_Idl_y = parent_aperture_for_distortion.sci_to_idl(
                    corners_Sci_x + parent_aperture_for_distortion.XSciRef,
                    corners_Sci_y + parent_aperture_for_distortion.YSciRef)
                tmp2_corners_Idl_x = tmp_corners_Idl_x + parent_aperture_for_distortion.V2Ref
                tmp2_corners_Idl_y = tmp_corners_Idl_y + parent_aperture_for_distortion.V3Ref
            else:
                tmp_corners_Idl_12_x, tmp_corners_Idl_12_y = parent_aperture_for_distortion_12.sci_to_idl(
                    corners_Sci_x + parent_aperture_for_distortion_12.XSciRef,
                    corners_Sci_y + parent_aperture_for_distortion_12.YSciRef)
                tmp2_corners_Idl_12_x = tmp_corners_Idl_12_x + parent_aperture_for_distortion_12.V2Ref
                tmp2_corners_Idl_12_y = tmp_corners_Idl_12_y + parent_aperture_for_distortion_12.V3Ref
                tmp_corners_Idl_34_x, tmp_corners_Idl_34_y = parent_aperture_for_distortion_34.sci_to_idl(
                    corners_Sci_x + parent_aperture_for_distortion_34.XSciRef,
                    corners_Sci_y + parent_aperture_for_distortion_34.YSciRef)
                tmp2_corners_Idl_34_x = tmp_corners_Idl_34_x + parent_aperture_for_distortion_34.V2Ref
                tmp2_corners_Idl_34_y = tmp_corners_Idl_34_y + parent_aperture_for_distortion_34.V3Ref

                tmp2_corners_Idl_x = np.hstack((tmp2_corners_Idl_12_x[0:2], tmp2_corners_Idl_34_x[2:]))
                tmp2_corners_Idl_y = np.hstack((tmp2_corners_Idl_12_y[0:2], tmp2_corners_Idl_34_y[2:]))

            corners_Idl_x = -1 * (tmp2_corners_Idl_x - parent_aperture.V2Ref)
            corners_Idl_y = +1 * (tmp2_corners_Idl_y - parent_aperture.V3Ref)

            for j in range(4):
                setattr(aperture, 'XIdlVert{:d}'.format(j + 1), corners_Idl_x[j])
                setattr(aperture, 'YIdlVert{:d}'.format(j + 1), corners_Idl_y[j])

    aperture_dict[AperName] = aperture

#sort SIAF entries in the order of the aperture definition file
aperture_dict = OrderedDict(sorted(aperture_dict.items(), key=lambda t: aperture_name_list.index(t[0])))

#third pass to set DDCNames apertures, which depend on other apertures
ddc_siaf_aperture_names = np.array([key for key in _ddc_apername_mapping.keys()])
ddc_v2 = np.array([aperture_dict[aperture_name].V2Ref for aperture_name in ddc_siaf_aperture_names])
ddc_v3 = np.array([aperture_dict[aperture_name].V3Ref for aperture_name in ddc_siaf_aperture_names])
for AperName in aperture_name_list:
    # if AperName not in siaf_detector_layout['AperName']:
    #     continue
    aperture = aperture_dict[AperName]
    separation_tel_from_ddc_aperture = np.sqrt((aperture.V2Ref - ddc_v2)**2 + (aperture.V3Ref - ddc_v3)**2)
    aperture_dict[AperName].DDCName = _ddc_apername_mapping[ddc_siaf_aperture_names[np.argmin(separation_tel_from_ddc_aperture)]]

    # treat Sci2IdlX11
    Sci2IdlX11_treshold = 1e-15
    if (aperture.Sci2IdlX11 is not None) and (aperture.Sci2IdlX11 < Sci2IdlX11_treshold):
        aperture_dict[AperName].Sci2IdlX11 = 0.0



# fourth pass: internal verification
for AperName in aperture_name_list:
    aperture = aperture_dict[AperName]
    aperture.verify()



######################################
# SIAF content generation finished
######################################

aperture_collection = pysiaf.ApertureCollection(aperture_dict)

emulate_delivery = True
emulate_delivery = False

if emulate_delivery:
    pre_delivery_dir = os.path.join(JWST_DELIVERY_DATA_ROOT, instrument)
    if not os.path.isdir(pre_delivery_dir):
        os.makedirs(pre_delivery_dir)

    # write the SIAF files to disk
    filenames = pysiaf.iando.write.write_jwst_siaf(aperture_collection, basepath=pre_delivery_dir, file_format=['xml', 'xlsx'])

    pre_delivery_siaf = pysiaf.Siaf(instrument, basepath=pre_delivery_dir)

    # compare new SIAF with PRD version
    ref_siaf = pysiaf.Siaf(instrument)
    compare.compare_siaf(pre_delivery_siaf, reference_siaf_input=ref_siaf, fractional_tolerance=1e-6, report_dir=pre_delivery_dir, tags={'reference': pysiaf.JWST_PRD_VERSION, 'comparison': 'pre_delivery'})

    1/0
    # run some tests on the new SIAF
    from pysiaf.tests import test_aperture
    test_aperture.test_jwst_aperture_transforms([pre_delivery_siaf], verbose=False, threshold=0.1)
    test_aperture.test_jwst_aperture_vertices([pre_delivery_siaf])



    1/0


# write the SIAFXML to disk
filenames = pysiaf.iando.write.write_jwst_siaf(aperture_collection, basepath=test_dir, file_format=['xml'], label='pysiaf')
print('SIAFXML written in {}'.format(filenames[0]))

# compare to SIAFXML produced the old way
# ref_siaf = pysiaf.Siaf(instrument, os.path.join(test_dir , '{}'.format('NIRCam_SIAF_2017-12-01.xml')))
ref_siaf = pysiaf.Siaf(instrument)

if 0:
    pre_delivery_dir = os.path.join(JWST_DELIVERY_DATA_ROOT, instrument)
    pre_delivery_siaf = pysiaf.Siaf(instrument, basepath=pre_delivery_dir)
    ref_siaf = pre_delivery_siaf

new_siaf = pysiaf.Siaf(instrument, filenames[0])

# compare.compare_siaf(new_siaf, reference_siaf_input=ref_siaf, fractional_tolerance=1e-1, selected_aperture_name=master_aperture_names)#['NRCA3_FULL_OSS', 'NRCA1_FULL_OSS']) # 'NRCA4_SUB160', 'NRCA4_FULL', 'NRCA3_SUB160', 'NRCA3_FULL', 'NRCA5_SUB400P', 'NRCB5_SUB400P',
# compare.compare_siaf(new_siaf, reference_siaf_input=ref_siaf, fractional_tolerance=1e-1, selected_aperture_name=[s+'_OSS' for s in master_aperture_names])#['NRCA3_FULL_OSS', 'NRCA1_FULL_OSS']) # 'NRCA4_SUB160', 'NRCA4_FULL', 'NRCA3_SUB160', 'NRCA3_FULL', 'NRCA5_SUB400P', 'NRCB5_SUB400P',
# compare.compare_siaf(new_siaf, reference_siaf_input=ref_siaf, fractional_tolerance=1e-6, selected_aperture_name='NRCAS_FULL NRCBS_FULL NRCALL_FULL'.split())
# compare.compare_siaf(new_siaf, reference_siaf_input=ref_siaf, fractional_tolerance=1e-6, selected_aperture_name='NRCA5_FULL'.split())
ignore_attributes = 'XIdlVert1 XIdlVert2 XIdlVert3 XIdlVert4 YIdlVert1 YIdlVert2 YIdlVert3 YIdlVert4'.split()
from pysiaf.aperture import DISTORTION_ATTRIBUTES
ignore_attributes += ([s for s in DISTORTION_ATTRIBUTES if 'Idl2Sci' in s])
compare.compare_siaf(new_siaf, reference_siaf_input=ref_siaf, fractional_tolerance=1e-6, ignore_attributes=ignore_attributes)
# compare.compare_siaf(new_siaf, reference_siaf_input=ref_siaf, fractional_tolerance=1e-4, ignore_attributes=ignore_attributes, report_dir=test_dir)
# compare.compare_siaf(new_siaf, reference_siaf_input=ref_siaf, fractional_tolerance=1e-6)


roundtrip_table = compare.compare_transformation_roundtrip(new_siaf, reference_siaf_input=ref_siaf, report_dir=test_dir)
# roundtrip_table = compare.compare_transformation_roundtrip(new_siaf, reference_siaf_input=ref_siaf, selected_aperture_name='NRCA5_FULL'.split(), report_dir=test_dir, make_plot=True)
# roundtrip_table = compare.compare_transformation_roundtrip(new_siaf, reference_siaf_input=ref_siaf, selected_aperture_name=[s for s in aperture_name_list if 'FULL' in s], report_dir=test_dir, make_plot=False)

1/0

# illustrate fix of inverse NIRCam coefficients
if 0:
    import pylab as pl

    pl.close('all')

    fig = pl.figure(figsize=(12, 6), facecolor='w', edgecolor='k')
    pl.clf()
    # pl.subplot(2,1,1)
    pl.plot(roundtrip_table['siaf0_dx_mean'], 'b-', label='PRD dx_mean')
    pl.plot(roundtrip_table['siaf0_dy_mean'], 'r-', label='PRD dy_mean')
    pl.plot(roundtrip_table['siaf1_dx_mean'], 'ko--', label='fixed dx_mean')
    pl.plot(roundtrip_table['siaf1_dy_mean'], 'go--', label='fixed dy_mean')
    pl.title('Mean absolute difference')
    pl.legend()
    # pl.subplot(2,1,2)
    # pl.plot(roundtrip_table['siaf0_dx_rms'], 'b-', label='PRD dx_mean')
    # pl.plot(roundtrip_table['siaf0_dy_rms'], 'r-', label='PRD dy_mean')
    # pl.plot(roundtrip_table['siaf1_dx_rms'], 'ko--', label='fixed dx_mean')
    # pl.plot(roundtrip_table['siaf1_dy_rms'], 'go--', label='fixed dy_mean')
    # pl.title('RMS absolute difference')

    pl.xticks(np.arange(len(roundtrip_table)), roundtrip_table['AperName'], rotation='vertical')
    pl.margins(0.2)
    pl.subplots_adjust(bottom=0.15)
    pl.show()

    roundtrip_table[roundtrip_table['AperName']=='NRCA5_FULL'].pprint()



if 0:
    ref_siaf = pysiaf.Siaf(instrument, basepath=os.path.join(pysiaf.constants._DATA_ROOT, 'JWST','PRDOPSSOC-G-012','SIAFXML/SIAFXML'))
    new_siaf = pysiaf.Siaf(instrument)
    compare.compare_siaf(new_siaf, reference_siaf_input=ref_siaf, fractional_tolerance=1e-6)


