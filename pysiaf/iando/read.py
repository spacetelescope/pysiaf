"""Functions to read Science Instrument Aperture Files (SIAF) and SIAF reference files.

For JWST SIAF, reading XML and CSV format are supported.

For HST SIAF, only .dat files can be read.


Authors
-------

    Johannes Sahlmann

References
----------
    Parts of read_hst_siaf were adapted from Matt Lallo's plotap.f.

    Parts of read_jwst_siaf were adapted from jwxml.

"""

from collections import OrderedDict
import numpy as np
import os

from astropy.table import Table
import lxml.etree as ET

from .. import aperture
from .. import siaf
from ..constants import HST_PRD_DATA_ROOT, JWST_PRD_DATA_ROOT, JWST_SOURCE_DATA_ROOT


def get_siaf(input_siaf, observatory='JWST'):
    """Return a Siaf object corresponding to input_siaf which can be a string path or a Siaf object.

    Parameters
    ----------
    input_siaf
    observatory

    Returns
    -------
    siaf_object: pysiaf.Siaf
        Siaf object

    """
    if type(input_siaf) == str:
        aperture_collection = read_jwst_siaf(filename=input_siaf)

        # initilize siaf as empty object
        siaf_object = siaf.Siaf(None)
        siaf_object.instrument = aperture_collection[list(aperture_collection.items())[0][0]].InstrName

        if siaf_object.instrument == 'NIRCAM':
            siaf_object.instrument = 'NIRCam'
        elif siaf_object.instrument == 'NIRSPEC':
            siaf_object.instrument = 'NIRSpec'

        siaf_object.apertures = aperture_collection
        siaf_object.description = os.path.basename(input_siaf)
        siaf_object.observatory = observatory

    elif type(input_siaf) == siaf.Siaf:
        siaf_object = input_siaf
        siaf_object.description = 'pysiaf.Siaf object'
    else:
        raise TypeError('Input has to be either a full path or a Siaf object.')

    return siaf_object


def read_hst_siaf(file=None):#, AperNames=None):
    """Read apertures from HST SIAF file and return a collection.

    This was partially ported from Lallo's plotap.f.

    Parameters
    ----------
    file : str
    AperNames : str list

    Returns
    -------
    apertures: dict
        Dictionary of apertures

    """
    if file is None:
        file = os.path.join(HST_PRD_DATA_ROOT, 'siaf.dat')

    # read all lines
    siaf_stream = open(file)
    data = siaf_stream.readlines()
    siaf_stream.close()

    # initialize dict of apertures
    apertures = OrderedDict()

    # inspect SIAF and populate Apertures
    CAJ_index = 0
    CAK_index = 0

    for l, text in enumerate(data):
        skip_aperture = False
        if (text.rstrip()[-3::] == 'CAJ') & (CAJ_index == 0):
            a = aperture.HstAperture()
            # Process the first 'CAJ' record.
            a.ap_name = text[0:10].strip()  # Aperture Identifier.
            a.v2_cent = np.float(text[10:25])  # SICS V2 Center. (same as a_v2_ref)
            a.v3_cent = np.float(text[25:40])  # SICS V3 Center. (same as a_v3_ref)
            a.a_shape = text[40:44]  # Aperture Shape.
            try:
                a.maj = np.float(text[44:59])  # Major Axis Dimension.
            except ValueError:  # when field is empty
                a.maj = None
            a.Mac_Flag = text[59]  # !SI Macro Aperture Flag.
            a.BR_OBJ_Flag = text[60]  # !Bright Object Alert Flag.
            a.brt_obj_thres = text[61:66]  # !Bright Object Alert Threshold.
            a.Macro_ID = text[66:70]  # !SI Macro Aperture Identifier.
            rec_type = text[70:73]  # !Record type.
            CAJ_index = 1
            aperture_name = a.ap_name

        elif (text.rstrip()[-3::] == 'CAJ') & (CAJ_index == 1):
            # Process the second 'CAJ' record.
            try:
                a.min = np.float(text[0:15])  # !Minor Axis Dimension.
            except ValueError:  # when field is empty
                a.min = None
            a.plate_scale = np.float(text[15:30])  # !Arcsecond per Pixel plate scale.
            a.a_area = np.float(text[30:45])  # !Area of SI Aperture.
            a.theta = np.float(text[45:60])  # !Aperture Rotation Angle.
            a.SIAS_Flag = text[60]  # !SIAS coordinate system flag. (If set then AK rec.)
            rec_type = text[70:73]  # !Record type.
            CAJ_index = 2

        elif (text.rstrip()[-3::] == 'CAJ') & (CAJ_index == 2):
            # Process the third 'CAJ' record.
            a.im_par = np.int(text[0:2])  # Image Parity.
            a.ideg = np.int(text[2])  # !Polynomial Degree.
            a.xa0 = np.float(text[3:18])  # !SIAS X Center. -> like JWST SCIENCE frame
            a.ya0 = np.float(text[18:33])  # !SIAS Y Center.
            a.xs0 = np.float(text[33:48])  # !SICS X Center. -> like JWST IDEAL frame
            a.ys0 = np.float(text[48:63])  # !SICS Y Center.
            rec_type = text[70:73]  # !Record type.
            CAJ_index = 0

        elif text.rstrip()[-2::] == 'AJ':
            a.SI_mne = text[0:4].strip()  # !Science Instrument Mnemonic
            a.Tlm_mne = text[4]  # !SI Telemetry Mnemonic.
            a.Det_mne = text[5]  # !SI Detector Mnemonic.
            a.A_mne = text[6:10]  # !SI Aperture Mnemonic.
            a.APOS_mne = text[10]  # !SI Aperture Position Mnemonic.
            rec_type = text[70:73]  # !Record type.

        elif text.rstrip()[-3::] == 'CAQ':
            a.v1x = np.float(text[0:15])  # !SICS Vertex 1_X -> like JWST IDEAL frame
            a.v1y = np.float(text[15:30])  # !SICS Vertex 1_Y
            a.v2x = np.float(text[30:45])  # !SICS Vertex 2_X
            a.v2y = np.float(text[45:60])  # !SICS Vertex 2_Y
            rec_type = text[70:73]  # !Record type.

        elif text.rstrip()[-2::] == 'AQ':
            a.v3x = np.float(text[0:15])  # !SICS Vertex 3_X
            a.v3y = np.float(text[15:30])  # !SICS Vertex 3_Y
            a.v4x = np.float(text[30:45])  # !SICS Vertex 4_X
            a.v4y = np.float(text[45:60])  # !SICS Vertex 4_Y
            rec_type = text[70:73]  # !Record type.

        elif text.rstrip()[-2::] == 'AP':
            # FGS pickles
            a.pi_angle = np.float(text[0:15])  # !Inner Radius Orientation Angle.
            a.pi_ext = np.float(text[15:30])  # !Angular Extent of the Inner Radius.
            a.po_angle = np.float(text[30:45])  # !Outer Radius Orientation Angle.
            a.po_ext = np.float(text[45:60])  # !Angular Extent of the Outer Radius.
            rec_type = text[70:73]  # !Record type.

        elif text.rstrip()[-2::] == 'AM':
            a.a_v2_ref = np.float(text[0:15])  # !V2 Coordinate of Aperture Reference Point. (same as v2_cent)
            a.a_v3_ref = np.float(text[15:30])  # !V3 Coordinate of Aperture Reference Point. (same as v3_cent)
            a.a_x_incr = np.float(text[30:45])  # !First Coordinate Axis increment.
            a.a_y_incr = np.float(text[45:60])  # !Second Coordinate Axis increment.

        elif text.rstrip()[-2::] == 'AN':
            if (a.a_shape == 'PICK') and ('FGS' in a.ap_name):
                # HST FGS are special in the sense that the idl_to_tel transformation is implemented via the TVS matrix
                # and not the standard way
                # a.set_fgs_tel_reference_point(a.a_v2_ref, a.a_v2_ref)
                a.set_idl_reference_point(a.a_v2_ref, a.a_v3_ref, verbose=False)
                # pass

            if (a.a_shape == 'PICK') | (a.a_shape == 'CIRC'):
                # TO BE IMPLEMENTED
                # FGS pickle record ends here
                # apertures.append(a)
                #                             read(10,1250)Beta1,     !Angle of increasing first coordinate axis.
                #      *               Beta2,     !Angle of increasing second coordinate axis.
                #      *               a_x_ref,   !X reference.
                #      *               a_y_ref,   !Y reference.
                #      *               X_TOT_PIX, !Total X-axis pixels.
                #      *               Y_TOT_PIX, !Total Y-axis pixels.
                #      *               rec_type   !Record type.
                #  1250   format(4(G15.8),2(I5),a3)
                # apertures.append(a)
                apertures[a.AperName] = a

        elif (text.rstrip()[-3::] == 'CAK') & (CAK_index == 0):
            # Process the first 'CAK' record.
            n_polynomial_coefficients = np.int(((a.ideg + 1) * (a.ideg + 2)) / 2)
            # the order is
            # SIAS to SICS X Transformation.
            # SIAS to SICS Y Transformation.
            # SICS to SIAS X Transformation.
            # SICS to SIAS X Transformation.

            polynomial_coefficients = np.ones((n_polynomial_coefficients, 4)) * -99
            for jj in np.arange(4):
                polynomial_coefficients[CAK_index, jj] = np.float(text[15 * jj:15 * (jj + 1)])
            CAK_index += 1

        elif (text.rstrip()[-3::] == 'CAK') & (CAK_index != 0):
            # Process the remaining 'CAK' records
            for jj in np.arange(4):
                polynomial_coefficients[CAK_index, jj] = np.float(text[15 * jj:15 * (jj + 1)])
            CAK_index += 1

        elif text.rstrip()[-2::] == 'AK':
            # Process the last polynomial coefficient record.
            for jj in np.arange(4):
                polynomial_coefficients[CAK_index, jj] = np.float(text[15 * jj:15 * (jj + 1)])
            a.polynomial_coefficients = polynomial_coefficients
            CAK_index = 0

            apertures[a.AperName] = a
            # apertures.append(a)

    return apertures


def get_jwst_siaf_instrument(tree):
    """Return the instrument specified in the first aperture of a SIAF xml tree.

    Returns
    -------
    instrument : str
        All Caps instrument name, e.g. NIRSPEC

    """
    for entry in tree.getroot().iter('SiafEntry'):
        for node in entry.iterchildren():
            if node.tag == 'InstrName':
                return node.text


def read_jwst_siaf(instrument=None, filename=None, basepath=None):
    """Read the JWST SIAF and return a collection of apertures.

    Parameters
    ----------
    instrument
    filename
    basepath

    Returns
    -------
    apertures : dict
        dictionary of apertures

    """
    if (filename is None) and (instrument is None):
        raise ValueError('Specify either input instrument or filename')

    if filename is None:
        if basepath is None:
            basepath = JWST_PRD_DATA_ROOT
        if not os.path.isdir(basepath):
            raise RuntimeError("Could not find SIAF data "
                               "in {}".format(basepath))
        filename = os.path.join(basepath, instrument + '_SIAF.xml')
    else:
        filename = filename

    apertures = OrderedDict()

    file_seed, file_extension = os.path.splitext(filename)
    if file_extension == '.xml':
        tree = ET.parse(filename)
        instrument = get_jwst_siaf_instrument(tree)

        # generate Aperture objects from SIAF XML file, parse the XML
        for entry in tree.getroot().iter('SiafEntry'):
            if instrument.upper() == 'NIRSPEC':
                jwst_aperture = aperture.NirspecAperture()
            else:
                jwst_aperture = aperture.JwstAperture()
            for node in entry.iterchildren():
                if (node.tag in aperture.ATTRIBUTES_THAT_CAN_BE_NONE) and (node.text is None):
                    value = node.text
                elif node.tag in aperture.INTEGER_ATTRIBUTES:
                    try:
                        value = int(node.text)
                    except TypeError:
                        print('{}: {}'.format(node.tag, node.text))
                        raise TypeError
                elif node.tag in aperture.STRING_ATTRIBUTES:
                    value = node.text
                else:
                    try:
                        value = float(node.text)
                    except TypeError:
                        print('{}: {}'.format(node.tag, node.text))
                        raise TypeError

                # except (ValueError, TypeError):
                #     value = node.text

                setattr(jwst_aperture, node.tag, value)

            apertures[jwst_aperture.AperName] = jwst_aperture

    else:
        raise NotImplementedError

    # handle special case of NIRSpec, where auxiliary TRANSFORM apertures are defined and hold transformation parameters
    # simple workaround is to attach the TRANSFORM aperture as attribute to the respective NIRSpec aperture
    if instrument.upper() == 'NIRSPEC':
        for AperName in apertures:
            jwst_aperture = apertures[AperName]
            if jwst_aperture.AperType in ['FULLSCA', 'OSS']:
                for transform_aperture_name in 'CLEAR_GWA_OTE F110W_GWA_OTE F140X_GWA_OTE'.split():
                    setattr(jwst_aperture, '_{}'.format(transform_aperture_name), apertures[transform_aperture_name])
                apertures[AperName] = jwst_aperture

    return apertures


def read_siaf_alignment_parameters(instrument):
    """Return astropy table.

    Parameters
    ----------
    instrument

    Returns
    -------
    : astropy table

    """
    filename = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, '{}_siaf_alignment.txt'.format(instrument.lower()))
    return Table.read(filename, format='ascii.basic', delimiter=',')


def read_siaf_aperture_definitions(instrument):
    """Return astropy table.

    Parameters
    ----------
    instrument

    Returns
    -------
    : astropy table

    """
    filename = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, '{}_siaf_aperture_definition.txt'.format(instrument.lower()))

    # converters = {'XDetRef': [ascii.convert_numpy(np.float32)]}
    # , converters = converters, guess = False
    return Table.read(filename, format='ascii.basic', delimiter=',', fill_values=('None', 0))


def read_siaf_ddc_mapping_reference_file(instrument):
    """Return dictionary with the DDC mapping.

    Parameters
    ----------
    instrument

    Returns
    -------
    : astropy table

    """
    ddc_mapping_file = os.path.join(JWST_SOURCE_DATA_ROOT, instrument,
                                    '{}_siaf_ddc_apername_mapping.txt'.format(instrument.lower()))


    ddc_mapping_table = Table.read(ddc_mapping_file, format='ascii.basic', delimiter=',')

    #generate dictionary
    _ddc_apername_mapping = {}
    for j, siaf_name in enumerate(ddc_mapping_table['SIAF_NAME'].data):
        _ddc_apername_mapping[siaf_name] = ddc_mapping_table['DDC_NAME'][j]

    return _ddc_apername_mapping


def read_siaf_detector_layout():
    """Return the SIAF detector layout read from the SIAF reference file.

    Returns
    -------
    : astropy table

    """
    layout_file = os.path.join(JWST_SOURCE_DATA_ROOT, 'siaf_detector_layout.txt')

    return Table.read(layout_file, format='ascii.basic', delimiter=',')


def read_siaf_detector_reference_file(instrument):
    """Return astropy table.

    Parameters
    ----------
    instrument

    Returns
    -------
    : astropy table

    """
    filename = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, '{}_siaf_detector_parameters.txt'.format(instrument.lower()))

    return Table.read(filename, format='ascii.basic', delimiter=',')


def read_siaf_distortion_coefficients(instrument, aperture_name):
    """Return astropy table.

    Parameters
    ----------
    instrument
    aperture_name

    Returns
    -------
    : astropy table

    """
    distortion_reference_file_name = os.path.join(JWST_SOURCE_DATA_ROOT, instrument,
                                                  '{}_siaf_distortion_{}.txt'.format(instrument.lower(), aperture_name.lower()))

    return Table.read(distortion_reference_file_name, format='ascii.basic', delimiter=',')


def read_siaf_xml_field_format_reference_file(instrument):
    """Return astropy table.

    Parameters
    ----------
    instrument

    Returns
    -------
    : astropy table

    """
    filename = os.path.join(JWST_SOURCE_DATA_ROOT, instrument, '{}_siaf_xml_field_format.txt'.format(instrument.lower()))
    return Table.read(filename, format='ascii.basic', delimiter=',')
