"""Script to generate NIRISS, FGS, NIRSpec SIAF files with homogeneous XML formatting.

See https://jira.stsci.edu/browse/JWSTSIAF-120 for details.


Authors
-------

    Johannes Sahlmann

"""
import os

from astropy.table import Table
import lxml.etree as ET

import pysiaf
from pysiaf.constants import JWST_DELIVERY_DATA_ROOT
from pysiaf.utils import compare
from pysiaf.tests import test_aperture
from pysiaf.tests import test_nirspec

show_field_formats = True

siaf_detector_layout = pysiaf.iando.read.read_siaf_detector_layout()



def read_xml_field_formats(instrument, filename, verbose=False):
    """Extract the SIAF xml field formats.

    Parameters
    ----------
    instrument
    filename
    verbose

    Returns
    -------
    T : astropy table

    """
    T = Table()
    print('*' * 100)
    print('{}'.format(instrument))
    # filename = os.path.join(basepath, instrument + '_SIAF.xml')
    primary_master_aperture_name = \
    siaf_detector_layout['AperName'][siaf_detector_layout['InstrName'] == instrument.upper()][0]

    tree = ET.parse(filename)

    # generate Aperture objects from SIAF XML file, parse the XML
    for entry in tree.getroot().iter('SiafEntry'):
        show = False
        field_index = 1
        for node in entry.iterchildren():
            # show = False
            field_number = '1.{:d}'.format(field_index)
            # print('{}   {}: {}'.format(primary_master_aperture_name, node.tag, node.text))
            # if (node.tag == 'InstrName'):
            #     show = True
            if (node.tag == 'AperName') and (node.text == primary_master_aperture_name):
                show = True
            if (node.tag in pysiaf.aperture.ATTRIBUTES_THAT_CAN_BE_NONE) and (node.text is None):
                value = node.text
                prd_data_class = 'None'
                python_format = 'None'
            elif node.tag in pysiaf.aperture.INTEGER_ATTRIBUTES:
                prd_data_class = 'int'
                python_format = 'd'
                try:
                    value = int(node.text)
                except (TypeError):
                    print('{}: {}'.format(node.tag, node.text))
                    raise TypeError
            elif (node.tag in pysiaf.aperture.STRING_ATTRIBUTES):
                prd_data_class = 'string'
                python_format = ''
                value = node.text
            else:
                prd_data_class = 'float'

                try:
                    decimal_part = node.text.split('.')[1]

                    if 'E' in decimal_part:
                        decimal_precision = len(decimal_part.split('E')[0])
                        python_format = '.{}e'.format(decimal_precision)
                    elif 'e' in decimal_part:
                        decimal_precision = len(decimal_part.split('e')[0])
                        python_format = '.{}e'.format(decimal_precision)
                    else:
                        decimal_precision = len(decimal_part)
                        python_format = '.{}f'.format(decimal_precision)

                    try:
                        value = float(node.text)
                    except (TypeError):
                        print('{}: {}'.format(node.tag, node.text))
                        raise TypeError

                except IndexError:  # e.g. MIRI 'DetSciYAngle'
                    prd_data_class = 'int'
                    python_format = 'd'
                    raise IndexError

            if show:
                if verbose:
                    print('{:5} {} {:10} {:10} {:30}'.format(field_number, node.tag, prd_data_class,
                                                         python_format, str(node.text)))
                if len(T) == 0:
                    T['field_nr'] = ['{:>5}'.format(field_number)]
                    T['field_name'] = ['{:>20}'.format(node.tag)]
                    T['format'] = ['{:>10}'.format(prd_data_class)]
                    T['pyformat'] = ['{:>10}'.format(python_format)]
                    T['example'] = ['{:>30}'.format(node.text)]
                else:
                    T.add_row((field_number, node.tag, prd_data_class, python_format, node.text))

            field_index += 1

    return T


def show_xml_field_formats(xml_formats, reference_instrument_name='NIRISS', out_dir=None):

    reference_table = xml_formats[reference_instrument_name]

    for col in ['format', 'pyformat', 'example']:
        reference_table.rename_column(col, '{}_{}'.format(reference_instrument_name, col).lower())
    for key in xml_formats.keys():
        if key != reference_instrument_name:
            for col in ['format', 'pyformat', 'example']:
                reference_table['{}_{}'.format(key, col).lower()] = xml_formats[key][col]

    columns_to_show = ['field_nr', 'field_name']
    for col in reference_table.colnames:
        if ('pyformat' in col) or ('format' in col):  # col.split('_')[1] in ['Format', 'pyformat']:
            columns_to_show.append(col)
    columns_to_show.append('{}_example'.format(reference_instrument_name.lower()))

    tag_list = list(reference_table['field_name'])
    rows_to_delete = [i for i in range(len(reference_table)) if ('Sci2IdlX' in tag_list[i] or
                                                                 'Sci2IdlY' in tag_list[i] or
                                                                 'Idl2SciX' in tag_list[i] or
                                                                 'Idl2SciY' in tag_list[i]) and (tag_list[i] not in ['Sci2IdlX00', 'Idl2SciY00'])]

    reference_table.remove_rows(rows_to_delete)

    reference_table[columns_to_show].pprint()

    if out_dir is None:
        out_dir = os.environ['HOME']
    reference_table[columns_to_show].write(os.path.join(out_dir, 'siaf_field_formats.csv'),
                                           format='ascii.basic', delimiter=',', overwrite=True)


xml_formats = {}
for instrument in ['NIRISS', 'FGS', 'NIRSpec']:

    siaf = pysiaf.Siaf(instrument)

    pre_delivery_dir = os.path.join(JWST_DELIVERY_DATA_ROOT, instrument)
    if not os.path.isdir(pre_delivery_dir):
        os.makedirs(pre_delivery_dir)

    # write the SIAF files to disk
    filenames = pysiaf.iando.write.write_jwst_siaf(siaf, basepath=pre_delivery_dir,
                                                   file_format=['xml', 'xlsx'])

    xml_formats[instrument] = read_xml_field_formats(instrument, filenames[0])

    pre_delivery_siaf = pysiaf.Siaf(instrument, basepath=pre_delivery_dir)

    # run checks on SIAF content
    ref_siaf = pysiaf.Siaf(instrument)
    compare.compare_siaf(pre_delivery_siaf, reference_siaf_input=ref_siaf,
                         fractional_tolerance=1e-6, report_dir=pre_delivery_dir,
                         tags={'reference': pysiaf.JWST_PRD_VERSION, 'comparison': 'pre_delivery'})

    if instrument.lower() not in ['nirspec']:
        print('\nRunning aperture_transforms test for pre_delivery_siaf')
        test_aperture.test_jwst_aperture_transforms([pre_delivery_siaf],
                                                    verbose=False)
        print('\nRunning aperture_vertices test for pre_delivery_siaf')
        test_aperture.test_jwst_aperture_vertices([pre_delivery_siaf])
    else:
        print('\nRunning regression test of pre_delivery_siaf against IDT test_data:')
        test_nirspec.test_against_test_data(siaf=pre_delivery_siaf)

        print('\nRunning nirspec_aperture_transforms test for pre_delivery_siaf')
        test_nirspec.test_nirspec_aperture_transforms(siaf=pre_delivery_siaf, verbose=False)

        print('\nRunning nirspec_slit_transforms test for pre_delivery_siaf')
        test_nirspec.test_nirspec_slit_transformations(siaf=pre_delivery_siaf, verbose=False)

if show_field_formats:
    show_xml_field_formats(xml_formats)
