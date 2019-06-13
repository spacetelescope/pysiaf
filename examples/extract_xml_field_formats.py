#!/usr/bin/env python
"""Script to extract and display the field formatting of the SIAF xml files.

Authors
-------

    Johannes Sahlmann

"""
import os
from collections import OrderedDict

from astropy.table import Table
import lxml.etree as ET

import pysiaf
from pysiaf.constants import JWST_DELIVERY_DATA_ROOT

show_field_formats = True

siaf_detector_layout = pysiaf.iando.read.read_siaf_detector_layout()


def read_xml_field_formats(instrument, filename, verbose=False):
    """Extract the SIAF xml field formats.

    Parameters
    ----------
    instrument : str
        instrument name
    filename : str
        Absolute path to SIAF.xml
    verbose : bool
        verbosity

    Returns
    -------
    T : astropy table
        Table containing the field formatting

    """
    T = Table()
    print('*' * 100)
    print('{}'.format(instrument))

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
                except TypeError:
                    print('{}: {}'.format(node.tag, node.text))
                    raise TypeError
            elif node.tag in pysiaf.aperture.STRING_ATTRIBUTES:
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
                    except TypeError:
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
    """Print field formats to screen and write to file.

    Parameters
    ----------
    xml_formats : dict
        Dictionary of extracted formats
    reference_instrument_name : str
        Name of instrument to compare against and used as example
    out_dir : str
        Directory to write csv output file

    """
    reference_table = xml_formats[reference_instrument_name]

    for col in ['format', 'pyformat', 'example']:
        reference_table.rename_column(col, '{}_{}'.format(reference_instrument_name, col).lower())
    for key in xml_formats.keys():
        if key != reference_instrument_name:
            for col in ['format', 'pyformat', 'example']:
                reference_table['{}_{}'.format(key, col).lower()] = xml_formats[key][col]

    columns_to_show = ['field_nr', 'field_name']
    for col in reference_table.colnames:
        if ('pyformat' in col) or ('format' in col):
            columns_to_show.append(col)
    columns_to_show.append('{}_example'.format(reference_instrument_name.lower()))

    tag_list = list(reference_table['field_name'])
    rows_to_delete = [i for i in range(len(reference_table)) if ('Sci2IdlX' in tag_list[i] or
                                                                 'Sci2IdlY' in tag_list[i] or
                                                                 'Idl2SciX' in tag_list[i] or
                                                                 'Idl2SciY' in tag_list[i]) and
                      (tag_list[i] not in ['Sci2IdlX00', 'Idl2SciY00'])]

    reference_table.remove_rows(rows_to_delete)

    reference_table[columns_to_show].pprint()

    if out_dir is None:
        out_dir = os.environ['HOME']
    reference_table[columns_to_show].write(os.path.join(out_dir, 'siaf_field_formats.csv'),
                                           format='ascii.basic', delimiter=',', overwrite=True)


if __name__ == '__main__':
    xml_formats = OrderedDict()
    for instrument in ['NIRCam', 'FGS', 'MIRI', 'NIRSpec', 'NIRISS']:
        pre_delivery_dir = os.path.join(JWST_DELIVERY_DATA_ROOT, instrument)
        filename = os.path.join(pre_delivery_dir, '{}_SIAF.xml'.format(instrument))
        xml_formats[instrument] = read_xml_field_formats(instrument, filename)

    if show_field_formats:
        show_xml_field_formats(xml_formats, reference_instrument_name='NIRCam')
