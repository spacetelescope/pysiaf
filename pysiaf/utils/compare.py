"""Functions to support comparisons between SIAF files.

Authors
-------
    Johannes Sahlmann

References
----------
    The format of the difference files was adapted from Colin Cox' matchcsv.py
    dict_compare was adapted from
    https://stackoverflow.com/questions/4527942/comparing-two-dictionaries-in-python/4527957

"""
from __future__ import absolute_import, print_function, division
from collections import OrderedDict
import os
import sys


from astropy.table import vstack, Table
import numpy as np
import pylab as pl

from ..constants import JWST_PRD_VERSION
from ..iando.read import get_siaf, read_siaf_aperture_definitions
from ..siaf import Siaf
from ..utils import tools
from ..aperture import compare_apertures


def compare_siaf(comparison_siaf_input, fractional_tolerance=1e-4, reference_siaf_input=None,
                 report_file=None, report_dir=None, verbose=True, make_figures=False,
                 selected_aperture_name=None, ignore_attributes=None, tags=None):
    """Compare two SIAF files and write a difference file.

    Generate comparison figures showing the apertures if specified.

    Parameters
    ----------
    comparison_siaf_input : str (absolute file name) or pysiaf.Siaf object
        The SIAF that will be compared to the reference_siaf
    fractional_tolerance : float
        Value above which a fractional difference in parameter value will be written in the report
    reference_siaf_input : str (absolute file name) or pysiaf.Siaf object
        The reference SIAF. Defaults to the current PRD content.
    report_file :
    report_dir :
    verbose
    make_figures
    selected_aperture_name

    """
    if verbose:
        print(comparison_siaf_input)
    comparison_siaf = get_siaf(comparison_siaf_input)
    instrument = comparison_siaf.instrument
    if verbose:
        print(instrument)

    if reference_siaf_input is None:
        reference_siaf = Siaf(instrument)
        reference_siaf_description = '{}-{}'.format(instrument, JWST_PRD_VERSION)
    else:
        reference_siaf = get_siaf(reference_siaf_input)
        reference_siaf_description = reference_siaf.description.replace('.', '_')

    if report_file is None:
        print_file = sys.stdout
    else:
        print_file = open(report_file, 'w')

    reference_tag = reference_siaf_description
    comparison_tag = comparison_siaf.description.replace('.', '_')
    if tags is not None:
        reference_tag = '{}'.format(tags['reference'])
        comparison_tag = '{}'.format(tags['comparison'])

    if report_dir is not None:
        report_file = os.path.join(report_dir, '{}_Diff_{}_{}.txt'.format(
            instrument, reference_tag, comparison_tag))
        print_file = open(report_file, 'w')

    if verbose:
        print('Reference:  {} apertures in {}'.format(
            len(reference_siaf), reference_tag), file=print_file)
        print('Comparison: {} apertures in {}\n'.format(
            len(comparison_siaf), comparison_tag), file=print_file)

    added_aperture_names, removed_aperture_names, modified_apertures, same_apertures = \
        dict_compare(comparison_siaf.apertures, reference_siaf.apertures)

    show_added = False
    show_removed = False

    if len(added_aperture_names) != 0:
        show_added = True
    if len(removed_aperture_names) != 0:
        show_removed = True

    if show_added:
        attributes_to_show = 'AperName XDetRef YDetRef XSciSize YSciSize XSciRef YSciRef V2Ref ' \
                             'V3Ref'.split()

        print('Number of added apertures {}:'.format(len(added_aperture_names)), file=print_file)
        print('\t{0:5s} {1}'.format('', ' '.join(['{:>12s}'.format(a) for a in
                                                  attributes_to_show])), file=print_file)
        for aperture_name in added_aperture_names:
            print('\tAdded {}'.format(' '.join(['{:12}'.format(
                getattr(comparison_siaf[aperture_name], a)) for a in attributes_to_show])),
                  file=print_file)
        print()

    if show_removed:
        print('Number of removed apertures {}:'.format(len(removed_aperture_names)),
              file=print_file)
        for aperture_name in removed_aperture_names:
            print('\tRemoved {}'.format(aperture_name), file=print_file)
        print()

    if selected_aperture_name is not None:
        if type(selected_aperture_name) == str:
            selected_aperture_name = [selected_aperture_name]

    print('Number of modified apertures {}. Significant modifications are listed below.'.format(
        len(modified_apertures)), file=print_file)
    print('Differences are reported for any text change', file=print_file)
    print('and fractional differences greater than {}\n'.format(fractional_tolerance),
          file=print_file)

    if selected_aperture_name is not None:
        print('Only the following selected apertures are shown: {}.\n'.format(
            selected_aperture_name), file=print_file)

    print('{:25} {:>15} {:>21} {:>21} {:>15} {:>10}'.format('Aperture', 'Attribute', 'Reference',
                                                            'Comparison', 'Difference', 'Percent'),
          file=print_file)
    report_table = None

    # sort SIAF entries in the order of the aperture definition file
    siaf_aperture_definitions = read_siaf_aperture_definitions(instrument)
    aperture_name_list = siaf_aperture_definitions['AperName'].tolist()
    modified_apertures = OrderedDict(
        sorted(modified_apertures.items(), key=lambda t: aperture_name_list.index(t[0])))

    for aperture_name in modified_apertures.keys():
        if (selected_aperture_name is not None) and (aperture_name not in selected_aperture_name):
            continue
        comparison_table = compare_apertures(reference_siaf[aperture_name],
                                             comparison_siaf[aperture_name],
                                             fractional_tolerance=fractional_tolerance,
                                             print_file=print_file, verbose=False,
                                             ignore_attributes=ignore_attributes)
        if report_table is None:
            report_table = comparison_table.copy()
        else:
            report_table = vstack((report_table, comparison_table))

    if report_file is not None:
        print_file.close()
        if verbose:
            print('Wrote difference file to {}'.format(report_file))

    if make_figures:
        for j, aperture_name in enumerate(report_table['aperture'].data):
            pl.close('all')
            aperture_name = aperture_name.decode()
            reference_aperture = reference_siaf[aperture_name]
            if ('FULL' in aperture_name) and ('MASK' not in aperture_name) and \
                    (reference_aperture.AperType in ['FULLSCA']) and \
                    (report_table['attribute'][j] in ['V2Ref', 'V3Ref']) and \
                    (report_table['difference'][j] != 'N/A') and \
                    (float(report_table['difference'][j]) > 1.0):
                print('Plotting {}'.format(reference_aperture.AperName))
                comparison_aperture = comparison_siaf[aperture_name]
                pl.figure(figsize=(7, 7), facecolor='w', edgecolor='k')
                pl.clf()
                reference_aperture.plot(fill=False, line_style='--', line_label='Reference')
                comparison_aperture.plot(line_label='Comparison')
                pl.legend(loc=1)
                pl.title(aperture_name)
                # pl.show()
                if report_dir is not None:
                    figure_name = os.path.join(report_dir, '{}_{}_Diff_{}_{}.pdf'.format(
                        instrument, aperture_name, reference_siaf_description,
                        comparison_siaf.description.replace('.', '_')))
                    pl.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0,
                               dpi=300)


def compare_transformation_roundtrip(comparison_siaf_input, fractional_tolerance=1e-4,
                                     reference_siaf_input=None, report_file=None, report_dir=None,
                                     verbose=True, make_figures=False, selected_aperture_name=None,
                                     instrument=None, make_plot=False, tags=None):
    """Compare the forward-backward roundtrip transformations of two SIAF files.

    and write a difference file.

    Parameters
    ----------
    comparison_siaf_input : str (absolute file name) or pysiaf.Siaf object
        The SIAF that will be compared to the reference_siaf
    fractional_tolerance : float
        Value above which a fractional difference in parameter value will be written in
        the report
    reference_siaf_input : str (absolute file name) or pysiaf.Siaf object
        The reference SIAF. Defaults to the current PRD content.
    report_file
    report_dir
    verbose
    make_figures
    selected_aperture_name
    instrument
    make_plot

    Returns
    -------
    roundtrip_table : astropy.table.Table object
        table containing roundtrip data

    """
    if verbose:
        print(comparison_siaf_input)
    comparison_siaf = get_siaf(comparison_siaf_input)
    instrument = comparison_siaf.instrument
    if verbose:
        print(instrument)

    if reference_siaf_input is None:
        reference_siaf = Siaf(instrument)
        reference_siaf_description = '{}-{}'.format(instrument, JWST_PRD_VERSION)
    else:
        reference_siaf = get_siaf(reference_siaf_input)
        reference_siaf_description = reference_siaf.description.replace('.', '_')

    reference_tag = reference_siaf_description
    comparison_tag = comparison_siaf.description.replace('.', '_')
    if tags is not None:
        reference_tag = '{}'.format(tags['reference'])
        comparison_tag = '{}'.format(tags['comparison'])



    if report_file is None:
        print_file = sys.stdout
    else:
        print_file = open(report_file, 'w')

    if report_dir is not None:
        report_file = os.path.join(report_dir, '{}_roundtrip_{}_{}.txt'.
                                   format(instrument, reference_tag, comparison_tag))

        print_file = open(report_file, 'w')

    if verbose:
        print('Reference:  {} apertures in {} (index 0 in table below)'.format(len(reference_siaf),
                                                      reference_tag),
              file=print_file)
        print('Comparison: {} apertures in {} (index 1 in table below)\n'.format(len(comparison_siaf),
                                                        comparison_tag),
              file=print_file)

    siaf_list = [reference_siaf, comparison_siaf]

    roundtrip_dict = OrderedDict()
    round_trip_tags = 'metric dx_mean dy_mean dx_rms dy_rms'.split()

    for key in 'AperName'.split() + ['siaf{}_{}'.format(j, tag) for j in range(len(siaf_list)) for
                                     tag in round_trip_tags]:
        roundtrip_dict[key] = []

    # index 0 is for reference SIAF (defaults to PRD)
    # index 1 is for comparison SIAF
    for AperName, aperture in reference_siaf.apertures.items():
        # pl.close('all')
        if (selected_aperture_name is not None) and (AperName not in list(selected_aperture_name)):
            continue
        for j, siaf in enumerate(siaf_list):
            aperture = siaf[AperName]
            coefficients = aperture.get_polynomial_coefficients()
            if (coefficients is not None) and (coefficients['Sci2IdlX'][0] is not None):
                if AperName not in roundtrip_dict['AperName']:
                    roundtrip_dict['AperName'].append(AperName)
                roundtrip_errors = tools.compute_roundtrip_error(coefficients['Sci2IdlX'],
                                                                 coefficients['Sci2IdlY'],
                                                                 coefficients['Idl2SciX'],
                                                                 coefficients['Idl2SciY'],
                                                                 offset_x=aperture.XSciRef,
                                                                 offset_y=aperture.YSciRef,
                                                                 instrument=instrument)
                for k, tag in enumerate(round_trip_tags):
                    # if tag != 'data':
                    roundtrip_dict['siaf{}_{}'.format(j, tag)].append(roundtrip_errors[k])

                if make_plot:
                    data = roundtrip_errors[-1]
                    pl.figure(19, figsize=(6, 6), facecolor='w', edgecolor='k')
                    pl.quiver(data['x'], data['y'], data['x']-data['x2'], data['y']-data['y2'],
                              angles='xy')
                    pl.xlabel('x_sci')
                    pl.ylabel('y_sci')
                    ax = pl.gca()
                    pl.text(0.5, 0.9, 'Maximum arrow length {:3.3f} pix'.format(
                        np.max(np.linalg.norm([data['x']-data['x2'], data['y']-data['y2']],
                                              axis=0))), horizontalalignment='center',
                            transform=ax.transAxes)
                    pl.title('siaf{}: {} Roundtrip error sci->idl->sci'.format(j, AperName))
                    # pl.show()
                    if report_dir is not None:
                        figure_name = os.path.join(report_dir, '{}_{}_siaf{}_roundtrip_error.pdf'.
                                                   format(instrument, AperName, j))
                        pl.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)

    roundtrip_table = Table(roundtrip_dict)
    for k, tag in enumerate(round_trip_tags):
        roundtrip_table['difference_{}'.format(tag)] = roundtrip_table['siaf{}_{}'.format(1, tag)] \
                                                       - roundtrip_table['siaf{}_{}'.format(0, tag)]
    absolute_differences = np.array([np.abs(roundtrip_table['difference_{}'.format(tag)]) for
                                     tag in round_trip_tags[1:]])
    bad_index = np.where(np.any(absolute_differences > 1e-9, axis=0))[0]

    fstring = '{' + ','.join(["'%s':  '%%1.6f'" % (c) for c in roundtrip_table.colnames if
                              '_mean' in c]) + '}'
    roundtrip_table['AperName siaf0_dx_mean siaf0_dy_mean siaf1_dx_mean siaf1_dy_mean '
                    'difference_dx_mean difference_dy_mean'.split()].\
        write(print_file, format='ascii.fixed_width', delimiter=',', delimiter_pad=' ', 
              bookend=False, overwrite=True, formats=eval(fstring))

    print('Apertures with significant roundtrip error differences:')
    roundtrip_table[bad_index].pprint()
    # roundtrip_table.write()

    if report_dir is not None:
        # pl.close('all')

        fig = pl.figure(figsize=(30, 6), facecolor='w', edgecolor='k')
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

        pl.ylabel('(pixel)')
        pl.xticks(np.arange(len(roundtrip_table)), roundtrip_table['AperName'], rotation='vertical',
                  fontsize=8)
        # pl.margins(0.2)
        pl.subplots_adjust(bottom=0.15)
        pl.show()

        figure_name = os.path.join(report_dir, '{}_comparison_roundtrip_error.pdf'.format(
            instrument))
        pl.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)

    return roundtrip_table


def dict_compare(dictionary_1, dictionary_2):
    """Compare two dictionaries and return keys of the differing items.

    From https://stackoverflow.com/questions/4527942/comparing-two-dictionaries-in-python/4527957

    Parameters
    ----------
    dictionary_1 : dict
        first dictionary
    dictionary_2 : dict
        second dictionary

    Returns
    -------
    added, removed, modified, same : set
        Sets of dictionary keys that were added, removed, modified, or are the same

    """
    d1_keys = set(dictionary_1.keys())
    d2_keys = set(dictionary_2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o: (dictionary_1[o], dictionary_2[o]) for o in intersect_keys
                if dictionary_1[o] != dictionary_2[o]}
    same = set(o for o in intersect_keys if dictionary_1[o] == dictionary_2[o])
    return added, removed, modified, same
