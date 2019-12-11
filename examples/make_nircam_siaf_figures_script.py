"""Script adaptation of make_nircam_siaf_figures.ipynb that also includes saving figures to PDF. """
import os
import matplotlib.pyplot as pl
import pysiaf

show_plot = True
save_plot = True

pl.close('all')
plot_dir = os.environ['HOME']

# Load NIRCam SIAF
instrument = 'NIRCam'
siaf = pysiaf.Siaf(instrument)

####################################################################################################
# Create a plot that shows the SUB160 apertures on Module B.

# Figure setup
fig = pl.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
# Plot the outline of each aperture, with reference points marked (plus symbol is default).
# Plotting blue and red lines separately, blue for short wavelength and red for long wavelength.
for aperture_name in ['NRCB1_SUB160', 'NRCB2_SUB160', 'NRCB3_SUB160', 'NRCB4_SUB160']:
    aperture = siaf[aperture_name]
    aperture.plot(color='b', fill_color='b', fill_alpha=0.3, mark_ref=True)

for aperture_name in ['NRCB5_SUB160']:
    aperture = siaf[aperture_name]
    aperture.plot(color='r', fill_color='r', fill_alpha=0.3, mark_ref=True)
pl.title('Module B, SUB160 Apertures')
if show_plot:
    pl.show()
if save_plot:
    fig_name = os.path.join(plot_dir, '{}_SUB160.pdf'.format(instrument))
    pl.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0)



####################################################################################################
# Make a plot of the full NIRCam FOV with SCA labels
fig = pl.figure(figsize=(12, 6), facecolor='w', edgecolor='k')

# LW full arrays
for aperture_name in ['NRCB5_FULL', 'NRCA5_FULL']:
    aperture = siaf[aperture_name]
    aperture.plot(color='r', fill_color='r', fill_alpha=0.3, lw=2)

# SW full arrays
for aperture_name in ['NRCB1_FULL', 'NRCB2_FULL', 'NRCB3_FULL', 'NRCB4_FULL',
                      'NRCA1_FULL', 'NRCA2_FULL', 'NRCA3_FULL', 'NRCA4_FULL']:
    aperture = siaf[aperture_name]
    aperture.plot(color='b', fill_color='b', fill_alpha=0.3, mark_ref=False, lw=2, name_label=aperture.AperName[3:5])

pl.title('NIRCam Field of View')
if show_plot:
    pl.show()
if save_plot:
    fig_name = os.path.join(plot_dir, '{}_fov.pdf'.format(instrument))
    pl.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0)

####################################################################################################
# Plot the coronagraph subarrays
fig = pl.figure(figsize=(10, 6), facecolor='w', edgecolor='k')

# Start with the LW subarrays (in red)
siaf.plot('tel', ['NRCA5_MASK335R', 'NRCA5_MASK430R', 'NRCA5_MASKLWB_F250W',
                      'NRCA5_MASKLWB_F300M', 'NRCA5_MASKLWB_F277W', 'NRCA5_MASKLWB_F335M',
                      'NRCA5_MASKLWB_F360M', 'NRCA5_MASKLWB_F356W', 'NRCA5_MASKLWB_F410M',
                      'NRCA5_MASKLWB_F430M', 'NRCA5_MASKLWB_F460M', 'NRCA5_MASKLWB_F480M',
                      'NRCA5_MASKLWB_F444W'],
              mark_ref=True, fill_color='None', color='Red', clear=False, lw=1)

# These are teh SW subarrays
siaf.plot('tel', ['NRCA2_MASK210R', 'NRCA4_MASKSWB_F182M', 'NRCA4_MASKSWB_F187N',
                      'NRCA4_MASKSWB_F210M', 'NRCA4_MASKSWB_F212N', 'NRCA4_MASKSWB_F200W'],
              mark_ref=True, fill_color='None', color='Blue', clear=False, lw=1)

# These are the TA apertures for SW (bright sources)
siaf.plot('tel', ['NRCA2_TAMASK210R', 'NRCA4_TAMASKSWB', 'NRCA4_TAMASKSWBS'],
              mark_ref=True, fill_color='None', color='Blue', clear=False, lw=1)

# These are the faint source TA apertures for SW
siaf.plot('tel', ['NRCA2_FSTAMASK210R', 'NRCA4_FSTAMASKSWB'],
              mark_ref=True, fill_color='None', color='Blue', clear=False, lw=1,
              ls='--')

# These are the bright source TA apertures for LW
siaf.plot('tel',
              ['NRCA5_TAMASK335R', 'NRCA5_TAMASK430R', 'NRCA5_TAMASKLWB', 'NRCA5_TAMASKLWBL'],
              mark_ref=True, fill_color='None', color='Red', clear=False, lw=1)

# These are the faint source TA apertures for SW
siaf.plot('tel', ['NRCA5_FSTAMASKLWB', 'NRCA5_FSTAMASK335R', 'NRCA5_FSTAMASK430R'],
              mark_ref=True, fill_color='None', color='Red', clear=False, lw=1,
              ls='--')

pl.title('Module A, Coronagraph Subarrays')

pl.ylim(-425, -390)  # restrict axes

# Fill in the main subarrays
for aperture_name in ['NRCA5_MASKLWB', 'NRCA5_MASK335R', 'NRCA5_MASK430R', 'NRCA4_MASKSWB', 'NRCA2_MASK210R']:
    aperture = siaf[aperture_name]
    if 'NRCA5' in aperture_name:
        color='r'
    else:
        color='b'
    aperture.plot(color=color, fill_color=color, fill_alpha=0.3, mark_ref=False)#, lw=2, name_label=aperture.AperName[3:5])

    # Annotations
    pl.text(aperture.V2Ref, aperture.V3Ref-15, aperture_name[3:], color=color, ha='center')

if show_plot:
    pl.show()
if save_plot:
    fig_name = os.path.join(plot_dir, '{}_coronagraph.pdf'.format(instrument))
    pl.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0)


####################################################################################################
# Plot some Grism subarrays to demonstrate the TA process
fig = pl.figure(figsize=(8, 6), facecolor='w', edgecolor='k')

# Start with LW grism subarray. First line marks the reference point in black, second line outlines the aperture
siaf.plot('tel', ['NRCA5_GRISM64_F444W'],
              mark_ref=True, label=None, color='k', fill_color='None', lw=1)
siaf.plot('tel', ['NRCA5_GRISM64_F444W'],
              mark_ref=False, label=None, color='r', fill_color='None', lw=2, clear=False)

# Add the TA boxes. Mark reference points in black, and outline in color
siaf.plot('tel', ['NRCA5_TAGRISMTS_SCI_F444W'],
              mark_ref=True, label=None, color='k', fill_color='None', lw=1, clear=False)
siaf.plot('tel', ['NRCA5_TAGRISMTS32'],
              mark_ref=True, label=None, color='k', fill_color='None', lw=1, clear=False)

siaf.plot('tel', ['NRCA5_TAGRISMTS_SCI_F444W'],
              mark_ref=False, label=None, color='Purple', fill_color='None', lw=2, clear=False)
siaf.plot('tel', ['NRCA5_TAGRISMTS32'],
              mark_ref=False, label=None, color='Yellow', fill_color='None', lw=2, clear=False)

# Add the associated SW subarrays in dotted lines
siaf.plot('tel', ['NRCA1_GRISMTS64', 'NRCA3_GRISMTS64'],
              mark_ref=False, label=None, color='k', ls=':', fill_color='None', lw=1, clear=False)

# restrict axes range
pl.xlim(100, 70)
pl.ylim(-559, -550)

pl.title('Module A, Grism Target Acquisition')


for aperture_name in ['NRCA5_FULL', 'NRCA5_GRISM64_F444W', 'NRCA5_GRISM64_F444W', 'NRCA5_TAGRISMTS32']:
    aperture = siaf[aperture_name]
    if 'TAGRISM' in aperture_name:
        color='y'
        alpha = 0.5
    else:
        color='r'
        alpha = 0.3
    aperture.plot(color=color, fill_color=color, fill_alpha=alpha, mark_ref=False)#, lw=2, name_label=aperture.AperName[3:5])

# Annotate
ax = pl.gca()
ax.text(94.5, -556.6, 'NRCA5_GRISM64_F444W (target position after grism deflection)', color='k')
ax.text(93.5, -555.0, 'NRCA5_TAGRISMTS_SCI_F444W (TA destination)', color='k')
ax.text(80.0, -553.5, 'NRCA5_TAGRISMTS32 (TA)', color='k')
ax.text(99.5, -558.0, 'SUBGRISM64', color='k')

if show_plot:
    pl.show()
if save_plot:
    fig_name = os.path.join(plot_dir, '{}_grism.pdf'.format(instrument))
    pl.savefig(fig_name, transparent=True, bbox_inches='tight', pad_inches=0)

