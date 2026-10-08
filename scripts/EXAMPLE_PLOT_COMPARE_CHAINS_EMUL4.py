"""Compare samplers on the roman_real w0waCDM cosmic-shear emulator posterior.

Draws one getdist triangle plot of the same posterior (the cosmic-shear
data-vector emulator, w0waCDM) sampled four ways:

  EXAMPLE_EMUL_MCMC4      Metropolis-Hastings (MH), burn-in 30% removed;
  EXAMPLE_EMUL_NAUTILUS2  Nautilus nested sampling;
  EXAMPLE_EMUL_EMCEE2     emcee ensemble sampling;
  EXAMPLE_EMUL_POLY2      PolyChord nested sampling.

The chains are read from projects/roman_real/chains/, where the figure
example_compare_chains_emul4.pdf and hidden copies of the chains
(.VM_P4_TMP*) are written. Run it once the four chains exist:

    python ./projects/roman_real/scripts/EXAMPLE_PLOT_COMPARE_CHAINS_EMUL4.py
"""
import getdist.plots as gplot
from getdist import MCSamples
from getdist import loadMCSamples
import os
import matplotlib
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# Figure style: matplotlib's global settings (rcParams) for fonts, ticks,
# grid and the saved-figure format (PDF, tight bounding box).
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['xtick.top'] = False
matplotlib.rcParams['ytick.right'] = False
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = '1.0'
matplotlib.rcParams['axes.labelsize'] = 'medium'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linewidth'] = '0.0'
matplotlib.rcParams['grid.alpha'] = '0.18'
matplotlib.rcParams['grid.color'] = 'lightgray'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'

parameter = [u'wa',u'w',u'As_1e9', u'omegam', u'ns', u'H0', u'omegab', u'chi2v2']
# ROOTDIR, the cocoa/Cocoa folder, is set by start_cocoa.sh.
chaindir  = os.environ['ROOTDIR'] + "/projects/roman_real/chains/"

# getdist analysis settings: Gaussian smoothing of the 1D and 2D densities
# by 0.25 standard deviations, plot ranges that hold 99.5% of each 1D
# posterior (range_confidence = 0.005), and fine histogram bins.
# analysissettings drops the first 30% of each MH chain
# as burn-in (ignore_rows = 0.3) when the chain is loaded; analysissettings2
# keeps every row, for the saved copies (already cut) and for samplers whose
# output needs no burn-in cut.
analysissettings={'smooth_scale_1D':0.25,
                  'smooth_scale_2D':0.25,
                  'ignore_rows': u'0.3',
                  'range_confidence' : u'0.005',
                  'fine_bins_2D': 1024,
                  'fine_bins_1D': 256}

analysissettings2={'smooth_scale_1D':0.25,
                   'smooth_scale_2D':0.25,
                   'ignore_rows': u'0.0',
                   'range_confidence' : u'0.005',
                   'fine_bins_2D': 1024,
                   'fine_bins_1D': 256}

root_chains = (
  'EXAMPLE_EMUL_MCMC4',
  'EXAMPLE_EMUL_NAUTILUS2',
  'EXAMPLE_EMUL_EMCEE2',
  'EXAMPLE_EMUL_POLY2',
)

# --------------------------------------------------------------------------------
# Each block loads one chain, adds the derived parameter chi2v2 and saves a
# hidden text copy (a name starting with '.') in chaindir, which the
# triangle plot reads. chi2v2 = chi2 + 2 minuslogprior is -2 ln(posterior)
# up to a constant (chi2 = -2 ln L, minuslogprior = -ln prior).
# For the Nautilus and emcee chains chi2v2 is chi2 alone, and wa is
# derived from the sampled w0pwa = w0 + wa as w0pwa - w.
samples=loadMCSamples(chaindir + root_chains[0],settings=analysissettings)
p = samples.getParams()
samples.addDerived(p.chi2+2*p.minuslogprior,name='chi2v2',label='{\\chi^2}')
samples.saveAsText(chaindir + '/.VM_P4_TMP1')
# --------------------------------------------------------------------------------
samples=loadMCSamples(chaindir+ root_chains[1], settings=analysissettings2)
p = samples.getParams()
samples.addDerived(p.chi2,name='chi2v2',label='{\\chi^2}')
samples.addDerived(p.w0pwa-p.w,name='wa',label='{w_a}')
samples.saveAsText(chaindir + '/.VM_P4_TMP2')
# --------------------------------------------------------------------------------
samples=loadMCSamples(chaindir+ root_chains[2],settings=analysissettings2)
p = samples.getParams()
samples.addDerived(p.chi2,name='chi2v2',label='{\\chi^2}')
samples.addDerived(p.w0pwa-p.w,name='wa',label='{w_a}')
samples.saveAsText(chaindir + '/.VM_P4_TMP3')
# --------------------------------------------------------------------------------
samples=loadMCSamples(chaindir+ root_chains[3],settings=analysissettings2)
p = samples.getParams()
samples.addDerived(p.chi2+2*p.minuslogprior,name='chi2v2',label='{\\chi^2}')
samples.saveAsText(chaindir + '/.VM_P4_TMP4')
# --------------------------------------------------------------------------------

# getdist triangle plot: the 1D posterior of each parameter on the
# diagonal and the 2D contours of every pair below it; the settings fix
# the figure width (inches), tick rotation, line widths and font sizes.
g=gplot.getSubplotPlotter(chain_dir=chaindir,
                          analysis_settings=analysissettings2,
                          width_inch=10.5)
g.settings.axis_tick_x_rotation=65
g.settings.lw_contour=1.0
g.settings.legend_rect_border = False
g.settings.figure_legend_frame = False
g.settings.axes_fontsize = 15.0
g.settings.legend_fontsize = 15.5
g.settings.alpha_filled_add = 0.85
g.settings.lab_fontsize=15.5
g.legend_labels=False

g.triangle_plot(
  params=parameter,
  roots=[chaindir + '/.VM_P4_TMP1',
         chaindir + '/.VM_P4_TMP2',
         chaindir + '/.VM_P4_TMP3',
         chaindir + '/.VM_P4_TMP4'],
  plot_3d_with_param=None,
  line_args=[{'lw': 1.0,'ls': 'solid', 'color':'lightcoral'},
              {'lw': 1.2,'ls': '--', 'color':'black'},
              {'lw': 2.1,'ls': 'dotted', 'color': 'maroon'},
              {'lw': 1.6,'ls': '-.', 'color': 'indigo'}
            ],
  contour_colors=['lightcoral','black','maroon', 'indigo'],
  contour_ls=['solid','--','dotted','-.'],
  contour_lws=[1.0,1.2,2.1,1.6],
  filled=[True,False,False,True],
  shaded=False,
  legend_labels=[
    'MH, 4-walkers, burn-in=0.3 (HMCODE)',
    'Nautilus, $n_{\\rm live}=1024$, $n_{\\rm eff} \\sim 15,000$',
    'EMCEE $n_{\\rm walkers}=3 \\times ndim$, $n_{\\rm eval} \\sim 2,000,000$',
    'PolyChord $n_{\\rm live}=512$, $n_{\\rm repeat}=3D$',
  ],
  legend_loc=(0.3, 0.85))
# Save the figure as a PDF in chaindir.
g.export(os.path.join(chaindir,"example_compare_chains_emul4.pdf"))