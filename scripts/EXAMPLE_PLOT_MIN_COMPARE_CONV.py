#!/usr/bin/env python3
"""Plot how the minimum found by the annealed emcee minimizer converges with n_stw.

n_stw is the number of steps per walker per temperature of the minimizer of
EXAMPLE_EMUL_MINIMIZE1.py (LCDM) and EXAMPLE_EMUL_MINIMIZE2.py (w0waCDM),
both on Roman real-space cosmic shear. The script reads 15 runs of each,
saved with --outroot EXAMPLE_EMUL_MIN<n>_test_<i> and --nstw 100 + 25 i
(i = 0..14); the last column of each one-row file is the chi2 of the
minimum. The figure shows |chi2_min(n_stw) - chi2_min(450)| for i = 1..13,
the distance to the run with the most steps, on a log axis.

Reads and writes in projects/roman_real/chains/ (ROOTDIR must be set);
the figure is example_compare_min_conv.pdf.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

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


plt.figure(figsize=(8, 5))
colors = ['royalblue','lightcoral','black', 'purple']
markers = ['o', 's', '^', 'v', 'D', '*', 'x', 'P', '<', '>']
linestyles = ['solid',
              '-', 
              '--', 
              '-.', 
              ':', 
              (0,(3,1,1,1)), 
              (0,(5,2)), 
              (0,(1,1)), 
              (0,(3,5,1,5)), 
              (0,(1,10)), 
              (0,(5,1))]
# sz runs per model; run i used n_stw = 100 + 25 i. The list comprehension
# builds one row [n_stw, chi2_min] per run: np.loadtxt reads the one-row
# file and [-1] takes its last column. Rows 1..13 are compared with the
# last row, the run with the largest n_stw.
sz=15
rt = os.environ['ROOTDIR']+"/projects/roman_real/chains/EXAMPLE_EMUL_MIN1_test_"
data = np.array([[100+25*i,np.loadtxt(f"{rt}{i}.txt")[-1]] for i in range(sz)])
plt.plot(data[1:sz-1,0], 
         abs(data[1:sz-1,1]-data[-1,1]), 
         marker=markers[5],
         linestyle=linestyles[5],
         color=colors[0], 
         label="$\\Lambda$CDM, Roman Real Cosmic Shear")
rt = os.environ['ROOTDIR']+"/projects/roman_real/chains/EXAMPLE_EMUL_MIN2_test_"
data = np.array([[100+25*i,np.loadtxt(f"{rt}{i}.txt")[-1]] for i in range(sz)])
plt.plot(data[1:sz-1,0], 
         abs(data[1:sz-1,1]-data[-1,1]), 
         marker=markers[5],
         linestyle=linestyles[5],
         color=colors[1], 
         label="w0waCDM, Roman Real Cosmic Shear")

# Axes styling: major and faint dashed minor grid lines, tick label sizes,
# and a log y axis from 1e-5 to 10.
ax = plt.gca()
ax.grid(True)
ax.grid(True, 
        which='minor', 
        color='grey', 
        linestyle='--', 
        linewidth=0.25, 
        alpha=0.1)
ax.minorticks_on()
ax.tick_params(axis='both', which='major',labelsize=15)
ax.tick_params(axis='both', which='minor',labelsize=15)
plt.yscale('log')
plt.ylim(1e-5, 10)
# Axis labels (matplotlib renders the $...$ parts as LaTeX math)
plt.xlabel("$n_{\\rm STW}$")
plt.ylabel("$\\Delta \\chi_{\\rm min}^2$")
ax.legend(fontsize=13, frameon=False)
plt.savefig(os.environ['ROOTDIR']+
            "/projects/roman_real/chains/example_compare_min_conv.pdf", 
            bbox_inches='tight')