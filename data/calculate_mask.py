"""Write the scale-cut mask example1.mask of the real-space 3x2pt data vector.

The mask has one flag per data-vector entry, 1 = keep and 0 = cut, in the
data-vector order [xi_+ | xi_- | gamma_t | w]:

  xi_+, xi_-  one row of N_ANG_BINS flags per source-bin pair (i <= j),
              kept above the angles ξp_CUTOFF and ξm_CUTOFF in arcmin
              (0 keeps every bin);
  gamma_t     one row per (lens, source) pair, lens-major, without the
              pairs in ggl_skip_combos;
  w           one row per lens bin (auto-correlations only).

gamma_t and w are cut below the angle that the length gc_CUTOFF subtends at
the mean redshift of the lens bin (ang_cut), so small, nonlinear scales of
the galaxy probes are removed.

Run it from this folder, where it reads example1.nz and overwrites
example1.mask (two columns: entry index, flag):

    python calculate_mask.py
"""
import numpy as np
import os
from astropy.cosmology import FlatLambdaCDM
import math as mt

redshift_file = 'example1.nz' #redshift file - change to your own file
ggl_skip_combos = [[6,0],[7,0],[7,1]] #skip these combinations of lens and source bins

ξp_CUTOFF = 0  # cutoff scale in arcminutes
ξm_CUTOFF = 0 # cutoff scale in arcminutes
gc_CUTOFF = 1.5 # Galaxy clustering cutoff in Mpc/h

THETA_MIN  = 2.5    # Minimum angular scale (in arcminutes)
THETA_MAX  = 250.  # Maximum angular scale (in arcminutes)
N_ANG_BINS = 15    # Number of angular bins
N_LENS = 8  # Number of lens tomographic bins
N_SRC  = 8  # Number of source tomographic bins
# source-bin pairs (i <= j) of cosmic shear, and the length of the xi_+
# (or xi_-) block
N_XI_PS = int(N_SRC * (N_SRC + 1) / 2) 
N_XI    = int(N_XI_PS * N_ANG_BINS)

def calculate_average_redshift(filename):
    """
    Return the mean redshift of each of the 8 bins of an n(z) file.

    The file has the redshift in column 0 and the n(z) of bin i in column i
    (i = 1..8, any normalization). The mean of bin i is
    sum(z n_i(z)) / sum(n_i(z)) over the tabulated rows.

    Parameters:
        filename (str): Path to the n(z) text file.

    Returns:
        list of 8 floats: the mean redshift of each bin.
    """
    data = np.loadtxt(filename)
    avg_redshifts = []

    redshifts = data[:, 0]  # First column: redshift bins
    for i in np.arange(1,9):
        # column i: the n(z) of bin i
        counts = data[:, i]      # Second column: number of data points

        avg_redshifts.append(np.sum(redshifts * counts) / np.sum(counts))

    return avg_redshifts

# mean redshift of each lens bin (the lens and source n(z) share the file)
zavg = calculate_average_redshift(redshift_file)
    
# Angular bins: N_ANG_BINS log-spaced bins between THETA_MIN and THETA_MAX.
# 2.90888208665721580e-4 = pi/(180*60) converts arcmin to radians. theta[i]
# is the area-weighted center of bin i, 2/3 (tmax^3 - tmin^3)/(tmax^2 -
# tmin^2), back in arcmin; the extra last entry stays 0, so the masks below
# read theta[:-1].
vtmin = THETA_MIN * 2.90888208665721580e-4;
vtmax = THETA_MAX * 2.90888208665721580e-4;
logdt = (mt.log(vtmax) - mt.log(vtmin))/N_ANG_BINS;
theta = np.zeros(N_ANG_BINS+1)

for i in range(N_ANG_BINS):
  tmin = mt.exp(mt.log(vtmin) + (i + 0.0) * logdt);
  tmax = mt.exp(mt.log(vtmin) + (i + 1.0) * logdt);
  x = 2./ 3.
  theta[i] = x * (tmax**3 - tmin**3) / (tmax**2- tmin**2)
  theta[i] = theta[i]/2.90888208665721580e-4

# H0 = 100 km/s/Mpc makes astropy's distances come out in Mpc/h, the unit of
# gc_CUTOFF; Omega_m = 0.3 is the fiducial.
cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
def ang_cut(z):
  """Return the angle, in arcmin, that gc_CUTOFF subtends at redshift z.

  theta = gc_CUTOFF / D_A(z), with D_A the angular-diameter distance in
  Mpc/h: gc_CUTOFF is treated as a proper (physical) transverse length.

  Arguments:
    z = redshift (the mean redshift of a lens bin)

  Returns:
    float, the cut angle in arcmin
  """
  theta_rad = gc_CUTOFF / cosmo.angular_diameter_distance(z).value
  return theta_rad * 180. / np.pi * 60.


# Cosmic shear -------------------------------------------------------------
# The comprehension repeats the boolean row of the N_ANG_BINS angles once per
# source pair; np.hstack joins the rows into one flat block of length N_XI.
ξp_mask = np.hstack([(theta[:-1] > ξp_CUTOFF) for i in range(N_XI_PS)])
ξm_mask = np.hstack([(theta[:-1] > ξm_CUTOFF) for i in range(N_XI_PS)])   

# Galaxy-galaxy lensing ----------------------------------------------------
# One row per (lens j, source k) pair, lens-major, skipping ggl_skip_combos
# (zero-based [lens, source] pairs); the cut uses the lens bin's mean z.
γt_mask = []  #initialize empty list for γt_mask
for j in range(N_LENS): 
    for k in range(N_SRC):
        if [j,k] in ggl_skip_combos:
            continue
        else:
            γt_mask.append((theta[:-1] > ang_cut(zavg[j])))
γt_mask = np.hstack(γt_mask) 

# Clustering w(theta) ------------------------------------------------------
# One row per lens bin, with the same cut as gamma_t. The full mask follows
# the data-vector order [xi_+ | xi_- | gamma_t | w].
w_mask = np.hstack([(theta[:-1] > ang_cut(zavg[j])) for j in range(N_LENS)])
mask = np.hstack([ξp_mask, ξm_mask, γt_mask, w_mask])

# Output -------------------------------------------------------------------
# Two columns, the entry index and the flag (written as 0.0 or 1.0).
np.savetxt("example1.mask",
 np.column_stack((np.arange(0,len(mask)),
 mask.astype(int))),fmt='%d %1.1f')

