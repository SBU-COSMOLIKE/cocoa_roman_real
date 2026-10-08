"""Profile one parameter of Roman real-space cosmic shear with annealed emcee.

The likelihood is roman_real.cosmic_shear evaluated by the neural-network
emulator of the whole cosmic-shear data vector (use_emulator: 1, NLA
intrinsic alignments).
Model: LCDM (no w or w0pwa among the emulator inputs).
Nonlinear power of the emulator's training set: HMcode2020.
The yaml's prior block adds Gaussian priors on the cosmological
parameters and the NLA amplitudes, which keep the search inside the
emulator's training region.

For each value of the profiled parameter on a grid around the minimum,
the other parameters are minimized (min_chi2 with that parameter held);
the curve of minimized chi2 = -2 ln(posterior) against the held value is
the profile. The minimum comes from --minfile (the output row of
EXAMPLE_EMUL_MINIMIZE3.py) or is computed first. The grid has the
minimum at its center and the half-width of [x0 - factor sigma,
x0 + factor sigma] clipped to the prior bounds (sigma from --cov, a chain
covariance, or from the prior, which caps factor at 1); --numpts sets its
size (an odd count: 11 for --numpts 10 or 11).

Output: <root>chains/<outroot>.<param>.txt, one row per grid point:
[held value, chi2, every sampled parameter, the chi2 of each likelihood,
the prior chi2].

Run from cocoa/Cocoa with MPI, for example (one command, wrapped here):

    mpirun -n 12 --oversubscribe python ./projects/roman_real/EXAMPLE_EMUL_PROFILE3.py
        --root ./projects/roman_real/ --cov chains/EXAMPLE_EMUL_MCMC3.covmat
        --outroot EXAMPLE_EMUL_PROFILE3 --factor 3 --nstw 450 --numpts 10
        --profile 1 --minfile ./projects/roman_real/chains/EXAMPLE_EMUL_MIN3.txt

--profile is the zero-based index of the parameter in the sampled order.
"""
import warnings
import os
# The filters below silence warnings that would repeat at every
# evaluation of a long run: scikit-learn's InconsistentVersionWarning
# (a stored model made with another scikit-learn version), sacc's column
# deprecation notice, numpy's invalid-value and overflow RuntimeWarnings,
# and the UserWarnings matched by their message text.
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings(
    "ignore",
    message=".*column is deprecated.*",
    module=r"sacc\.sacc"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*invalid value encountered*"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*overflow encountered*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Function not smooth or differentiabl*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Hartlap correction*"
)
import functools, iminuit, copy, argparse, random, time 
import emcee, itertools
import numpy as np
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from getdist import IniFile
from schwimmbad import MPIPool
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(prog='EXAMPLE_EMUL_PROFILE1')
parser.add_argument("--nstw",
                    dest="nstw",
                    help="Number of likelihood evaluations (steps) per temperature per walker",
                    type=int,
                    nargs='?',
                    const=1,
                    default=200)
parser.add_argument("--root",
                    dest="root",
                    help="Name of the Output File",
                    nargs='?',
                    const=1,
                    default="./projects/example/")
parser.add_argument("--outroot",
                    dest="outroot",
                    help="Name of the Output File",
                    nargs='?',
                    const=1,
                    default="test.dat")
parser.add_argument("--profile",
                    dest="profile",
                    help="Which Parameter to Profile",
                    type=int,
                    nargs='?',
                    const=1,
                    default=1)
parser.add_argument("--factor",
                    dest="factor",
                    help="Factor that set the bounds (multiple of cov matrix)",
                    type=float,
                    nargs='?',
                    const=1.0,
                    default=3.0)
parser.add_argument("--numpts",
                    dest="numpts",
                    help="Number of Points to Compute Minimum",
                    type=int,
                    nargs='?',
                    const=1,
                    default=20)
parser.add_argument("--minfile",
                    dest="minfile",
                    help="Minimization Result",
                    nargs='?',
                    const=1)
parser.add_argument("--cov",
                    dest="cov",
                    help="Chain Covariance Matrix",
                    nargs='?',
                    const=1,
                    default=None)
# parse_known_args returns the options this parser defines and leaves
# any other command-line argument in unknown instead of stopping with an
# error (MPI launchers and mpi4py.futures may pass their own).
args, unknown = parser.parse_known_args()
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# The cobaya configuration as yaml text (a raw string, so the LaTeX
# backslashes of the labels stay as written): the roman_real
# cosmic-shear likelihood evaluated by the data-vector emulator
# (use_emulator: 1), the priors of every sampled parameter, and the
# emulator theory block (ord: its input parameters, in order;
# fast_params: the shear calibrations, which cosmolike applies to the
# emulated vector).
yaml_string=r"""
likelihood:
  roman_real.cosmic_shear:
    use_emulator: 1
    path: ./external_modules/data/roman_real
    data_file: example1.dataset # that assumes lens = source
prior:
  # These priors are meant to prevent the sampler to wander far off training
  g1: "lambda As_1e9: stats.norm.logpdf(As_1e9, loc=2.35, scale=1.6)"
  g2: "lambda ns: stats.norm.logpdf(ns, loc=0.96, scale=0.05)"
  g3: "lambda H0: stats.norm.logpdf(H0, loc=70, scale=10.0)"
  g4: "lambda omegab: stats.norm.logpdf(omegab, loc=0.045, scale=0.012)"
  g5: "lambda omegam: stats.norm.logpdf(omegam, loc=0.3 , scale=0.25)"
  g8: "lambda roman_A1_1: stats.norm.logpdf(roman_A1_1, loc=0, scale=2.5)"
  g9: "lambda roman_A1_2: stats.norm.logpdf(roman_A1_2, loc=-1.7, scale=2.5)" 
params:
  As_1e9:
    prior:
      min: 0.5
      max: 5
    ref:
      dist: norm
      loc: 2.1
      scale: 0.65
    proposal: 0.4
    latex: 10^9 A_\mathrm{s}
    renames: A
  ns:
    prior:
      min: 0.87
      max: 1.07
    ref:
      dist: norm
      loc: 0.96605
      scale: 0.01
    proposal: 0.01
    latex: n_\mathrm{s}
  H0:
    prior:
      min: 55
      max: 91
    ref:
      dist: norm
      loc: 67.32
      scale: 5
    proposal: 3
    latex: H_0
  omegab:
    prior:
      min: 0.03
      max: 0.07
    ref:
      dist: norm
      loc: 0.0495
      scale: 0.004
    proposal: 0.004
    latex: \Omega_\mathrm{b}
  omegam:
    prior:
      min: 0.1
      max: 0.9
    ref:
      dist: norm
      loc: 0.316
      scale: 0.02
    proposal: 0.02
    latex: \Omega_\mathrm{m}
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # Nuisance parameters below (it overwrites the default settings)
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  roman_DZ_S1:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^1
  roman_DZ_S2:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^2
  roman_DZ_S3:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^3
  roman_DZ_S4:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^4
  roman_DZ_S5:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^5
  roman_DZ_S6:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^6
  roman_DZ_S7:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^7
  roman_DZ_S8:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^8
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # Intrinsic alignment
  roman_A1_1:
    prior:
      min: -5
      max:  5
    ref:
      dist: norm
      loc: 0.7
      scale: 0.5
    proposal: 0.5
    latex: A_\mathrm{1-IA,roman}^1
  roman_A1_2:
    prior:
      min: -5
      max:  5
    ref:
      dist: norm
      loc: -1.7
      scale: 0.5
    proposal: 0.5
  roman_A2_1:
    value: 0.0
    latex: A_\mathrm{2-IA,Roman}^1
  roman_A2_2:
    value: 0.0
    latex: A_\mathrm{2-IA,Roman}^2
  roman_BTA_1:
    value: 0.0
    latex: A_\mathrm{BTA-IA,Roman}^1
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  roman_M1:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^1
  roman_M2:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^2
  roman_M3:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^3
  roman_M4:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^4
  roman_M5:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^5
  roman_M6:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^6
  roman_M7:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^7
  roman_M8:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^8
theory:
  emul_cosmic_shear:
    path: ./cobaya/cobaya/theories/
    stop_at_error: True
    extra_args: 
      device: 'cuda'
      file:  ['./projects/roman_real/emulators/nla_cosmic_shear/w0wa_hmcode2020_cs_cnn.pt']
      extra: ['./projects/roman_real/emulators/nla_cosmic_shear/w0wa_hmcode2020_cs_cnn.h5']
      ord: [['As_1e9','ns','H0','omegab', 'omegam',
             'roman_DZ_S1','roman_DZ_S2','roman_DZ_S3','roman_DZ_S4',
             'roman_DZ_S5','roman_DZ_S6','roman_DZ_S7','roman_DZ_S8',
             'roman_A1_1','roman_A1_2']]
      fast_params: [['roman_M1','roman_M2','roman_M3','roman_M4',
                     'roman_M5','roman_M6','roman_M7','roman_M8']]
      extrapar: [{'MLA': 'CNN', 
                  'INT_DIM_RES': 256, 
                  'CNN_DIM': 576, 
                  'KERNEL_DIM': 21,
                  'OUTPUT_DIM': 1080}]
"""
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# The cobaya Model evaluates the priors and the likelihood at a point;
# no cobaya sampler is involved.
model = get_model(yaml_load(yaml_string))
def chi2(p):
    """Return -2 ln(posterior) at a parameter point, priors included.

    Arguments:
      p = the sampled-parameter values in the order of
          model.parameterization.sampled_params(), as a sequence or
          as a dict (its values are taken in order)

    Returns:
      float, -2 (ln prior + ln likelihood); 1e20 where either is
      infinite or NaN, so the samplers treat the point as excluded.
      cached=False makes cobaya recompute the likelihood every time.

    Raises:
      ValueError when a parameter value is infinite or NaN.
    """
    p = [float(v) for v in p.values()] if isinstance(p, dict) else p
    if np.any(np.isinf(p)) or  np.any(np.isnan(p)):
      raise ValueError(f"At least one parameter value was infinite (CoCoa) param = {p}")
    point = dict(zip(model.parameterization.sampled_params(), p))
    res1 = model.logprior(point,make_finite=False)
    if np.isinf(res1) or  np.any(np.isnan(res1)):
      return 1e20
    res2 = model.loglike(point,
                         make_finite=False,
                         cached=False,
                         return_derived=False)
    if np.isinf(res2) or  np.any(np.isnan(res2)):
      return 1e20
    return -2.0*(res1+res2)
def chi2v2(p):
    """Return the chi2 of each likelihood and of the prior at a point.

    Arguments:
      p = the sampled-parameter values, as in chi2

    Returns:
      numpy array [-2 ln L_1, ..., -2 ln L_n, -2 ln prior], one entry
      per likelihood of the model, then the prior.
    """
    p = [float(v) for v in p.values()] if isinstance(p, dict) else p
    point = dict(zip(model.parameterization.sampled_params(), p))
    logposterior = model.logposterior(point, as_dict=True)
    chi2likes=-2*np.array(list(logposterior["loglikes"].values()))
    chi2prior=-2*np.atleast_1d(model.logprior(point,make_finite=False))
    return np.concatenate((chi2likes, chi2prior))
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def min_chi2(x0,
             cov, 
             fixed=-1, 
             nstw=200,
             nwalkers=5,
             pool=None):

    """Find the minimum of chi2 with an annealed emcee search.

    emcee is an ensemble sampler: nwalkers points (walkers) move together,
    and each proposal is built from the positions of other walkers. The
    search samples exp(-chi2/(2T)) at decreasing temperatures, so the
    colder stages gather the walkers near the minimum (see the stage loop
    below): T = 1, 0.25, 0.1, 0.005, 0.001 for a free minimization, and
    T = 0.3, 0.1, 0.005, 0.001 with a held parameter, whose starting point
    is already near the minimum. The starting point competes with the
    stage results, so the returned point is never worse than x0.

    Arguments:
      x0       = starting point, the sampled-parameter values
      cov      = covariance [ndim, ndim] of the starting clouds
      fixed    = index of a parameter held at x0[fixed], or -1 for none
      nstw     = emcee steps per walker per temperature
      nwalkers = number of walkers
      pool     = an MPI pool that evaluates the walkers, or None

    Returns:
      the best point (the held parameter left out)
    """
    def mychi2(params, *args):
        """Return chi2/T at params, with the fixed parameter put back.

        *args collects the extra positional arguments into the tuple
        args = (z, fixed, T): the held value, its index (> -1 when a
        parameter is held) and the temperature.
        """
        z, fixed, T = args
        params = np.array(params, dtype='float64')
        if fixed > -1:
            params = np.insert(params, fixed, z)
        return chi2(p=params)/T
    # Profile mode: parameter `fixed` is held at z = x0[fixed], so the
    # walkers move in the other ndim - 1 dimensions: its entry leaves x0,
    # and its row and column leave cov.
    if fixed > -1:
        z      = x0[fixed]
        x0     = np.delete(x0, (fixed))
        args = (z, fixed, 1.0)
        cov = np.delete(cov, (fixed), axis=0)
        cov = np.delete(cov, (fixed), axis=1)
    else:
        # no held parameter: the index -2 (any value <= -1) disables
        # the re-insertion in mychi2
        args = (0.0, -2.0, 1.0)

    def logprob(params, *args):
        """Return the log probability -chi2/(2T) that emcee samples;
        -inf marks an excluded point (chi2 = 1e20, inf or NaN)."""
        res = mychi2(params, *args)
        if (res > 1.e19 or np.isinf(res) or  np.isnan(res)):
          return -np.inf
        else:
          return -0.5*res
    
    class GaussianStep:
       """Gaussian draws around a point, covariance stepsize * cov.

       An instance is called like a function (that is what __call__
       defines): GaussianStep(stepsize=s)(x) is one draw centered on x.
       """
       def __init__(self, stepsize=0.2):
           """Store the draw covariance stepsize * cov."""
           self.cov = stepsize*cov
       def __call__(self, x):
           """Return one draw centered on x, an array of shape [1, ndim]."""
           return np.random.multivariate_normal(x, self.cov, size=1)
    
    ndim        = int(x0.shape[0])
    nwalkers    = int(nwalkers)
    nstw        = int(nstw)
    # the annealing ladder (colder start with a held parameter); the
    # starting clouds have covariance (T/3) cov
    if fixed == -1:
      temperature = np.array([1.0, 0.25, 0.1, 0.005, 0.001], dtype='float64')
    else:
      temperature = np.array([0.3, 0.1, 0.005, 0.001], dtype='float64')
    stepsz      = temperature/3.0

    partial_samples = [x0]
    partial = [mychi2(x0, *args)]

    # Stage i: nwalkers walkers start from a Gaussian cloud around the best
    # point so far (covariance stepsz[i] * cov = (T_i/3) cov), run nstw
    # emcee steps each on exp(-chi2/(2 T_i)), and the stage's best sample
    # seeds the next stage. The moves mix differential evolution (80%)
    # and its snooker variant (20%), both built from other walkers.
    for i in range(len(temperature)):
        x = [] # Initial point
        for j in range(nwalkers):
            x.append(GaussianStep(stepsize=stepsz[i])(x0)[0,:]) 
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, 
                                        ndim=ndim, 
                                        log_prob_fn=logprob, 
                                        args=(args[0], args[1], temperature[i]),
                                        moves=[(emcee.moves.DEMove(), 0.8),
                                               (emcee.moves.DESnookerMove(), 0.2)],
                                        pool=pool)
        sampler.run_mcmc(np.array(x,dtype='float64'), 
                         nstw, 
                         skip_initial_state_check=True)
        samples = sampler.get_chain(flat=True, discard=0)
        # the stage's sample with the highest log probability; get_chain
        # and get_log_prob list the samples in the same order
        j = np.argmin(-1.0*np.array(sampler.get_log_prob(flat=True)))
        partial_samples.append(samples[j])
        partial.append(mychi2(samples[j], *args))
        x0 = copy.deepcopy(samples[j])
        sampler.reset()
    # the best point over every stage
    j = np.argmin(np.array(partial))
    return partial_samples[j]
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def prf(x0, nstw, cov, fixed=-1, nwalkers=5, pool=None):
    """Run min_chi2 from x0 (converted to a float64 array).

    The name refers to the profile step; with fixed = -1 it is a plain
    minimization. The arguments and the result are those of min_chi2.
    """
    res =  min_chi2(x0=np.array(x0, dtype='float64'), 
                    fixed=fixed,
                    cov=cov, 
                    nstw=nstw, 
                    nwalkers=nwalkers,
                    pool=pool)
    return res
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# The block below runs only when the file is executed as a script.
# MPIPool (schwimmbad) makes MPI rank 0 the master, which runs the
# code below; every other rank waits in pool.wait() and evaluates the
# walkers the master sends until the master closes the pool.
if __name__ == '__main__':
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        # ndim sampled parameters; nwalkers = max(3 ndim, number of MPI
        # processes), so every rank has a walker to evaluate
        dim      = model.prior.d()     
        nwalkers = max(3*dim, pool.comm.Get_size())
        nstw = args.nstw

        # 1st: the covariance that sets sigma and the starting clouds: --cov
        # (a chain covmat, path relative to --root) or the prior, with which
        # the grid half-width is capped at 1 sigma -------------------------
        if args.cov is None:
          cov = model.prior.covmat(ignore_external=False) # cov from prior
          factor = min(1.0, args.factor)
        else:
          cov = np.loadtxt(args.root+args.cov)[0:model.prior.d(),0:model.prior.d()]
          factor = args.factor
        sigma = np.sqrt(np.diag(cov))

        # 2nd: the minimum, from --minfile (a minimization row: the sampled
        # parameters first, chi2 last) or computed here (slow) --------------
        if args.minfile is not None: # load minimum from running MCMC
          x0 = np.loadtxt(args.minfile)
          chi20 = x0[-1]
          x0 = x0[0:model.prior.d()]
        else: # Compute the minimum (slow)
          (x0, results) = model.get_valid_point(max_tries=1000, 
                                     ignore_fixed_ref=False,
                                     logposterior_as_dict=True)
          res = np.array(list(prf(x0=x0, 
                                  nstw=int(5.*nstw/4.), 
                                  nwalkers=nwalkers,
                                  pool=pool,
                                  cov=cov,
                                  fixed=-1)), dtype="object")
          x0 = np.array(res, dtype='float64')[0:model.prior.d()]
          chi20 = chi2(x0)
          print(f"Global Min: params = {x0}, and chi2 = {chi20}")

        # The minimum must reproduce its chi2 under this script's
        # configuration (within 0.02): a minimum from another setup fails
        if (abs(chi2(x0)-chi20)>0.02):
          raise ValueError("Inconsistency Min and Profile setups")

        # 3rd: the profile grid ---------------------------------------------
        start = np.zeros(model.prior.d(), dtype='float64')
        stop  = np.zeros(model.prior.d(), dtype='float64')
        start = x0 - factor*sigma
        stop  = x0 + factor*sigma
        
        # clip [x0 - factor sigma, x0 + factor sigma] to the bounds holding
        # 0.999999 of each prior's mass
        bounds0 = model.prior.bounds(confidence=0.999999)
        for i in range(model.prior.d()):
            if (start[i] < bounds0[i][0]):
              start[i] = bounds0[i][0]
            if (stop[i] > bounds0[i][1]):
              stop[i] = bounds0[i][1]

        # the grid is centered on the minimum with the clipped range's
        # half-width; an even number of points, plus the minimum's own value
        # inserted in the middle (index numpts//2)
        half_range = (stop[args.profile] - start[args.profile]) / 2.0
       
        numpts = args.numpts-1 if args.numpts%2 == 1 else args.numpts 
      
        param  = np.linspace(start = x0[args.profile] - half_range,
                             stop  = x0[args.profile] + half_range,
                             num = numpts)
        numpts=numpts+1
        param = np.insert(param, numpts//2, x0[args.profile])
        
        # 4th: print the profiled parameter and its grid ---------------------
        names = list(model.parameterization.sampled_params().keys()) # Cobaya Call
        print(f"nstw (evals/Temp/walkers)={args.nstw}, "
              f" param={names[args.profile]}\n"
              f"profile param values = {param}")
        
        # 5th: result rows, copies of the minimum with the profiled column set
        # to the grid; the middle row's chi2 is the minimum's ----------------
        xf = np.tile(x0, (numpts, 1))
        xf[:,args.profile] = param

        chi2res = np.zeros(numpts)  
        chi2res[numpts//2] = chi20
        
        # 6th: from the middle to the right end; each minimization starts from
        # the previous grid point's result -----------------------------------
        tmp = np.array(xf[numpts//2,:], dtype='float64')
        for i in range(numpts//2+1,numpts): 
            tmp[args.profile] = param[i]
            res = prf(tmp, 
                      fixed=args.profile,
                      nstw=int(nstw), 
                      nwalkers=nwalkers,
                      pool=pool,
                      cov=cov)
            xf[i,:] = np.insert(res, args.profile, param[i])
            tmp = np.array(xf[i,:],dtype='float64')
            chi2res[i] = chi2(xf[i,:])
            print(f"Partial ({i+1}/{numpts}): params={tmp}, and chi2={chi2res[i]}")
        
        # 7th: from the middle to the left end -------------------------------
        tmp = np.array(xf[numpts//2,:], dtype='float64')
        for i in range(numpts//2-1, -1, -1):
            tmp[args.profile] = param[i]
            res = prf(tmp, 
                      fixed=args.profile,
                      nstw=int(nstw), 
                      nwalkers=nwalkers,
                      pool=pool,
                      cov=cov)
            xf[i,:] = np.insert(res, args.profile, param[i])
            tmp = np.array(xf[i,:],dtype='float64')
            chi2res[i] = chi2(xf[i,:])
            print(f"Partial ({i+1}/{numpts}): params={tmp}, and chi2={chi2res[i]}")
        
        # 8th: append chi2v2 (each likelihood, the prior) ---------------------
        xf = np.column_stack((xf, 
                              np.array([chi2v2(d) for d in xf], dtype='float64')))

        # 9th: save [grid value, chi2, row] per grid point -------------------
        os.makedirs(os.path.dirname(f"{args.root}chains/"),exist_ok=True)
        hd = [names[args.profile],"chi2"] + names
        hd = hd + list(model.info()['likelihood'].keys()) + ["prior"]
        np.savetxt(f"{args.root}chains/{args.outroot}.{names[args.profile]}.txt",
                   np.concatenate([np.c_[param, chi2res],xf], axis=1),
                   fmt="%.9e",
                   header=f"nstw={args.nstw}, param={names[args.profile]}\n"+' '.join(hd),
                   comments="# ")
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------