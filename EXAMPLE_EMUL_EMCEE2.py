"""Sample the Roman real-space cosmic-shear posterior with emcee.

The likelihood is roman_real.cosmic_shear evaluated by the neural-network
emulator of the whole cosmic-shear data vector (use_emulator: 1, NLA
intrinsic alignments).
Model: w0waCDM (w and w0pwa = w0 + wa sampled).
Nonlinear power of the emulator's training set: Halofit (Takahashi).

emcee is an ensemble sampler: nwalkers = max(3 ndim, number of MPI
processes) walkers move together, each proposal built from the positions
of other walkers (differential-evolution moves, 80%, and their snooker
variant, 20%). Every walker starts from its own valid point drawn from
the yaml's ref distributions and takes maxfeval/nwalkers steps. The
autocorrelation time tau of the chains sets the burn-in (the first
5 max(tau) steps are discarded) and the thinning (every 0.5 min(tau)-th
step is kept).

Outputs in <root>chains/: <outroot>.1.txt (columns: weight 1, ln
posterior, the sampled parameters, chi2 = -2 ln posterior), .ranges,
.paramnames and .covmat (for getdist), and the emcee checkpoint
<outroot>.h5.

Run from cocoa/Cocoa with MPI, for example (one command, wrapped here):

    mpirun -n 12 --oversubscribe python ./projects/roman_real/EXAMPLE_EMUL_EMCEE2.py
        --root ./projects/roman_real/ --outroot EXAMPLE_EMUL_EMCEE2
        --maxfeval 1000000
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
from emcee.autocorr import AutocorrError
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from getdist import IniFile
from getdist import loadMCSamples
from schwimmbad import MPIPool
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(prog='EXAMPLE_EMUL_EMCEE')

parser.add_argument("--maxfeval",
                    dest="maxfeval",
                    help="Minimizer: maximum number of likelihood evaluations",
                    type=int,
                    nargs='?',
                    const=1,
                    default=5000)
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
parser.add_argument("--progress",
                    dest="progress",
                    help="Show Emcee Progress",
                    nargs='?',
                    type=bool,
                    default=False)
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
  w0pwa:
    prior:
      min: -5 
      max: -0.01
    ref:
      dist: norm
      loc: -0.99
      scale: 0.05
    proposal: 0.05
    latex: w_{0,\mathrm{DE}}+w_{a,\mathrm{DE}}
  w:
    prior:
      min: -3
      max: -0.01
    ref:
      dist: norm
      loc: -0.99
      scale: 0.05
    proposal: 0.05
    latex: w_{0,\mathrm{DE}}
  wa:
    derived: 'lambda w0pwa, w: w0pwa - w'
    latex: 'w_{a,\mathrm{DE}}'
  As:
    derived: 'lambda As_1e9: 1e-9 * As_1e9'
    latex: A_\mathrm{s}
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
theory:
  emul_cosmic_shear:
    path: ./cobaya/cobaya/theories/
    stop_at_error: True
    extra_args: 
      device: 'cuda'
      file:  ['./projects/roman_real/emulators/nla_cosmic_shear/w0wa_takahashi_cs_cnn.pt']
      extra: ['./projects/roman_real/emulators/nla_cosmic_shear/w0wa_takahashi_cs_cnn.h5']
      ord: [['As_1e9','ns','H0','omegab', 'omegam', 'w', 'w0pwa',
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
    res1 = model.logprior(point, make_finite=False)
    if np.isinf(res1) or  np.any(np.isnan(res1)):
      return 1.e20
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
def chain(x0,
          ndim,
          nwalkers,
          cov,
          names,
          maxfeval=3000, 
          pool=None,
          checkpoint=None): 

    """Run the emcee chains and return the thinned samples after burn-in.

    Arguments:
      x0       = starting points, one row per walker, [nwalkers, ndim]
      ndim     = number of sampled parameters
      nwalkers = number of walkers
      cov      = the prior covariance (not used)
      names    = the sampled-parameter names, in order
      maxfeval = emcee steps per walker
      pool     = an MPI pool that evaluates the walkers, or None
      checkpoint = the HDF5 file emcee writes every step to

    Returns:
      [samples, tau]: samples is an [nsamples, ndim + 3] array with columns
      (weight 1, ln posterior, the parameters, chi2 = -2 ln posterior);
      tau holds the autocorrelation time of each parameter.
    """
    def logprob(params, *args):
        """Return ln(posterior) = -chi2/2, or -inf for an excluded
        point; *args collects any extra arguments (none are used)."""
        res = chi2(params)
        if (res > 1.e19 or np.isinf(res) or  np.isnan(res)):
          return -np.inf
        else:
          return -0.5*res

    backend = emcee.backends.HDFBackend(checkpoint)
    
    sampler = emcee.EnsembleSampler(nwalkers=nwalkers, 
                                    ndim=ndim, 
                                    log_prob_fn=logprob, 
                                    parameter_names=names,
                                    moves=[(emcee.moves.DEMove(), 0.8),
                                           (emcee.moves.DESnookerMove(), 0.2)],
                                    pool=pool,
                                    backend=backend)
    sampler.run_mcmc(x0, 
                     maxfeval, 
                     skip_initial_state_check=True, 
                     progress=args.progress)
    
    # tau: integrated autocorrelation time per parameter (quiet=True
    # warns instead of failing when the chains are shorter than 50 tau)
    tau = sampler.get_autocorr_time(quiet=True, has_walkers=True)
    print(f"Partial Result: tau = {tau}, nwalkers={nwalkers}")

    burn_in = int(5*np.max(tau))
    thin    = int(0.5*np.min(tau))
    xf      = sampler.get_chain(flat=True, discard=burn_in, thin=thin)
    lnpf    = sampler.get_log_prob(flat=True, discard=burn_in, thin=thin)
    weights = np.ones((len(xf),1), dtype='float64')
    local_chi2 = -2*lnpf
    
    return [np.concatenate([weights,
                           lnpf[:,None], 
                           xf, 
                           local_chi2[:,None]], axis=1), 
            tau]

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
        
        dim      = model.prior.d()                                      # Cobaya call
        bounds   = model.prior.bounds(confidence=0.999999)              # Cobaya call
        names    = list(model.parameterization.sampled_params().keys()) # Cobaya Call
        nwalkers = max(3*dim,pool.comm.Get_size())
        # nwalkers = max(3 ndim, number of MPI processes); maxfeval is the
        # total budget, maxevals the steps of each walker
        maxevals = int(args.maxfeval/(nwalkers))
        print(f"\n\n\n"
              f"maxfeval={args.maxfeval}, "
              f"nwalkers={nwalkers}, "
              f"maxfeval per walker = {maxevals}"
              f"\n\n\n")
        # starting points: one valid point per walker, drawn from the ref
        # distributions of the yaml ----------------------------------------
        x0 = [] # Initial point x0
        for j in range(nwalkers):
          (tmp_x0, tmp) = model.get_valid_point(max_tries=10000, 
                                                ignore_fixed_ref=False,
                                                logposterior_as_dict=True)
          x0.append(tmp_x0[0:dim])
        x0 = np.array(x0, dtype='float64')
        
        # prior covariance (passed to chain, which does not use it) ---------
        cov = model.prior.covmat(ignore_external=False) # cov from prior
        
        # the HDF5 checkpoint emcee writes every step to --------------------
        checkpoint = f"{args.root}chains/{args.outroot}.h5"
      
        # run the chains -------------------------------------------------------
        res = chain(x0=np.array(x0, dtype='float64'),
                    ndim=dim,
                    nwalkers=nwalkers,
                    cov=cov, 
                    names=names,
                    maxfeval=maxevals,
                    pool=pool,
                    checkpoint=checkpoint)

        # the samples: weight, ln posterior, parameters, chi2 ----------------
        os.makedirs(os.path.dirname(f"{args.root}chains/"),exist_ok=True)
        hd=f"nwalkers={nwalkers}, maxfeval={args.maxfeval}, max tau={res[1]}\n"
        np.savetxt(f"{args.root}chains/{args.outroot}.1.txt",
                   res[0],
                   fmt="%.7e",
                   header=hd + ' '.join(names),
                   comments="# ")
        # .ranges: the bounds of each parameter (first line: the column
        # names) -------------------------------------------------------------
        hd = ["weights","lnp"] + names + ["chi2*"]
        rows = [(str(n),float(l),float(h)) for n,l,h in zip(names, bounds[:,0], bounds[:,1])]
        with open(f"{args.root}chains/{args.outroot}.ranges", "w") as f: 
          f.write(f"# {' '.join(hd)}\n")
          f.writelines(f"{n} {l:.5e} {h:.5e}\n" for n, l, h in rows)

        # .paramnames: name and LaTeX label per column; the * of chi2* marks
        # a derived parameter ------------------------------------------------
        param_info = model.info()['params']
        latex  = [param_info[x]['latex'] for x in names]
        names.append("chi2*")
        latex.append("\\chi^2")
        np.savetxt(f"{args.root}chains/{args.outroot}.paramnames", 
                   np.column_stack((names,latex)),
                   fmt="%s")
    
        # .covmat: the parameter covariance of the saved samples, read back
        # with getdist -------------------------------------------------------
        samples = loadMCSamples(f"{args.root}chains/{args.outroot}",
                                settings={'ignore_rows': u'0.0'})
        np.savetxt(f"{args.root}chains/{args.outroot}.covmat",
                   np.array(samples.cov(), dtype='float64'),
                   fmt="%.5e",
                   header=' '.join(names),
                   comments="# ")
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------