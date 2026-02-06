## Running Cosmolike projects (Basic instructions) <a name="running_cosmolike_projects"></a> 

From `Cocoa/Readme` instructions:

> [!Note]
> We provide several cosmolike projects that can be loaded and compiled using `setup_cocoa.sh` and `compile_cocoa.sh` scripts. To activate them, comment the following lines on `set_installation_options.sh` 
> 
>     [Adapted from Cocoa/set_installation_options.sh shell script]
>     (...)
>
>     # ------------------------------------------------------------------------------
>     # The keys below control which cosmolike projects will be installed and compiled
>     # ------------------------------------------------------------------------------
>     #export IGNORE_COSMOLIKE_LSSTY1_CODE=1
>     #export IGNORE_COSMOLIKE_DES_Y3_CODE=1
>     #export IGNORE_COSMOLIKE_ROMAN_FOURIER_CODE=1
>     export IGNORE_COSMOLIKE_ROMAN_REAL_CODE=1
>
>     (...)
> 
>     # ------------------------------------------------------------------------------
>     # Cosmolike projects below -------------------------------------------
>     # ------------------------------------------------------------------------------
>     (...)
>     export ROMAN_REAL_URL="https://github.com/CosmoLike/cocoa_roman_real.git"
>     export ROMAN_REAL_NAME="roman_real"
>     #BRANCH: if unset, load the latest commit on the specified branch
>     #export ROMAN_REAL_BRANCH="main"
>     #COMMIT: if unset, load the specified commit
>     export ROMAN_REAL_COMMIT="23a774c32480b7b4bd5da5f637270310bc88f86c"
>     #BRANCH: if unset, load the specified TAG
>     #export ROMAN_REAL_TAG="v4.0-beta17"


> [!NOTE]
> In case users need to rerun `setup_cocoa.sh`, Cocoa will not download previously installed packages, cosmolike projects, or large datasets, unless the following keys are set on `set_installation_options.sh`
>
>     [Adapted from Cocoa/set_installation_options.sh shell script]
>     # ------------------------------------------------------------------------------
>     # OVERWRITE_EXISTING_XXX_CODE=1 -> setup_cocoa overwrites existing PACKAGES ----
>     # overwrite: delete the existing PACKAGE folder and install it again -----------
>     # redownload: delete the compressed file and download data again ---------------
>     # These keys are only relevant if you run setup_cocoa multiple times -----------
>     # ------------------------------------------------------------------------------
>     (...)
>     export OVERWRITE_EXISTING_ALL_PACKAGES=1    # except cosmolike projects
>     #export OVERWRITE_EXISTING_COSMOLIKE_CODE=1 # dangerous (possible loss of uncommitted work)
>                                                 # if unset, users must manually delete cosmolike projects
>     #export REDOWNLOAD_EXISTING_ALL_DATA=1      # warning: some data is many GB
>

> [!NOTE]
> If users want to recompile cosmolike, there is no need to rerun the Cocoa general scripts. Instead, run the following three commands:
>
>      source start_cocoa.sh
>
> and
> 
>      source ./installation_scripts/setup_cosmolike_projects.sh
>
> and
> 
>       source ./installation_scripts/compile_all_projects.sh
> 
> or (in case users just want to compile the lsst-y1 project)
>
>       source ./projects/roman_real/scripts/compile_roman_real.sh

> [!TIP]
> Assuming Cocoa is installed on a local (not remote!) machine, type the command below after step 2️⃣ to run Jupyter Notebooks.
>
>     jupyter notebook --no-browser --port=8888
>
> The terminal will then show a message similar to the following template:
>
>     (...)
>     [... NotebookApp] Jupyter Notebook 6.1.1 is running at:
>     [... NotebookApp] http://f0a13949f6b5:8888/?token=XXX
>     [... NotebookApp] or http://127.0.0.1:8888/?token=XXX
>     [... NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
>
> Now go to the local internet browser and type `http://127.0.0.1:8888/?token=XXX`, where XXX is the previously saved token displayed on the line
> 
>     [... NotebookApp] or http://127.0.0.1:8888/?token=XXX
>
> The project roman_real contains jupyter notebook examples located at `projects/roman_real`.

To run the example

 **Step :one:**: activate the cocoa Conda environment,  and the private Python environment 
    
      conda activate cocoa

and

      source start_cocoa.sh
 
 **Step :two:**: Select the number of OpenMP cores (below, we set it to 8).
    
    export OMP_PROC_BIND=close; export OMP_NUM_THREADS=8; export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE
      
 **Step :three:**: The folder `projects/roman_real` contains examples. So, run the `cobaya-run` on the first example following the commands below.

- **One model evaluation**:

  - Linux

        mpirun -n 1 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --report-bindings \
           --bind-to core:overload-allowed --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
           cobaya-run ./projects/roman_real/EXAMPLE_EVALUATE1.yaml -f

  -  macOS (arm)

         mpirun -n 1 --oversubscribe cobaya-run ./projects/roman_real/EXAMPLE_EVALUATE1.yaml -f

- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux

        mpirun -n 4 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --report-bindings \
           --bind-to core:overload-allowed --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
           cobaya-run ./projects/roman_real/EXAMPLE_MCMC1.yaml -f

   -  macOS (arm)
     
          mpirun -n 4 --oversubscribe cobaya-run ./projects/roman_real/EXAMPLE_MCMC1.yaml -f

# Running ML emulators <a name="cobaya_base_code_examples_emul"></a>

Cocoa contains a few transformer- and CNN-based neural network emulators capable of simulating the CMB, cosmolike outputs, matter power spectrum, and distances. We provide a few scripts that exemplify their API. To run them, users ensure the following lines are commented out in `set_installation_options.sh` before running the `setup_cocoa.sh` and `compile_cocoa.sh`. By default, these lines should be commented out, but it is worth checking.

      [Adapted from Cocoa/set_installation_options.sh shell script] 
      # insert the # symbol (i.e., unset these environmental keys  on `set_installation_options.sh`)
      #export IGNORE_EMULTRF_CODE=1              #SaraivanovZhongZhu (SZZ) transformer/CNN-based emulators
      #export IGNORE_EMULTRF_DATA=1            
      #export IGNORE_LIPOP_LIKELIHOOD_CODE=1     # to run EXAMPLE_EMUL_(EVALUATE/MCMC/NAUTILUS/EMCEE1).yaml
      #export IGNORE_LIPOP_CMB_DATA=1           
      #export IGNORE_ACTDR6_CODE=1               # to run EXAMPLE_EMUL_(EVALUATE/MCMC/NAUTILUS/EMCEE1).yaml
      #export IGNORE_ACTDR6_DATA=1         
      #export IGNORE_NAUTILUS_SAMPLER_CODE=1     # to run PROJECTS/EXAMPLE/EXAMPLE_EMUL_NAUTILUS1.py
      #export IGNORE_POLYCHORD_SAMPLER_CODE=1    # to run PROJECTS/EXAMPLE/EXAMPLE_EMUL_POLY1.yaml
      #export IGNORE_GETDIST_CODE=1              # to run EXAMPLE_TENSION_METRICS.ipynb
      #export IGNORE_TENSIOMETER_CODE=1          # to run EXAMPLE_TENSION_METRICS.ipynb
      
> [!TIP]
> What if users have not configured ML-related keys before sourcing `setup_cocoa.sh`?
> 
> Answer: Comment the keys below before rerunning `setup_cocoa.sh`.
> 
>     [Adapted from Cocoa/set_installation_options.sh shell script]
>     # These keys are only relevant if you run setup_cocoa multiple times
>     #export OVERWRITE_EXISTING_ALL_PACKAGES=1    
>     #export OVERWRITE_EXISTING_COSMOLIKE_CODE=1 
>     #export REDOWNLOAD_EXISTING_ALL_DATA=1

Now, users must follow all the steps below.

 **Step :one:**: Activate the private Python environment by sourcing the script `start_cocoa.sh`

    source start_cocoa.sh

 **Step :two:**: Ensure OpenMP is **OFF**.
    
    export OMP_NUM_THREADS=1
    
 **Step :three:** Run `cobaya-run` on the first emulator example following the commands below.

 - **One model evaluation**:

  - Linux
    
        mpirun -n 1 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            cobaya-run ./projects/roman_real/EXAMPLE_EMUL_EVALUATE1.yaml -f

  - macOS (arm)
 
         mpirun -n 1 --oversubscribe  cobaya-run ./projects/roman_real/EXAMPLE_EMUL_EVALUATE1.yaml -f

- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux
    
        mpirun -n 4 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            cobaya-run ./projects/roman_real/EXAMPLE_EMUL_MCMC1.yaml -r

  - macOS (arm)

        mpirun -n 4 --oversubscribe cobaya-run ./projects/roman_real/EXAMPLE_EMUL_MCMC1.yaml -r

- **Halofit Comparison**

  The scripts that generated the plots below are provided at `scripts/EXAMPLE_PLOT_COMPARE_CHAINS_EMUL[1-4].py`.

  <p align="center">
  <img width="750" height="750" alt="project_roman_real_plot_halofit_comparison_1" src="https://github.com/user-attachments/assets/e9779555-e27c-4d19-a1f0-0533267762ab" />
  </p>

- **PolyChord**:

  - Linux
    
        mpirun -n 90 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            cobaya-run ./projects/roman_real/EXAMPLE_EMUL_POLY1.yaml -r

  - macOS (arm)

        mpirun -n 12 --oversubscribe cobaya-run ./projects/roman_real/EXAMPLE_EMUL_POLY1.yaml -r

> [!Note]
> When running `PolyChord` or any of our scripts in more than one node, replace `--mca btl vader,tcp,self` by `--mca btl tcp,self`.

- **Nautilus**:

  - Linux
    
        mpirun -n 90 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            python -m mpi4py.futures ./projects/roman_real/EXAMPLE_EMUL_NAUTILUS1.py \
                --root ./projects/roman_real/ --outroot "EXAMPLE_EMUL_NAUTILUS1"  \
                --maxfeval 750000 --nlive 2048 --neff 15000 --flive 0.01 --nnetworks 5

  - macOS (arm)

        mpirun -n 12 --oversubscribe python -m mpi4py.futures ./projects/roman_real/EXAMPLE_EMUL_NAUTILUS1.py \
                --root ./projects/roman_real/ --outroot "EXAMPLE_EMUL_NAUTILUS1"  \
                --maxfeval 750000 --nlive 2048 --neff 15000 --flive 0.01 --nnetworks 5

- **Emcee**:

  - Linux
    
        mpirun -n 51 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            python ./projects/roman_real/EXAMPLE_EMUL_EMCEE1.py --root ./projects/roman_real/ \
                --outroot "EXAMPLE_EMUL_EMCEE1" --maxfeval 1000000

  - macOS (arm)

        mpirun -n 12 --oversubscribe python ./projects/roman_real/EXAMPLE_EMUL_EMCEE1.py \
            --root ./projects/roman_real/ --outroot "EXAMPLE_EMUL_EMCEE1" --maxfeval 1000000

  The number of steps per MPI worker is $n_{\\rm sw} =  {\\rm maxfeval}/n_{\\rm w}$,
  with the number of walkers being $n_{\\rm w}={\\rm max}(3n_{\\rm params},n_{\\rm MPI})$.
  For proper convergence, each walker should traverse 50 times the autocorrelation length ($\tau$),
  which is provided in the header of the output chain file. A reasonable rule of thumb is to assume
  $\tau > 200$ and therefore set ${\\rm maxfeval} > 10,000 \times n_{\\rm w}$.
  Finally, our code sets burn-in (per walker) at $5 \times \tau$.

  With these numbers, users may ask when `Emcee` is preferable to `Metropolis-Hastings`?
  Here are a few numbers based on our `Planck CMB (l < 396) + SN + BAO + LSST-Y1` test case.
  1) `MH` achieves convergence with $n_{\\rm sw} \sim 150,000$ (number of steps per walker), but only requires four walkers.
  2) `Emcee` has $\tau \sim 300$, so it requires $n_{\\rm sw} \sim 15,000$ when running with $n_{\\rm w}=114$.
  
  Conclusion: `Emcee` requires $\sim 3$ more evaluations in this case, but the number of evaluations per MPI worker (assuming one MPI worker per walker) is reduced by $\sim 10$.
  Therefore, `Emcee` seems well-suited for chains where the evaluation of a single cosmology is time-consuming (and there is no slow/fast decomposition).

  What if the user runs an `Emcee` chain with `maxeval` insufficient for convergence? `Emcee` saves the chain checkpoint at `chains/outroot.h5`.

- **Sampler Comparison**

  The scripts that generated the plots below are provided at `scripts/EXAMPLE_PLOT_COMPARE_CHAINS_EMUL[1-4].py`.

  <p align="center">
  <img width="750" height="750" alt="project_roman_real_plot_sampler_comparison_1" src="https://github.com/user-attachments/assets/bad7e26a-fc2b-4370-a11a-1b6aefd0bab9" />
  </p>
  
- **Global Minimizer**:

  Our minimizer is a reimplementation of `Procoli`, developed by Karwal et al (arXiv:2401.14225) 

  - Linux
    
        mpirun -n 51 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            python ./projects/roman_real/EXAMPLE_EMUL_MINIMIZE1.py --root ./projects/roman_real/ \
                --outroot "EXAMPLE_EMUL_MIN1" --nstw 450

  - macOS (arm)

        mpirun -n 12 python ./projects/roman_real/EXAMPLE_EMUL_MINIMIZE1.py --root ./projects/roman_real/ \
              --outroot "EXAMPLE_EMUL_MIN1" --nstw 450

  The number of steps per Emcee walker per temperature is $n_{\\rm stw}$,
  and the number of walkers is $n_{\\rm w}={\\rm max}(3n_{\\rm params},n_{\\rm MPI})$.
  The minimum number of total evaluations is $3n_{\\rm params} \times n_{\rm T} \times n_{\\rm stw}$, which can be distributed among $n_{\\rm MPI} = 3n_{\\rm params}$ MPI processes for faster results.
    
  The scripts that generated the plots below are provided at `scripts/EXAMPLE_PLOT_MIN_COMPARE_CONV[1-2].py`

  <p align="center">
  <img width="750" height="750" alt="compare_min_roman_real" src="https://github.com/user-attachments/assets/1db89ed7-c186-4967-85a6-1f9b59e1f3ce" />
  </p>

  In our testing, $n_{\\rm stw} \sim 250$ worked reasonably well up to $n_{\rm param} \sim \mathcal{O}(10)$.

- **Profile**: 

  - Linux
    
          mpirun -n 51 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            python ./projects/roman_real/EXAMPLE_EMUL_PROFILE1.py \
              --root ./projects/roman_real/ --cov 'chains/EXAMPLE_EMUL_MCMC1.covmat' \
              --outroot "EXAMPLE_EMUL_PROFILE1" --factor 3 --nstw 450 --numpts 10 \
              --profile ${SLURM_ARRAY_TASK_ID} \
              --minfile="./projects/roman_real/chains/EXAMPLE_EMUL_MIN1.txt"

  -  macOS (arm)

          mpirun -n 51 --oversubscribe python ./projects/roman_real/EXAMPLE_EMUL_PROFILE1.py \
              --root ./projects/roman_real/ --cov 'chains/EXAMPLE_EMUL_MCMC1.covmat' \
              --outroot "EXAMPLE_EMUL_PROFILE1" --factor 3 --nstw 450 --numpts 10 \
              --profile ${SLURM_ARRAY_TASK_ID} \
              --minfile="./projects/roman_real/chains/EXAMPLE_EMUL_MIN1.txt"

  The argument `factor` specifies the start and end of the parameter being profiled:

      start value ~ mininum value - factor*np.sqrt(np.diag(cov))
      end   value ~ mininum value + factor*np.sqrt(np.diag(cov))

  We advise ${\rm factor} \sim 3$ for parameters that are well constrained by the data when a covariance matrix is provided.
  If `cov` is not supplied, the code estimates one internally from the prior.
  If a parameter is poorly constrained or `cov` is not given, we recommend ${\rm factor} \ll 1$.

  The script of the plot below is provided at `projects/roman_real/scripts/EXAMPLE_PLOT_PROFILE[1-2].py`

  Profile 1: `LSST-Y1 Cosmic Shear only`

  <p align="center">
  <img width="1156" height="858" alt="example_lssty1_profile1" src="https://github.com/user-attachments/assets/2fea9d3c-524a-49d9-ae89-cb2bb26594e9" />
  </p>

> [!Warning]
> When running Profiles, you should never set flat priors on parameters that are not well constrained by the data. 
> By doing that, you then risk having the profile selecting values near the boundary of parameter space (a  big problem, especially when using emulators)
> You can convert a flat prior to a Gaussian one by setting the standard deviation to be $\sigma^2 = (hi - lo)^2/12$, where (lo, hi) are the prior boundaries

# Running Hybrid Cosmolike-ML emulators <a name="cobaya_base_code_examples_emul2"></a>

> [!Warning]
> The code and examples associated with this section are still in alpha stage

Our main line of research involves emulators that simulate the entire Cosmolike data vectors, and each project (LSST, Roman, DES) contains its own README with emulator examples. The speed of such emulators is incredible, especially when GPUs are available, and our emulators do take advantage of the CPU-GPU integration on Apple MX chips. For example, the average timing of lsst-y1 cosmic shear data vector emulation is around 0.005s ($\sim$ 200828 evaluations in $\sim$ 850.5 seconds) on a macOS M2 Pro.

While the data vector emulators are incredibly fast, there is an intermediate approach that emulates only the Boltzmann outputs (comoving distance, linear and nonlinear matter power spectrum). This hybrid-ML case can offer greater flexibility, especially in the initial phases of a research project, as changes to the modeling of nuisance parameters or to the assumed galaxy distributions do not require retraining of the network. 

Examples in the hybrid case all have the prefix **EXAMPLE_EMUL2** (note the `2`). The required flags on `set_installation_options.sh` are similar to what we showed in the previous emulator section.

Now, users must follow all the steps below.

 **Step :one:**: Activate the private Python environment by sourcing the script `start_cocoa.sh`

    source start_cocoa.sh

 **Step :two:**: Select the number of OpenMP cores. Below, we set it to 4, the ideal setting for hybrid examples.

  - Linux
    
        export OMP_NUM_THREADS=4; export OMP_PROC_BIND=close; export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE

  - macOS (arm)
    
        export OMP_NUM_THREADS=4; export OMP_PROC_BIND=disabled; export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE
    
 **Step :three:** Run `cobaya-run` on the first emulator example, following the commands below (here we only provide lsst-y1 examples).

- **One model evaluation**:

  - Linux
    
        mpirun -n 1 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --report-bindings \
           --bind-to core:overload-allowed --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
           cobaya-run ./projects/roman_real/EXAMPLE_EMUL2_EVALUATE1.yaml -f

  - macOS (arm)
    
        mpirun -n 1 --oversubscribe  cobaya-run ./projects/roman_real/EXAMPLE_EMUL2_EVALUATE1.yaml -f
    
- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux
    
        mpirun -n 4 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --report-bindings \
            --bind-to core:overload-allowed --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
            cobaya-run ./projects/roman_real/EXAMPLE_EMUL_EMUL2_MCMC1.yaml -r

  - macOS (arm)

        mpirun -n 4 --oversubscribe cobaya-run ./projects/roman_real/EXAMPLE_EMUL2_MCMC1.yaml -r
    
Details on the matter power spectrum emulator designs will be presented in the [emulator_code](https://github.com/CosmoLike/emulators_code) repository. Basically, we apply standard neural network techniques to generalize the *syren-new* Eq. 6 of [arXiv:2410.14623](https://arxiv.org/abs/2410.14623) formula for the linear power spectrum (w0waCDM with a fixed neutrino mass of $0.06$ eV) to new models, extended ranges, or higher precision. Similarly, we use networks to generalize the *syren-Halofit* LCDM nonlinear boost fit (Eq. 11 of [arXiv:2402.17492](https://arxiv.org/abs/2402.17492)).

> [!NOTE] 
> Users can decide not to correct the *syren-new* formula for the linear power spectrum (flag in the yaml). Although we have not conducted extensive studies of the caveats of the syren-new approximation, it appears sufficient for w0waCDM forecasts when combined with the Euclid Emulator to compute the nonlinear boost.
>
> For back-of-the-envelope LCDM calculations (e.g., to test cosmolike features), users can also choose not to correct the *syren-Halofit* formula for the LCDM nonlinear boost (see figure below). In this case, the overhead on top of cosmolike computations is minimum, at the order of $0.01$ seconds on a macOS M2Pro laptop. 

- **Emulator Comparison**

  The scripts that generated the plots below are provided at `scripts/EXAMPLE_PLOT_COMPARE_CHAINS_EMUL[1-4].py`.

  <p align="center">
  <img width="750" height="750" alt="project_roman_real_plot_sampler_comparison_1" src="https://github.com/user-attachments/assets/a85b3fcc-d82c-4c4a-9341-677445a03dd0" />
  </p>
