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

        mpirun -n 1 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
           --bind-to core:overload-allowed --report-bindings  \
           --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
           cobaya-run ./projects/roman_real/EXAMPLE_EVALUATE1.yaml -f

  -  macOS (arm)

         mpirun -n 1 --oversubscribe cobaya-run ./projects/roman_real/EXAMPLE_EVALUATE1.yaml -f

- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux

        mpirun -n 4 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
           --bind-to core:overload-allowed --report-bindings  \
           --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
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
    
        mpirun -n 1 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --rank-by slot \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            cobaya-run ./projects/roman_real/EXAMPLE_EMUL_EVALUATE1.yaml -f

  - macOS (arm)
 
         mpirun -n 1 --oversubscribe  cobaya-run ./projects/roman_real/EXAMPLE_EMUL_EVALUATE1.yaml -f

- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux
    
        mpirun -n 4 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --rank-by slot \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            cobaya-run ./projects/roman_real/EXAMPLE_EMUL_MCMC1.yaml -r

  - macOS (arm)

        mpirun -n 4 --oversubscribe cobaya-run ./projects/roman_real/EXAMPLE_EMUL_MCMC1.yaml -r

- **PolyChord**:

  - Linux
    
        mpirun -n 90 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --rank-by slot \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            cobaya-run ./projects/roman_real/EXAMPLE_EMUL_POLY1.yaml -r

  - macOS (arm)

        mpirun -n 12 --oversubscribe cobaya-run ./projects/roman_real/EXAMPLE_EMUL_POLY1.yaml -r

> [!Note]
> When running `PolyChord` or any of our scripts in more than one node, replace `--mca btl vader,tcp,self` by `--mca btl tcp,self`.

- **Nautilus**:

  - Linux
    
        mpirun -n 90 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --rank-by slot \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            python -m mpi4py.futures ./projects/roman_real/EXAMPLE_EMUL_NAUTILUS1.py \
                --root ./projects/lsst_y1/ --outroot "EXAMPLE_EMUL_NAUTILUS1"  \
                --maxfeval 750000 --nlive 2048 --neff 15000 --flive 0.01 --nnetworks 5

  - macOS (arm)

        mpirun -n 12 --oversubscribe python -m mpi4py.futures ./projects/lsst_y1/EXAMPLE_EMUL_NAUTILUS1.py \
                --root ./projects/roman_real/ --outroot "EXAMPLE_EMUL_NAUTILUS1"  \
                --maxfeval 750000 --nlive 2048 --neff 15000 --flive 0.01 --nnetworks 5


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
    
        mpirun -n 1 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
           --bind-to core:overload-allowed --report-bindings  \
           --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
           cobaya-run ./projects/roman_real/EXAMPLE_EMUL2_EVALUATE1.yaml -f

  - macOS (arm)
    
        mpirun -n 1 --oversubscribe  cobaya-run ./projects/roman_real/EXAMPLE_EMUL2_EVALUATE1.yaml -f
    
- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux
    
        mpirun -n 4 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --rank-by slot \
            --bind-to core:overload-allowed --map-by slot \
            cobaya-run ./projects/roman_real/EXAMPLE_EMUL_EMUL2_MCMC1.yaml -r

  - macOS (arm)

        mpirun -n 4 --oversubscribe cobaya-run ./projects/roman_real/EXAMPLE_EMUL2_MCMC1.yaml -r
    
Details on the matter power spectrum emulator designs will be presented in the [emulator_code](https://github.com/CosmoLike/emulators_code) repository. Basically, we apply standard neural network techniques to generalize the *syren-new* Eq. 6 of [arXiv:2410.14623](https://arxiv.org/abs/2410.14623) formula for the linear power spectrum (w0waCDM with a fixed neutrino mass of $0.06$ eV) to new models, extended ranges, or higher precision. Similarly, we use networks to generalize the *syren-Halofit* LCDM nonlinear boost fit (Eq. 11 of [arXiv:2402.17492](https://arxiv.org/abs/2402.17492)).

> [!NOTE] 
> Users can decide not to correct the *syren-new* formula for the linear power spectrum (flag in the yaml). Although we have not conducted extensive studies of the caveats of the syren-new approximation, it appears sufficient for w0waCDM forecasts when combined with the Euclid Emulator to compute the nonlinear boost.
>
> For back-of-the-envelope LCDM calculations (e.g., to test cosmolike features), users can also choose not to correct the *syren-Halofit* formula for the LCDM nonlinear boost (see figure below). In this case, the overhead on top of cosmolike computations is minimum, at the order of $0.01$ seconds on a macOS M2Pro laptop. 
