## Running Cosmolike projects (Basic instructions) <a name="running_cosmolike_projects"></a> 

> [!Note]
> We provide several cosmolike projects that can be loaded and compiled using `setup_cocoa.sh` and `compile_cocoa.sh` scripts. To activate them, comment the following lines on `set_installation_options.sh` 
> 
>     [Adapted from Cocoa/set_installation_options.sh shell script] 
>     # The keys below control which cosmolike projects will be installed and compiled 
>     # inset # symbol in the lines below (i.e., unset these environmental keys)
>     #export IGNORE_COSMOLIKE_LSSTY1_CODE=1
>     export IGNORE_COSMOLIKE_DES_Y3_CODE=1
>     export IGNORE_COSMOLIKE_ROMAN_FOURIER_CODE=1
>     export IGNORE_COSMOLIKE_ROMAN_REAL_CODE=1
>
> If users comment these lines (unsetting the corresponding IGNORE keys) after running `setup_cocoa.sh` and `compile_cocoa.sh`, there is no need to rerun these general scripts, which would reinstall many packages (slow). Instead, run the following three commands:
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

To run the example

 **Step :one:**: activate the cocoa Conda environment,  and the private Python environment 
    
      conda activate cocoa

and

      source start_cocoa.sh
 
 **Step :two:**: Select the number of OpenMP cores (below, we set it to 8).
    
    export OMP_PROC_BIND=close; export OMP_NUM_THREADS=8
      
### Examples not involving Cosmolike

 **Step :three:**: The folder `projects/example` contains examples. So, run the `cobaya-run` on the first example following the commands below.

One model evaluation:
      
       mpirun -n 1 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --bind-to core:overload-allowed --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/roman_real/EXAMPLE_EVAL1.yaml -f
 
MCMC:

      mpirun -n 1 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --bind-to core:overload-allowed --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/roman_real/EXAMPLE_MCMC1.yaml -f

## Running Cosmolike projects (Advanced instructions in case you want to clone this manually - not advised) <a name="running_cosmolike_projects"></a> 
(From lsst_y1 repository)

In this tutorial, we assume the user installed Cocoa via the *Conda installation* method, and the name of the Conda environment is `cocoa`. We also presume the user's terminal is in the folder where Cocoa was cloned.

 **Step :one:**: activate the cocoa Conda environment, go to the `cocoa/Cocoa/projects` folder, and clone this project:
    
      conda activate cocoa

and

      cd ./cocoa/Cocoa/projects

and

      ${CONDA_PREFIX}/bin/git clone --depth 1 https://github.com/CosmoLike/cocoa_roman_real.git --branch hb_roman roman_real

:warning: Cocoa scripts and YAML files assume the removal of the `cocoa_` prefix when cloning the repository.
      
 **Step :two:**: go back to the Cocoa main folder and activate the private Python environment
    
      cd ../

and

      source start_cocoa.sh
 
:warning: Remember to run the `start_cocoa.sh` shell script only **after cloning** the project repository (or if you already in the `(.local)` environment, run `start_cocoa.sh` again). 

**Step :three:**: compile the project
 
      source ./projects/roman_real/scripts/compile_roman_real.sh

:interrobang: The script `compile_cocoa.sh` also compiles every Cosmolike project on the `cocoa/Cocoa/projects/` folder.

**Step :four:**: select the number of OpenMP cores (below, we set it to 4), and run a template YAML file

    
      export OMP_PROC_BIND=close; export OMP_NUM_THREADS=8
      
One model evaluation:
      
       mpirun -n 1 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --bind-to core:overload-allowed --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/roman_real/EXAMPLE_EVAL1.yaml -f
 
MCMC:

      mpirun -n 1 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --bind-to core:overload-allowed --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/roman_real/EXAMPLE_MCMC1.yaml -f
      

## Deleting Cosmolike projects <a name="running_cosmolike_projects"></a>

Do not delete the `roman_real` folder from the project folder without first running the shell script `stop_cocoa.sh`. Otherwise, Cocoa will have ill-defined soft links. 

:interrobang: Where the ill-defined soft links will be located? 
     
     Cocoa/cobaya/cobaya/likelihoods/
     Cocoa/external_modules/code/
     Cocoa/external_modules/data/ 
    
:interrobang: Why does Cocoa behave like this? The shell script `start_cocoa.sh` creates symbolic links so Cobaya can see the likelihood and data files. Cocoa also adds the Cobaya-Cosmolike interface of all cosmolike-related projects to the `LD_LIBRARY_PATH` and `PYTHONPATH` environmental variables.
