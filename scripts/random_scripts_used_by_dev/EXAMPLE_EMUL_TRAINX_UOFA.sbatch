#!/bin/bash

#SBATCH --job-name=TREMUL
#SBATCH --output=TREMUL-%A_%a.out
#SBATCH --error TREMUL-%A_%a.out.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=5
#SBATCH --partition=long-40core
#SBATCH --time=48:00:00

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
module load slurm

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"
if [ -z "${ROOTDIR}" ]; then
  sleep $(( 5 + SLURM_ARRAY_TASK_ID*25 )) # help avoid different scripts running start_cocoa.sh simultaneously
                                          # clash can still happen when one script runs start_cocoa.sh, the other
                                          # script is reading a file (start_cocoa.sh invokes stop_cocoa.sh)
                                          # which delete links and ROOTDIR so the running script can find file/likelihood
  conda activate cocoa
  source start_cocoa.sh
fi

export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

declare -a temp=( "512" "256" "256" )
declare -a mode=( 'train' 'val' 'test' )

declare -a nparams=( "100000" )
declare -a datavsfile=( 'w0wa_takahashi_nobaryon_dvs_' )
declare -a paramfile=( 'w0wa_params_' )
declare -a failfile=( 'w0wa_params_failed_' )
declare -a yaml=( 'w0wa_takahashi_nobaryon_cs_CNN.yaml' )

declare -i NY=3
declare -i AID=${SLURM_ARRAY_TASK_ID:-0}
declare -i i=$(( AID / NY ))
declare -i j=$(( AID % NY ))
(( i < ${#datavsfile[@]} )) || exit 0 # padding: skip unused slots

"${CONDA_PREFIX}"/bin/mpirun -n ${SLURM_NTASKS} --oversubscribe --mca pml ^ucx \
  --mca btl vader,tcp,self --bind-to core:overload-allowed --rank-by slot \
  --map-by slot:pe=${OMP_NUM_THREADS} \
  python external_modules/code/emulators/emultrf/emultraining/dataset_generator_lensing.py \
    --root projects/roman_real/  \
    --fileroot emulators/w0wa_nla_halofit_cosmic_shear_cnn/ \
    --nparams "${nparams[$i]}" \
    --temp "${temp[$j]}" \
    --yaml "${yaml[$i]}" \
    --datavsfile "${datavsfile[$i]}""${mode[$j]}" \
    --paramfile  "${paramfile[$i]}""${mode[$j]}" \
    --failfile  "${failfile[$i]}""${mode[$j]}" \
    --chain 0 \
    --maxcorr 0.15
