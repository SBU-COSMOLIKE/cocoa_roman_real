#!/bin/bash

#SBATCH --job-name=RY1_MC1
#SBATCH --output=RY1_MC1-%A_%a.out
#SBATCH -e RY1_MC1-%A_%a.out.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=7
#SBATCH --partition=extended-28core
#SBATCH --time=135:00:00

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

cd $SLURM_SUBMIT_DIR
conda activate cocoapy38
source start_cocoa

export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

$CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} --oversubscribe \
  --mca btl vader,tcp,self --bind-to core:overload-allowed \
  --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run \
  ./projects/roman_real/EXAMPLE_MCMC${SLURM_ARRAY_TASK_ID}.yaml -r