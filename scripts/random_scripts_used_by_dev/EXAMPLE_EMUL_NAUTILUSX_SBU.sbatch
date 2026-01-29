#!/bin/bash

#SBATCH --job-name=NAU
#SBATCH --output=NAU-%A_%a.out
#SBATCH -e NAU-%A_%a.out.err
#SBATCH --nodes=1
#SBATCH --ntasks=94
#SBATCH --cpus-per-task=1
#SBATCH --partition=hbm-long-96core
#SBATCH --time=48:00:00

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR

source /home/u8/vivianmiranda/miniconda/etc/profile.d/conda.sh
conda activate cocoa
source start_cocoa.sh

export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

$CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} --oversubscribe \
  --mca btl vader,tcp,self --bind-to core:overload-allowed \
  --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run \
  python -m mpi4py.futures \
  ./projects/roman_real/EXAMPLE_EMUL_NAUTILUS${SLURM_ARRAY_TASK_ID}.py \
    --root ./projects/roman_real/ \
    --outroot "EXAMPLE_EMUL_NAUTILUS${SLURM_ARRAY_TASK_ID}" \
    --maxfeval 750000 --nlive 2048 --neff 15000 --flive 0.01 --nnetworks 5