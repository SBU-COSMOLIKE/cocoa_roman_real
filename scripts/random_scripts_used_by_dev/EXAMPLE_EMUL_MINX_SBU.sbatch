#!/bin/bash

#SBATCH --job-name=MIN
#SBATCH --output=MIN-%A_%a.out
#SBATCH -e MIN-%A_%a.out.err
#SBATCH --nodes=1
#SBATCH --ntasks=75
#SBATCH --cpus-per-task=1
#SBATCH --partition=hbm-medium-96core
#SBATCH --time=12:00:00

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
if [ -z "${ROOTDIR}" ]; then
  sleep $(( 5 + SLURM_ARRAY_TASK_ID*15 )) # help avoid different scripts running start_cocoa.sh simultaneously
                                          # clash can still happen when one script runs start_cocoa.sh, the other
                                          # script is reading a file (start_cocoa.sh invokes stop_cocoa.sh)
                                          # which delete links and ROOTDIR so the running script can find file/likelihood
  conda activate cocoa
  source start_cocoa.sh
fi

export OMP_PROC_BIND=close
export OMP_NUM_THREADS=1

mpirun -n ${SLURM_NTASKS} --oversubscribe --mca btl vader,tcp,self \
  --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} \
  python ./projects/roman_real/EXAMPLE_EMUL_MININIMIZE${SLURM_ARRAY_TASK_ID}.py \
  --nstw 160 --root ./projects/roman_real/ \
  --outroot "example_minimize${SLURM_ARRAY_TASK_ID}"