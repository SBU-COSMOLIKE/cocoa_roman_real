#!/bin/bash

#SBATCH --job-name=PR
#SBATCH --output=PR-%A_%a.out
#SBATCH -e PR-%A_%a.out.err
#SBATCH --nodes=1
#SBATCH --ntasks=75
#SBATCH --cpus-per-task=1
#SBATCH --partition=hbm-long-96core
#SBATCH --time=24:00:00

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
  sleep $(( 5 + SLURM_ARRAY_TASK_ID*25 )) # help avoid different scripts running start_cocoa.sh simultaneously
                                          # clash can still happen when one script runs start_cocoa.sh, the other
                                          # script is reading a file (start_cocoa.sh invokes stop_cocoa.sh)
                                          # which delete links and ROOTDIR so the running script can find file/likelihood
  conda activate cocoa
  source start_cocoa.sh
fi

export OMP_PROC_BIND=close
export OMP_NUM_THREADS=1

# the parameters are 
# As_1e9    ns        H0      omegab  omegam  
# roman_DZ_S1 roman_DZ_S2 roman_DZ_S3 roman_DZ_S4 
# roman_DZ_S5 roman_DZ_S6 roman_DZ_S7 roman_DZ_S8
# roman_A1_1 roman_A1_2 
# roman_M1 roman_M2 roman_M3 roman_M4 
# roman_M5 roman_M6 roman_M7 roman_M8
declare -a factor1=( "3" "1.0" "1.0" "1.0" "3" 
                     "3" "3" "3" "3" 
                     "3" "3" "3" "3"
                     "3" "3" 
                     "3" "3" "3" "3" 
                     "3" "3" "3" "3 ")
# the parameters are 
# As_1e9    ns        H0      omegab  omegam w0pwa w 
# roman_DZ_S1 roman_DZ_S2 roman_DZ_S3 roman_DZ_S4 
# roman_DZ_S5 roman_DZ_S6 roman_DZ_S7 roman_DZ_S8
# roman_A1_1 roman_A1_2 
# roman_M1 roman_M2 roman_M3 roman_M4 
# roman_M5 roman_M6 roman_M7 roman_M8
declare -a factor2=( "3" "1.0" "1.0" "1.0" "3" "1.0" "1.0" 
                     "3" "3" "3" "3" 
                     "3" "3" "3" "3"
                     "3" "3" 
                     "3" "3" "3" "3" 
                     "3" "3" "3" "3 ")
declare -a grid=( factor1 factor2 )

declare -i NY=100 # so from 0-99 is all about PROFILE1, 100-199 all about PROFILE2
declare -i i=$(( $SLURM_ARRAY_TASK_ID / NY + 1 ))
declare -i j=$(( $SLURM_ARRAY_TASK_ID % NY ))
(( i < ${#grid[@]} )) || exit 0 # padding: skip unused slots
declare -n row="${grid[$i]}"
(( j < ${#row[@]} )) || exit 0  # padding: skip unused slots
declare factor="${row[$j]}"

mpirun -n ${SLURM_NTASKS} --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
  --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
  python ./projects/roman_real/EXAMPLE_EMUL_PROFILE1.py \
    --root ./projects/roman_real/ --cov "chains/EXAMPLE_EMUL_MCMC${i}.covmat" \
    --outroot "EXAMPLE_EMUL_PROFILE$i" --factor "$factor" \
    --nstw 400 --numpts 15 --profile "${j}" \
    --minfile="./projects/roman_real/chains/EXAMPLE_EMUL_MIN${i}.txt"
