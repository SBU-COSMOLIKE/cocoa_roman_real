#!/bin/bash

#SBATCH --job-name=PR
#SBATCH --output=PR-%A_%a.out
#SBATCH --error PR-%A_%a.out.err
#SBATCH --nodes=1
#SBATCH --ntasks=94
#SBATCH --ntasks-per-node=94
#SBATCH --ntasks-per-socket=47
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR
declare -i NY=100 # so from 0-99 is all about PROFILE1, 100-199 all about PROFILE2
if [ -z "${ROOTDIR}" ]; then
  declare -i tmp=$(( $SLURM_ARRAY_TASK_ID % NY ))
  sleep $(( 5 + $tmp*15 )) # help avoid different scripts running start_cocoa.sh simultaneously
                           # clash can still happen when one script runs start_cocoa.sh, the other
                           # script is reading a file (start_cocoa.sh invokes stop_cocoa.sh)
                           # which delete links and ROOTDIR so the running script can find file/likelihood
  conda activate cocoa
  source start_cocoa.sh
fi

export OMP_PROC_BIND=close
export OMP_NUM_THREADS=1
ulimit -u 2000000 # require line when nmpi is high

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
declare -a nstws=( "400" "450" )
declare -i AID=${SLURM_ARRAY_TASK_ID:-0}
declare -i i=$(( AID / NY ))
declare -i j=$(( AID % NY ))
(( i < ${#grid[@]} )) || exit 0 # padding: skip unused slots
declare -n row="${grid[$i]}"
declare nstw="${nstws[$i]}"
(( j < ${#row[@]} )) || exit 0  # padding: skip unused slots
declare factor="${row[$j]}"

"${CONDA_PREFIX}"/bin/mpirun -n ${SLURM_NTASKS} --oversubscribe --mca pml ^ucx \
  --mca btl vader,tcp,self --bind-to core:overload-allowed --rank-by slot \
  --map-by slot:pe=${OMP_NUM_THREADS} python \
  ./projects/roman_real/EXAMPLE_EMUL_PROFILE$((i+1)).py \
    --root ./projects/roman_real/ --cov "chains/EXAMPLE_EMUL_MCMC$((i+1)).covmat" \
    --outroot "EXAMPLE_EMUL_PROFILE$((i+1))" --factor "$factor" \
    --nstw $nstw --numpts 15 --profile "${j}" \
    --minfile="./projects/roman_real/chains/EXAMPLE_EMUL_MIN$((i+1)).txt"
