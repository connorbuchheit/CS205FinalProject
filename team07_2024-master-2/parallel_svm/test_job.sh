#!/usr/bin/env bash
#SBATCH --job-name=ipm_test
#SBATCH --output=ipm_test%j.out
#SBATCH --error=ipm_test%j.err
#SBATCH --nodes=2
#SBATCH --mem=16384
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=3
#SBATCH --time=01:30:00

spack load python@3.11.6%gcc@7.3.1 gcc@13.2.0 openmpi@4.1.6

srun -n4 -c3 --mpi=pmix ./ICFApp -k 0 Data/cover_type_x_10000_tr.csv Data/cover_type_y_10000_tr.csv Data/cover_type_x_10000_te.csv Data/cover_type_y_10000_te.csv Data/cover_type_x_10000_predict.out

# return the exit code of srun above
exit $?