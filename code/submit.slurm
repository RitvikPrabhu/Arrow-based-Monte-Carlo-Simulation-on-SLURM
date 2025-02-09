#!/bin/bash
#SBATCH --job-name=11NodeRun
#SBATCH --nodes=11
#SBATCH --ntasks=11
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH -A HPCBIGDATA2
#SBATCH -p p100_normal_q
#SBATCH --exclusive

module reset
module load Anaconda3
source activate ArrowBLAST

srun ./launch.sh

