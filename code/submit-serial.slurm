#!/bin/bash
#SBATCH --job-name=1NodeRun
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH -A HPCBIGDATA2
#SBATCH -p p100_normal_q
#SBATCH --exclusive

module reset
module load Anaconda3
source activate ArrowBLAST

python baseline.py --verbose

