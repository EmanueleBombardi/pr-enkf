#!/bin/bash
#SBATCH --job-name=name
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --account=urban

source $HOME/of10.sh

EnKF_FoamMultiVariable_multiDomain_clustering_Update_v10 > log.EnKF_FoamMultiVariable_multiDomain_clustering_Update_v10