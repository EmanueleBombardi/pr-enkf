#!/bin/bash
#SBATCH --job-name=masterEnKF_setup
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --partition=<partition>
#SBATCH --account=<account>

source <path to OpenFOAM 10 environment script>

# The EnKF binary must be the last command
EnKF_FoamMultiVariable_multiDomain_clustering_v10 > log.EnKF_setup
