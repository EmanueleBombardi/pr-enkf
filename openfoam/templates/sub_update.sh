#!/bin/bash
#SBATCH --job-name=masterEnKF_Update
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --partition=<partition>
#SBATCH --account=<account>

source <path to OpenFOAM 10 environment script>

# The EnKF binary must be the last command, on a line containing "Update".
# The driver sets the log name to log.EnKF_Update_cycleN at every cycle.
EnKF_FoamMultiVariable_multiDomain_clustering_Update_v10 > log.EnKF_Update_cycle1
