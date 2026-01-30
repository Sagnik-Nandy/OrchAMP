#!/bin/bash
#SBATCH -A stat-users
#SBATCH --partition=econ     #gomez-cascade-extra,gomez-cascade,gomez,ses,schuler,randeria,batch,batch7,stat,stat-skylake,stat-cascade,stat-sapphire
#SBATCH --qos=normal
#SBATCH -t 5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1                 # optional, if your code benefits; adjust as needed
#SBATCH --mem=32G
#SBATCH --array=0-17                    # 2 settings × 3 lambdas × 3 rhos = 18 tasks; run up to 6 at once
#SBATCH -o /home/nandy.15/Research/Experiments_revision/Slurm_Scripts/Effects_of_Data_Integration/Output_Messages/slurm_out_%A_%a_%x_%j_%t.out
#SBATCH -e /home/nandy.15/Research/Experiments_revision/Slurm_Scripts/Effects_of_Data_Integration/Error_Messages/slurm_err_%A_%a_%x_%j_%t.err



# This slurm script is used to run the effect_of_data_integration.py script with different parameters. Running this script will reproduce results from Table 1 in Section 3.1 of the manuscript.

module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate all_amp_projects || { echo "conda activate failed"; exit 1; }

# Prefer env’s libs (often helps with rpy2/pyzmq)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export R_HOME="$CONDA_PREFIX/lib/R"
export R_LIBS_USER="$CONDA_PREFIX/lib/R/library"

# Threading controls to match cpus-per-task (optional but recommended)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "Python  : $(which python)"
echo "Conda   : $CONDA_PREFIX"
python -V

cd /home/nandy.15/Research/Experiments_revision

# --- array math ---
setting_values=("std" "sparse")
lam_values=(2.0 2.4 2.8)
rho_values=(0.8 0.9 1.0)

num_lam=${#lam_values[@]}
num_rho=${#rho_values[@]}
num_settings=${#setting_values[@]}

idx=$SLURM_ARRAY_TASK_ID
per_setting=$(( num_lam * num_rho ))
setting_idx=$(( idx / per_setting ))
remainder=$(( idx % per_setting ))
lam_idx=$(( remainder / num_rho ))
rho_idx=$(( remainder % num_rho ))

setting=${setting_values[$setting_idx]}
lam=${lam_values[$lam_idx]}
rho=${rho_values[$rho_idx]}

echo "Running $setting; lam=$lam; rho=$rho"

"$CONDA_PREFIX/bin/python" \
  "/home/nandy.15/Research/Experiments_revision/Python_Scripts/effect_of_data_integration.py" \
  "$setting" "$lam" "$rho"

