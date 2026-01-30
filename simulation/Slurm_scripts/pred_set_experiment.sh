#!/bin/bash
#SBATCH -A stat-users
#SBATCH --partition=econ     # other options: gomez-cascade-extra,gomez-cascade,gomez,ses,schuler,randeria,batch,batch7,stat,stat-skylake,stat-cascade,stat-sapphire
#SBATCH --qos=normal
#SBATCH -t 10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-26
#SBATCH -o /home/nandy.15/Research/Experiments_revision/Slurm_Scripts/Calibrate_pred_Set_mod_1/Output_Messages/slurm_out_%A_%a_%x_%j_%t.out
#SBATCH -e /home/nandy.15/Research/Experiments_revision/Slurm_Scripts/Calibrate_pred_Set_mod_1/Error_Messages/slurm_err_%A_%a_%x_%j_%t.err

# This slurm script runs the prediction set calibration experiments
# using the parameters defined in the parameter grid below. Running this script will reproduce the results of Section 3.2 (Table 2) in the paper.

# --- Environment ---
module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
# Use your Unity env; change if needed
conda activate all_amp_projects || { echo "conda activate failed"; exit 1; }

# Prefer env’s libs (helps with rpy2/pyzmq on Unity)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export R_HOME="$CONDA_PREFIX/lib/R"
export R_LIBS_USER="$CONDA_PREFIX/lib/R/library"

# Threading consistent with cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "=== Job ${SLURM_ARRAY_TASK_ID} on $(hostname) ==="
echo "Python  : $(which python)"
echo "Conda   : $CONDA_PREFIX"
python -V

# --- Project root ---
cd /home/nandy.15/Research/Experiments_revision || { echo "cd failed"; exit 1; }

# --- Parameter grids (same as your RCC script) ---
N_values=(2000 3000 4000)
lam_values=(5 7 9)
rho_values=(0.8 0.9 1)

num_N=${#N_values[@]}
num_lam=${#lam_values[@]}
num_rho=${#rho_values[@]}

idx=$SLURM_ARRAY_TASK_ID
N_idx=$(( idx / (num_lam * num_rho) ))
rem=$(( idx % (num_lam * num_rho) ))
lam_idx=$(( rem / num_rho ))
rho_idx=$(( rem % num_rho ))

N=${N_values[$N_idx]}
lam=${lam_values[$lam_idx]}
rho=${rho_values[$rho_idx]}

echo "Running coverage calibration with N=$N, lambda=$lam, rho=$rho"

# --- Run script ---
# Update PY_SCRIPT if your path/name differs
PY_SCRIPT="/home/nandy.15/Research/Experiments_revision/Python_Scripts/pred_set.py"

"$CONDA_PREFIX/bin/python" "$PY_SCRIPT" "$N" "$lam" "$rho"
status=$?

echo "Exit status: $status"
exit $status
