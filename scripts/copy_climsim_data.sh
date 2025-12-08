#!/bin/bash
#
# move_climsim_parallel_tar_hydra.sbatch
#
# Parallel tar-stream copy tuned to your Hydra/Slurm defaults:
# - timeout_min: 1440  => --time=24:00:00
# - mem_gb: 64         => --mem=64G
# - cpus_per_task: adjustable (default here 20; change to match dataset.general_dataset_config.num_workers if needed)
# - partition: standard
# - qos: high
# - account: iecdt
#
# If you want to integrate with Hydra's submitit, set HYDRA_SWEEP_DIR so SUBMITIT_FOLDER is correct:
#   export HYDRA_SWEEP_DIR=/path/to/hydra/sweep/dir
#   sbatch ...
#
#SBATCH --job-name=move-climsim-fast
#SBATCH --output=move-climsim-fast.%j.log
#SBATCH --error=move-climsim-fast.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20               # adjust to dataset.general_dataset_config.num_workers if required
#SBATCH --mem=64G
#SBATCH --time=24:00:00                  # 1440 minutes (from timeout_min)
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --account=iecdt

set -euo pipefail

# -------------------------
# Editable runtime variables
# -------------------------
# Source and destination - edit if needed
SRC="/gws/nopw/j04/iecdt/bstanleyclamp/ClimSim_lowres/train"
DST="/work/scratch-pw5/bradlesc/climsim/lowres"

# Logging directory
LOGDIR="${HOME}/move_climsim_logs"

# Hydra/Submitit integration (optional)
# If running under Hydra submitit, set HYDRA_SWEEP_DIR before submitting so this path matches
HYDRA_SWEEP_DIR="${HYDRA_SWEEP_DIR:-}"
SUBMITIT_FOLDER="${HYDRA_SWEEP_DIR:+${HYDRA_SWEEP_DIR}/.submitit/%j}"

# Parallelism: use all CPUs allocated to the job but cap at 20 if you want to limit.
# SLURM_CPUS_PER_TASK will be set by Slurm; fallback to 20 if not set.
PARALLEL_JOBS="${SLURM_CPUS_PER_TASK:-20}"
# If you want a hard upper bound (since you mentioned "up to 20 cpus"), you can uncomment the next line:
# PARALLEL_JOBS=$(python - <<'PY' ; import os; print(min(int(os.environ.get('SLURM_CPUS_PER_TASK',20)),20)) ; PY

echo "Starting job on $(hostname) at $(date)"
echo "SRC=${SRC}"
echo "DST=${DST}"
echo "LOGDIR=${LOGDIR}"
echo "PARALLEL_JOBS=${PARALLEL_JOBS}"
[ -n "${SUBMITIT_FOLDER}" ] && echo "SUBMITIT_FOLDER=${SUBMITIT_FOLDER}"

mkdir -p "${LOGDIR}"
mkdir -p "${DST}"

# Tools check (we proceed even if some are missing; tar alone is enough)
for cmd in tar pv parallel rsync; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "WARNING: $cmd not found in PATH. Some features (progress, parallel) may be limited."
  fi
done

cd "${SRC}"

# Build list of top-level entries (null-delimited for safety)
find . -maxdepth 1 -mindepth 1 -print0 > "${LOGDIR}/entries.list0"

# Per-entry copy function
copy_entry() {
  # Input: null-delimited path like "./subdir" or "./file"
  entry="$1"
  # Normalize name for log file
  clean=$(printf "%s" "$entry" | sed 's#^\./##' | tr '/' '_')
  logbase="${LOGDIR}/copy_${clean}"
  mkdir -p "$(dirname "${logbase}")" 2>/dev/null || true

  echo "==== START COPY ${clean} at $(date)" > "${logbase}.out"
  # Tar-stream the entry; use pv for progress if available.
  if command -v pv >/dev/null 2>&1; then
    tar -cpf - "$entry" 2>>"${logbase}.out" | pv -p -t -e -r 2>>"${logbase}.out" | (cd "${DST}" && tar -xpf -) >>"${logbase}.out" 2>&1
    rc=${PIPESTATUS[0]:-0}
  else
    tar -cpf - "$entry" 2>>"${logbase}.out" | (cd "${DST}" && tar -xpf -) >>"${logbase}.out" 2>&1
    rc=${PIPESTATUS[0]:-0}
  fi
  echo "==== END COPY ${clean} at $(date) rc=${rc}" >> "${logbase}.out"
  return "${rc}"
}

export -f copy_entry
export SRC DST LOGDIR

echo "Launching parallel copy with ${PARALLEL_JOBS} jobs at $(date)"
# Use GNU parallel if available, otherwise fall back to xargs-style loop
if command -v parallel >/dev/null 2>&1; then
  # GNU parallel with null-delimited input
  parallel --jobs "${PARALLEL_JOBS}" --null --joblog "${LOGDIR}/parallel_joblog.txt" \
    bash -lc 'copy_entry "$0"' :::: "${LOGDIR}/entries.list0"
else
  echo "GNU parallel not available; running serially (this will be slower)."
  # Serial fallback: iterate over null-delimited list
  while IFS= read -r -d '' ent; do
    copy_entry "$ent" || echo "Entry $ent failed (rc=$?)"
  done < "${LOGDIR}/entries.list0"
fi

echo "Parallel copy finished at $(date)"
echo "Per-entry logs are in ${LOGDIR} (files named copy_<entry>.out)."

# -------------------------
# Verification (optional)
# -------------------------
# We default to a dry-run checksum rsync; this is I/O heavy (reads all copied data).
# To skip heavy verification, comment out the rsync block below and run a lighter check later.
if command -v rsync >/dev/null 2>&1; then
  RSYNC_VERIFY_LOG="${LOGDIR}/rsync_verify.$(date +%Y%m%d_%H%M%S).log"
  echo "Starting rsync verification (checksum, dry-run). This may take time..."
  rsync -a --checksum --dry-run --delete --itemize-changes "${SRC}/" "${DST}/" 2>&1 | tee "${RSYNC_VERIFY_LOG}"
  echo "Rsync verification dry-run completed; see ${RSYNC_VERIFY_LOG}"
else
  echo "rsync not found; skipping automated verification. Please run:"
  echo "  rsync -a --checksum --dry-run ${SRC}/ ${DST}/"
fi

echo "Job completed at $(date)"
