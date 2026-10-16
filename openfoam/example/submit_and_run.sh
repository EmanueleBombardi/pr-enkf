#!/bin/bash
# submit_and_update.sh - Iterative EnKF workflow script
# EnKF_multiVariable_multiDomain_clustering — SLURM version
#
# EnKFDict conventions:
#   iterationCount  N;          current cycle index (1 = fresh start)
#   nCycles         M;          total number of EnKF analysis cycles
#   updateInterval  4;          same interval for all sims, or
#   updateInterval  (4 8 4);    one per sim, in sim1, sim2, ... order
#                               (must be on a single line)
#
# In cycle c, sim k runs until iteration c * updateInterval[k]; then ONE
# joint EnKF analysis is performed across all sims (shared parameters).
#
# IMPORTANT: Run inside a persistent session to avoid login node timeout.
# Usage: tmux new -s enkf; bash submit_and_update.sh

# ---------------------------------------------------------------
# Directory settings
# ---------------------------------------------------------------
MASTER_DIR=$(realpath "$(pwd)")
ENKF_DICT="$MASTER_DIR/system/EnKFDict"
RUN_MEMBERS_SCRIPT="$MASTER_DIR/run_all_members.sh"
COMPLETION_MARKER="$MASTER_DIR/all_members_complete.txt"

die()
{
    echo "ERROR: $*" >&2
    exit 1
}

# ---------------------------------------------------------------
# Read a single scalar value from the top-level EnKFDict
# (trailing // comments are stripped)
# ---------------------------------------------------------------
get_enkf_value()
{
    local param="$1"
    grep -E "^\s*$param\s+" "$ENKF_DICT" | head -1 \
        | sed 's://.*$::' | awk '{print $2}' | sed 's/;//'
}

# ---------------------------------------------------------------
# Detect simulation directories (sim1, sim2, ...), numerically sorted
# — same ordering as detectSimulationDirectories() in the C++ class
# ---------------------------------------------------------------
detect_simulation_directories()
{
    local sim_dirs=()
    for dir in "$MASTER_DIR"/sim*; do
        if [ -d "$dir" ] && [[ $(basename "$dir") =~ ^sim[0-9]+$ ]]; then
            sim_dirs+=("$dir")
        fi
    done
    IFS=$'\n' sorted=($(sort -V <<<"${sim_dirs[*]}")); unset IFS
    echo "${sorted[@]}"
}

# ---------------------------------------------------------------
# Read updateInterval (scalar or single-line list) into INTERVALS[],
# one entry per sim. Scalar -> replicated. Wrong length -> error.
# Must be called after SIM_DIRS is set.
# ---------------------------------------------------------------
read_update_intervals()
{
    local line value v
    line=$(grep -E "^\s*updateInterval\s+" "$ENKF_DICT" | head -1 | sed 's://.*$::')
    [ -n "$line" ] || die "updateInterval not found in $ENKF_DICT"

    value=${line#*updateInterval}

    if [[ "$value" == *"("* && "$value" != *")"* ]]; then
        die "updateInterval list must be on a single line, e.g.  updateInterval (4 8 4);"
    fi

    value=$(tr -d '();' <<< "$value")

    local raw=()
    read -r -a raw <<< "$value"

    [ ${#raw[@]} -gt 0 ] || die "updateInterval is empty in $ENKF_DICT"

    for v in "${raw[@]}"; do
        [[ "$v" =~ ^[0-9]+$ ]] && [ "$v" -gt 0 ] \
            || die "updateInterval entry '$v' is not a positive integer"
    done

    INTERVALS=()
    if [ ${#raw[@]} -eq 1 ]; then
        for ((s=0; s<${#SIM_DIRS[@]}; s++)); do
            INTERVALS+=("${raw[0]}")
        done
    elif [ ${#raw[@]} -eq ${#SIM_DIRS[@]} ]; then
        INTERVALS=("${raw[@]}")
    else
        die "updateInterval has ${#raw[@]} entries but ${#SIM_DIRS[@]} sim directories were found. Must be 1 or nSims."
    fi
}

# ---------------------------------------------------------------
# Update endTime in controlDict for all members in all sims:
#   endTime(sim k) = cycle * INTERVALS[k]
# Fails if any controlDict is missing or was not updated.
# ---------------------------------------------------------------
update_end_time()
{
    local cycle=$1
    local s i sim_dir sim_name ckpt ctrl

    for s in "${!SIM_DIRS[@]}"; do
        sim_dir="${SIM_DIRS[$s]}"
        sim_name=$(basename "$sim_dir")
        ckpt=$(( cycle * INTERVALS[s] ))

        echo "  $sim_name: endTime -> $ckpt (interval ${INTERVALS[$s]})"

        for ((i=0; i<MEMBER_COUNT; i++)); do
            ctrl="$sim_dir/EnKFmembers/member_$i/system/controlDict"
            [ -f "$ctrl" ] || die "controlDict not found: $ctrl"

            sed -i "s/endTime\s\+[0-9]\+;/endTime         $ckpt;/" "$ctrl"

            grep -qE "^\s*endTime\s+$ckpt;" "$ctrl" \
                || die "endTime not updated to $ckpt in $ctrl (unexpected controlDict format?)"
        done
    done
}

# ---------------------------------------------------------------
# Latest numeric time directory in a case (empty if none)
# ---------------------------------------------------------------
latest_time_dir()
{
    find "$1" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null \
        | grep -E '^[0-9]+(\.[0-9]+)?$' | sort -g | tail -1
}

# ---------------------------------------------------------------
# Verify every member actually advanced during this cycle and was
# post-processed. Guards against the C++ resolveTimeDirectory()
# silently renaming the PREVIOUS checkpoint to the new expected one
# when a member crashed.
#   latest == checkpoint                -> OK
#   prev checkpoint < latest < ckpt     -> stopped early (e.g.
#                                          residualControl), warning
#   latest <= prev checkpoint / missing -> crashed, abort
# ---------------------------------------------------------------
verify_checkpoints()
{
    local cycle=$1
    local s i sim_name ckpt prev m lt
    local failed=0 early=0

    for s in "${!SIM_DIRS[@]}"; do
        sim_name=$(basename "${SIM_DIRS[$s]}")
        ckpt=$(( cycle * INTERVALS[s] ))
        prev=$(( (cycle - 1) * INTERVALS[s] ))

        for ((i=0; i<MEMBER_COUNT; i++)); do
            m="${SIM_DIRS[$s]}/EnKFmembers/member_$i"
            lt=$(latest_time_dir "$m")

            if [ -z "$lt" ] || ! awk -v a="$lt" -v b="$prev" 'BEGIN{exit !(a>b)}'; then
                echo "  FAILED: $sim_name/member_$i  latest time '${lt:-none}' <= previous checkpoint $prev"
                failed=$((failed + 1))
                continue
            fi

            if [ ! -d "$m/postProcessing/sampleDict/$lt" ]; then
                echo "  FAILED: $sim_name/member_$i  no postProcessing/sampleDict/$lt"
                failed=$((failed + 1))
                continue
            fi

            if [ "$lt" != "$ckpt" ]; then
                echo "  NOTE:   $sim_name/member_$i  stopped at $lt instead of $ckpt (will be renamed by the update)"
                early=$((early + 1))
            fi
        done
    done

    [ "$failed" -eq 0 ] || die "$failed member(s) did not complete cycle $cycle. Check log.<solver> and log.postProcess in the members listed above."
    [ "$early" -eq 0 ] || echo "  $early member(s) stopped before their checkpoint."
    echo "  All members completed cycle $cycle."
}

# ---------------------------------------------------------------
# Update iterationCount (= cycle index) in the top-level EnKFDict
# ---------------------------------------------------------------
update_iteration_count()
{
    local new_count=$1
    sed -i "s/iterationCount\s\+[0-9]\+;/iterationCount       $new_count;/" "$ENKF_DICT"
    [ "$(get_enkf_value iterationCount)" = "$new_count" ] \
        || die "Failed to update iterationCount to $new_count in $ENKF_DICT"
    echo "Updated iterationCount to $new_count in $ENKF_DICT"
}

# ---------------------------------------------------------------
# Tag the update solver's log file in sub_update.sh with the cycle:
#   ... > log.<name>          ->  ... > log.<name>_cycle<N>
#   ... > log.<name>_cycle<M> ->  ... > log.<name>_cycle<N>
# Only touches lines containing "Update" that redirect to "log.*",
# so it works whatever the executable suffix (e.g. _v10).
# ---------------------------------------------------------------
update_log_file()
{
    local cycle=$1
    local sub_update="$MASTER_DIR/sub_update.sh"

    if [ ! -f "$sub_update" ]; then
        echo "WARNING: sub_update.sh not found at $sub_update"
        return
    fi

    sed -i -E "/Update/ s/(>\s*log\.[A-Za-z0-9_.]+)_cycle[0-9]+/\1/" "$sub_update"
    sed -i -E "/Update/ s/(>\s*log\.[A-Za-z0-9_.]+)/\1_cycle${cycle}/" "$sub_update"

    if grep -q "_cycle${cycle}" "$sub_update"; then
        echo "Tagged update log in sub_update.sh with _cycle${cycle}"
    else
        echo "WARNING: could not tag log name in sub_update.sh (no '> log.*' on a line containing 'Update')."
    fi
}

# ---------------------------------------------------------------
# SLURM helpers
# ---------------------------------------------------------------
submit_job()
{
    local script=$1 job_id
    job_id=$(sbatch "$script" | awk '{print $NF}')
    [ -n "$job_id" ] || die "Failed to submit $script"
    echo "$job_id"
}

wait_for_job()
{
    local job_id=$1
    echo "Waiting for SLURM job $job_id to complete..."
    while squeue -j "$job_id" 2>/dev/null | grep -q "$job_id"; do
        sleep 30
    done
    echo "Job $job_id left the queue."
}

check_job_state()
{
    local job_id=$1 label=$2 state
    sleep 10   # let sacct catch up
    state=$(sacct -j "$job_id" -X -n -o State | head -1 | tr -d ' ')
    [ "$state" = "COMPLETED" ] || die "$label job $job_id ended with state '$state'."
    echo "$label job $job_id: COMPLETED"
}

# ===============================================================
# Sanity checks
# ===============================================================
for f in "$ENKF_DICT" "$MASTER_DIR/sub.sh" "$MASTER_DIR/sub_update.sh"; do
    [ -f "$f" ] || die "Required file not found: $f"
done

# ===============================================================
# Detect simulation directories
# ===============================================================
SIM_DIRS=($(detect_simulation_directories))
[ ${#SIM_DIRS[@]} -gt 0 ] || die "No simulation directories matching 'sim[0-9]+' in $MASTER_DIR"

# ===============================================================
# Read configuration from EnKFDict
# ===============================================================
ITERATION_COUNT=$(get_enkf_value "iterationCount")
MEMBER_COUNT=$(get_enkf_value "nEnsembles")
N_CYCLES=$(get_enkf_value "nCycles")

if [ -z "$N_CYCLES" ]; then
    if [ -n "$(get_enkf_value endEnKF)" ]; then
        die "EnKFDict uses obsolete 'endEnKF'. Replace it with 'nCycles' (old equivalent: endEnKF / updateInterval)."
    fi
    die "Could not read nCycles from $ENKF_DICT"
fi

for var in ITERATION_COUNT MEMBER_COUNT N_CYCLES; do
    [[ "${!var}" =~ ^[0-9]+$ ]] || die "Could not read a valid integer $var from $ENKF_DICT (got '${!var}')"
done

[ "$ITERATION_COUNT" -ge 1 ] || die "iterationCount must be >= 1 (got $ITERATION_COUNT)"

read_update_intervals

echo ""
echo "EnKF_multiVariable_multiDomain_clustering configuration:"
echo "  EnKFDict:          $ENKF_DICT"
echo "  Simulations:       ${#SIM_DIRS[@]}"
for s in "${!SIM_DIRS[@]}"; do
    echo "    $(basename "${SIM_DIRS[$s]}"): updateInterval ${INTERVALS[$s]}  (final checkpoint $(( N_CYCLES * INTERVALS[s] )))"
done
echo "  Members per sim:   $MEMBER_COUNT"
echo "  Total members:     $(( ${#SIM_DIRS[@]} * MEMBER_COUNT ))"
echo "  nCycles:           $N_CYCLES"
echo "  Starting cycle:    $ITERATION_COUNT"
echo ""

if [ "$ITERATION_COUNT" -gt "$N_CYCLES" ]; then
    echo "iterationCount ($ITERATION_COUNT) > nCycles ($N_CYCLES): nothing to do."
    exit 0
fi

# ===============================================================
# Step 1: Initial setup job (sub.sh) — fresh start only
# ===============================================================
if [ "$ITERATION_COUNT" -eq 1 ]; then
    echo "Fresh start (iterationCount = 1): submitting setup job (sub.sh)..."
    SETUP_JOB_ID=$(submit_job "$MASTER_DIR/sub.sh") || exit 1
    echo "Submitted setup job with ID: $SETUP_JOB_ID"
    wait_for_job "$SETUP_JOB_ID"
    check_job_state "$SETUP_JOB_ID" "Setup"
    echo ""
else
    echo "Resuming at cycle $ITERATION_COUNT: skipping setup (sub.sh)."
    echo ""
fi

[ -f "$RUN_MEMBERS_SCRIPT" ] || die "run_all_members.sh not found at $RUN_MEMBERS_SCRIPT (did setup run?)"

# ===============================================================
# Main EnKF cycle loop
#   iterationCount == cycle being run; incremented only after the
#   analysis of that cycle has completed successfully.
# ===============================================================
CYCLE=$ITERATION_COUNT

while [ "$CYCLE" -le "$N_CYCLES" ]; do

    echo "==============================================================="
    echo "EnKF cycle $CYCLE / $N_CYCLES"
    echo "==============================================================="

    # 1. Per-sim endTime for all members
    update_end_time "$CYCLE"

    # 2. Remove stale completion marker
    [ -f "$COMPLETION_MARKER" ] && rm "$COMPLETION_MARKER" && echo "Removed stale completion marker."

    # 3. Run all member simulations (one SLURM job; internally one srun
    #    step per member, sized per sim by numberOfSubdomains)
    echo "Submitting member simulations..."
    MEMBERS_JOB_ID=$(submit_job "$RUN_MEMBERS_SCRIPT") || exit 1
    echo "Submitted members job with ID: $MEMBERS_JOB_ID"
    wait_for_job "$MEMBERS_JOB_ID"
    check_job_state "$MEMBERS_JOB_ID" "Members"

    if [ -f "$COMPLETION_MARKER" ]; then
        cat "$COMPLETION_MARKER"
    else
        echo "WARNING: completion marker missing although the job COMPLETED."
    fi

    # 4. Check every member really advanced and was post-processed
    echo "Verifying member outputs for cycle $CYCLE..."
    verify_checkpoints "$CYCLE"

    # 5. Tag the update log with this cycle
    update_log_file "$CYCLE"

    # 6. EnKF analysis (reads iterationCount = $CYCLE from EnKFDict)
    echo "Submitting EnKF update job (sub_update.sh) for cycle $CYCLE..."
    UPDATE_JOB_ID=$(submit_job "$MASTER_DIR/sub_update.sh") || exit 1
    echo "Submitted update job with ID: $UPDATE_JOB_ID"
    wait_for_job "$UPDATE_JOB_ID"
    check_job_state "$UPDATE_JOB_ID" "Update"

    # 7. Advance to the next cycle
    CYCLE=$((CYCLE + 1))
    update_iteration_count "$CYCLE"

    echo "Cycle $((CYCLE - 1)) finished successfully."
    echo "---------------------------------------------------------------"
    echo ""

done

echo "==============================================================="
echo "All $N_CYCLES EnKF cycles completed successfully."
echo "Final iterationCount: $(get_enkf_value iterationCount)  (= nCycles + 1)"
echo "==============================================================="