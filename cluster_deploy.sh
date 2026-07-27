#!/bin/bash
# Metacentrum deployment helper
# Syncs the project to the cluster and optionally submits PBS jobs.
#
# Usage:
#   ./cluster_deploy.sh sync              — sync files only
#   ./cluster_deploy.sh submit broad      — sync + submit both stands at given stage
#   ./cluster_deploy.sh submit broad beech — sync + submit beech only
#   ./cluster_deploy.sh submit broad spruce — sync + submit spruce only
#   ./cluster_deploy.sh status            — check running jobs
#   ./cluster_deploy.sh fetch             — download results (logs + calib_res) back

# ---------------------------------------------------------------------------
# CONFIGURE THESE
# ---------------------------------------------------------------------------
META_USER="vstein"                              # your Metacentrum username
META_HOST="${META_USER}@skirit.ics.muni.cz"    # login node (change if needed)
REMOTE_DIR="/storage/plzen1/home/${META_USER}/coupledOpt"
# ---------------------------------------------------------------------------

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

ACTION="${1:-sync}"
STAGE="${2:-broad}"
STANDS="${3:-both}"   # both | beech | spruce

echo "=== Metacentrum deploy: ${ACTION} ==="
echo "Remote: ${META_HOST}:${REMOTE_DIR}"

# ---------------------------------------------------------------------------
# SYNC
# ---------------------------------------------------------------------------
sync_files() {
    echo "--- Syncing project files ---"
    rsync -avz --progress \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        --exclude 'drutes_run_*/' \
        --exclude 'nonconvergent/' \
        --exclude 'figs/' \
        --exclude '.git/' \
        --exclude '*.out' \
        "${LOCAL_DIR}/" \
        "${META_HOST}:${REMOTE_DIR}/"
    echo "Sync done."
    echo ""
    echo "IMPORTANT: The drutes binary (drutes_temp_beech/bin/drutes,"
    echo "drutes_temp_spruce/bin/drutes) must be compiled ON the cluster."
    echo "SSH in and run 'make' inside the DRUtES source tree, then copy"
    echo "the resulting binary into the template dirs before submitting."
}

# ---------------------------------------------------------------------------
# SUBMIT
# ---------------------------------------------------------------------------
submit_jobs() {
    local stage="$1"
    local stands="$2"
    echo "--- Submitting PBS jobs (stage=${stage}, stands=${stands}) ---"

    if [[ "$stands" == "both" || "$stands" == "beech" ]]; then
        ssh "$META_HOST" "cd ${REMOTE_DIR} && qsub -v STAGE=${stage} de_beech.pbs"
        echo "Beech job submitted."
    fi

    if [[ "$stands" == "both" || "$stands" == "spruce" ]]; then
        ssh "$META_HOST" "cd ${REMOTE_DIR} && qsub -v STAGE=${stage} de_spruce.pbs"
        echo "Spruce job submitted."
    fi
}

# ---------------------------------------------------------------------------
# STATUS
# ---------------------------------------------------------------------------
show_status() {
    echo "--- Active jobs (queued / running / held) ---"
    ssh "$META_HOST" "qstat -u ${META_USER}"
    echo ""
    echo "--- Incl. recently finished (-x history) ---"
    # Plain qstat drops jobs the moment they finish or fail; -x shows history.
    # 'S' column: Q=queued R=running F=finished. Exit status via: qstat -xf <jobid>
    ssh "$META_HOST" "qstat -xu ${META_USER}"
}

# ---------------------------------------------------------------------------
# FETCH RESULTS
# ---------------------------------------------------------------------------
fetch_results() {
    echo "--- Fetching results ---"
    rsync -avz \
        "${META_HOST}:${REMOTE_DIR}/de_log_beech.csv" \
        "${META_HOST}:${REMOTE_DIR}/de_log_spruce.csv" \
        "${META_HOST}:${REMOTE_DIR}/finetune_log_beech.csv" \
        "${META_HOST}:${REMOTE_DIR}/finetune_log_spruce.csv" \
        "${META_HOST}:${REMOTE_DIR}/calib_res/" \
        "${META_HOST}:${REMOTE_DIR}/calib_beech.out" \
        "${META_HOST}:${REMOTE_DIR}/calib_spruce.out" \
        "${META_HOST}:${REMOTE_DIR}/morris_results_beech.csv" \
        "${META_HOST}:${REMOTE_DIR}/morris_samples_beech.csv" \
        "${META_HOST}:${REMOTE_DIR}/morris_mustar_sigma_beech.png" \
        "${META_HOST}:${REMOTE_DIR}/morris_ranking_beech.png" \
        "${META_HOST}:${REMOTE_DIR}/morris_beech.out" \
        "${LOCAL_DIR}/" 2>/dev/null || true
    echo "Results fetched."
}

# ---------------------------------------------------------------------------
# DISPATCH
# ---------------------------------------------------------------------------
case "$ACTION" in
    sync)
        sync_files
        ;;
    submit)
        sync_files
        submit_jobs "$STAGE" "$STANDS"
        ;;
    status)
        show_status
        ;;
    fetch)
        fetch_results
        ;;
    *)
        echo "Unknown action: $ACTION"
        echo "Usage: $0 sync | submit <stage> [beech|spruce|both] | status | fetch"
        exit 1
        ;;
esac
