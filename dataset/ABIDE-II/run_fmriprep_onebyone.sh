#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-dataset/ABIDE-II/raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-dataset/ABIDE-II/fmriprep}"
TMP_ROOT="${FMRIPREP_TMP_ROOT:-dataset/ABIDE-II/.fmriprep_tmp}"
IMAGE="${FMRIPREP_IMAGE:-nipreps/fmriprep:25.2.5}"
LICENSE_FILE="${FS_LICENSE:-dataset/ABIDE-II/raw/license.txt}"
SPACE="${FMRIPREP_SPACE:-MNI152NLin6Asym}"
MAX_PARALLEL="${FMRIPREP_MAX_PARALLEL:-2}"
NPROCS="${FMRIPREP_NPROCS:-32}"
OMP_THREADS="${FMRIPREP_OMP_THREADS:-4}"
MEM_MB="${FMRIPREP_MEM_MB:-90000}"
ONLY_SUBJECT="${1:-}"
PROCESSED_COUNT=0
ELIGIBLE_TOTAL=0
ELAPSED_TOTAL=0
ELIGIBLE_SITES=()
ELIGIBLE_SUBJECT_ROOTS=()

mkdir -p "$OUTPUT_ROOT" "$TMP_ROOT"
if [[ ! -d "$RAW_ROOT" ]]; then
  echo "Missing raw root: $RAW_ROOT" >&2
  exit 2
fi
if [[ ! -f "$LICENSE_FILE" ]]; then
  echo "Missing FreeSurfer license: $LICENSE_FILE" >&2
  exit 2
fi

command -v docker >/dev/null || {
  echo "Docker is required" >&2
  exit 2
}

cleanup() {
  if [[ -n "${SUBJECT_TMP:-}" && -d "${SUBJECT_TMP:-}" ]]; then
    rm -rf "$SUBJECT_TMP"
  fi
}
trap cleanup EXIT INT TERM

format_duration() {
  local seconds="$1"
  printf '%02dh:%02dm:%02ds' "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

show_progress() {
  local subject="$1"
  local started_at="$2"
  local now elapsed eta average
  now=$(date +%s)
  elapsed=$((now - started_at))
  if [[ "$PROCESSED_COUNT" -gt 0 ]]; then
    average=$((ELAPSED_TOTAL / PROCESSED_COUNT))
    eta=$((average * (ELIGIBLE_TOTAL - PROCESSED_COUNT)))
    printf '\rProgress: [%d/%d] %s | current %s | ETA %s' \
      "$PROCESSED_COUNT" "$ELIGIBLE_TOTAL" "$subject" \
      "$(format_duration "$elapsed")" "$(format_duration "$eta")"
  else
    printf '\rProgress: [0/%d] %s | current %s | ETA calculating' \
      "$ELIGIBLE_TOTAL" "$subject" "$(format_duration "$elapsed")"
  fi
}

copy_selected_outputs() {
  local source_root="$1"
  local destination_root="$2"
  local copied=0
  while IFS= read -r -d '' source_file; do
    local relative_path="${source_file#"$source_root"/}"
    local destination_file="$destination_root/$relative_path"
    mkdir -p "$(dirname "$destination_file")"
    cp -p "$source_file" "$destination_file"
    copied=$((copied + 1))
  done < <(
    find "$source_root" -type f \( \
      -name "*space-${SPACE}_desc-preproc_bold.nii.gz" -o \
      -name "*desc-confounds_timeseries.tsv" -o \
      -name "*desc-confounds_timeseries.json" -o \
      -name "*space-${SPACE}_desc-brain_mask.nii.gz" -o \
      -name "*_bold.json" \
    \) -print0
  )
  if [[ "$copied" -eq 0 ]]; then
    echo "No selected fMRIPrep outputs found under $source_root" >&2
    return 1
  fi
  echo "  retained files: $copied"
}

for site_root in "$RAW_ROOT"/ABIDEII-*; do
  [[ -d "$site_root" ]] || continue
  for subject_root in "$site_root"/sub-*; do
    [[ -d "$subject_root" ]] || continue
    subject="$(basename "$subject_root")"
    if [[ -n "$ONLY_SUBJECT" && "$subject" != "$ONLY_SUBJECT" ]]; then
      continue
    fi
    bold_count=$(find "$subject_root" -type f -name '*_bold.nii.gz' | wc -l)
    t1_count=$(find "$subject_root" -type f -name '*_T1w.nii.gz' | wc -l)
    if [[ "$bold_count" -eq 1 && "$t1_count" -eq 1 ]]; then
      ELIGIBLE_SITES+=("$site_root")
      ELIGIBLE_SUBJECT_ROOTS+=("$subject_root")
      ELIGIBLE_TOTAL=$((ELIGIBLE_TOTAL + 1))
    fi
  done
done

if [[ "$ELIGIBLE_TOTAL" -eq 0 ]]; then
  echo "No subjects with exactly one BOLD and one T1w found." >&2
  exit 1
fi
echo "Subjects to process: $ELIGIBLE_TOTAL"

run_subject() {
  local site_root="$1"
  local subject_root="$2"
  local subject="$(basename "$subject_root")"
  local label="${subject#sub-}"
  local site="$(basename "$site_root")"
  local site_id="${site#ABIDEII-}"
  local metadata_root="$(dirname "$RAW_ROOT")/$site"
  local bold_count
  local t1_count
  bold_count=$(find "$subject_root" -type f -name '*_bold.nii.gz' | wc -l)
  t1_count=$(find "$subject_root" -type f -name '*_T1w.nii.gz' | wc -l)

  if [[ "$bold_count" -ne 1 || "$t1_count" -ne 1 ]]; then
    return 0
  fi

  local final_bold
  final_bold=$(find "$OUTPUT_ROOT/$subject" -type f -name "*space-${SPACE}_desc-preproc_bold.nii.gz" 2>/dev/null | head -1 || true)
  if [[ -n "$final_bold" ]]; then
    PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
    return 0
  fi

  SUBJECT_TMP=$(mktemp -d "$TMP_ROOT/subject.XXXXXX")
  local bids_tmp="$SUBJECT_TMP/bids"
  local output_tmp="$SUBJECT_TMP/fmriprep"
  local work_tmp="$SUBJECT_TMP/work"
  mkdir -p "$bids_tmp" "$output_tmp" "$work_tmp"

  ln -s "/raw/$site/$subject" "$bids_tmp/$subject"
  for metadata_file in dataset_description.json participants.tsv task-rest_bold.json T1w.json; do
    if [[ -f "$metadata_root/$metadata_file" ]]; then
      cp -p "$metadata_root/$metadata_file" "$bids_tmp/$metadata_file"
    elif [[ -f "$site_root/$metadata_file" ]]; then
      cp -p "$site_root/$metadata_file" "$bids_tmp/$metadata_file"
    fi
  done
  if [[ ! -f "$bids_tmp/dataset_description.json" ]]; then
    printf '{"Name":"ABIDE-II","BIDSVersion":"1.8.0","DatasetType":"raw"}\n' \
      > "$bids_tmp/dataset_description.json"
  fi

  local log_file="$SUBJECT_TMP/fmriprep.log"
  local started_at
  local docker_pid
  local docker_status
  started_at=$(date +%s)
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$(realpath "$bids_tmp"):/data:ro" \
    -v "$(realpath "$RAW_ROOT"):/raw:ro" \
    -v "$(realpath "$output_tmp"):/out" \
    -v "$(realpath "$work_tmp"):/work" \
    -v "$(realpath "$LICENSE_FILE"):/opt/freesurfer/license.txt:ro" \
    "$IMAGE" \
    /data /out participant \
    --participant-label "$label" \
    --fs-license-file /opt/freesurfer/license.txt \
    --output-spaces "$SPACE" \
    --fs-no-reconall \
    --nprocs "$NPROCS" \
    --omp-nthreads "$OMP_THREADS" \
    --mem-mb "$MEM_MB" \
    --work-dir /work \
    --skip-bids-validation \
    > "$log_file" 2>&1 &
  docker_pid=$!
  while kill -0 "$docker_pid" 2>/dev/null; do
    show_progress "$site_id/$subject" "$started_at"
    sleep 5
  done
  if wait "$docker_pid"; then
    docker_status=0
  else
    docker_status=$?
  fi
  printf '\n'
  if [[ "$docker_status" -ne 0 ]]; then
    echo "[$site_id/$subject] fMRIPrep failed; log: $log_file" >&2
    tail -40 "$log_file" >&2
    return "$docker_status"
  fi

  copy_selected_outputs "$output_tmp" "$OUTPUT_ROOT"
  local finished_at
  finished_at=$(date +%s)
  ELAPSED_TOTAL=$((ELAPSED_TOTAL + finished_at - started_at))
  PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
  echo "[$site_id/$subject] done in $(format_duration "$((finished_at - started_at))"); temporary files deleted"
  rm -rf "$SUBJECT_TMP"
  SUBJECT_TMP=""
}

if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "FMRIPREP_MAX_PARALLEL must be a positive integer: $MAX_PARALLEL" >&2
  exit 2
fi

declare -a ACTIVE_PIDS=()
declare -A ACTIVE_LABELS=()
completed_count=0
next_index=0
failed_count=0
batch_started_at=$(date +%s)

show_parallel_progress() {
  local now elapsed average eta active_count
  now=$(date +%s)
  elapsed=$((now - batch_started_at))
  active_count=${#ACTIVE_PIDS[@]}
  if [[ "$completed_count" -gt 0 ]]; then
    average=$((elapsed / completed_count))
    eta=$((average * (ELIGIBLE_TOTAL - completed_count)))
    printf '\rProgress: [%d/%d] active=%d | ETA %s' \
      "$completed_count" "$ELIGIBLE_TOTAL" "$active_count" "$(format_duration "$eta")"
  else
    printf '\rProgress: [0/%d] active=%d | ETA calculating' \
      "$ELIGIBLE_TOTAL" "$active_count"
  fi
}

while [[ "$next_index" -lt "$ELIGIBLE_TOTAL" || "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
  while [[ "$next_index" -lt "$ELIGIBLE_TOTAL" && "${#ACTIVE_PIDS[@]}" -lt "$MAX_PARALLEL" ]]; do
    site_root="${ELIGIBLE_SITES[$next_index]}"
    subject_root="${ELIGIBLE_SUBJECT_ROOTS[$next_index]}"
    subject="$(basename "$subject_root")"
    site_id="$(basename "$site_root")"
    site_id="${site_id#ABIDEII-}"
    run_subject "$site_root" "$subject_root" &
    worker_pid=$!
    ACTIVE_PIDS+=("$worker_pid")
    ACTIVE_LABELS["$worker_pid"]="$site_id/$subject"
    next_index=$((next_index + 1))
  done

  show_parallel_progress
  sleep 5
  running_pids="$(jobs -pr)"
  remaining_pids=()
  for worker_pid in "${ACTIVE_PIDS[@]}"; do
    if printf '%s\n' "$running_pids" | grep -qx "$worker_pid"; then
      remaining_pids+=("$worker_pid")
      continue
    fi
    if wait "$worker_pid"; then
      worker_status=0
    else
      worker_status=$?
    fi
    completed_count=$((completed_count + 1))
    if [[ "$worker_status" -ne 0 ]]; then
      failed_count=$((failed_count + 1))
      echo "[$(ACTIVE_LABELS[$worker_pid])] failed with status $worker_status" >&2
    fi
    unset 'ACTIVE_LABELS[$worker_pid]'
  done
  ACTIVE_PIDS=("${remaining_pids[@]}")
done
printf '\n'

if [[ "$failed_count" -gt 0 ]]; then
  echo "Finished with $failed_count failed subject(s)." >&2
  exit 1
fi

cat > "$OUTPUT_ROOT/dataset_description.json" <<EOF
{
  "Name": "ABIDE-II fMRIPrep retained derivatives",
  "PipelineDescription": {"Name": "fMRIPrep", "Version": "25.2.5"},
  "OutputSpace": "$SPACE",
  "RetainedFiles": ["preproc_bold", "confounds_timeseries", "bold_json", "brain_mask"]
}
EOF

echo "Finished. Retained derivatives: $OUTPUT_ROOT"
