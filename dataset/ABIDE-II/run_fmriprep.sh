#!/usr/bin/env bash
set -euo pipefail

BIDS_ROOT="${1:-ABIDEII/bids}"
OUTPUT_ROOT="${2:-ABIDEII/derivatives}"
WORK_ROOT="${3:-ABIDEII/work}"
IMAGE="${FMRIPREP_IMAGE:-nipreps/fmriprep:25.2.3}"
LICENSE_FILE="${FS_LICENSE:-$HOME/.freesurfer.txt}"

mkdir -p "$OUTPUT_ROOT" "$WORK_ROOT"
if [[ ! -f "$LICENSE_FILE" ]]; then
  echo "Missing FreeSurfer license: $LICENSE_FILE" >&2
  exit 2
fi

docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -v "$(realpath "$BIDS_ROOT"):/data:ro" \
  -v "$(realpath "$OUTPUT_ROOT"):/out" \
  -v "$(realpath "$WORK_ROOT"):/work" \
  -v "$(realpath "$LICENSE_FILE"):/opt/freesurfer/license.txt:ro" \
  "$IMAGE" \
  /data /out participant \
  --fs-license-file /opt/freesurfer/license.txt \
  --output-spaces MNI152NLin6Asym \
  --work-dir /work \
  --skip-bids-validation
