#!/usr/bin/env bash
# Download ESCO v1.1.1 English classification + relations CSVs.
#
# Source: tabiya-tech/tabiya-esco-datasets-and-tools, a public GitHub mirror of
# the ESCO v1.1.1 download packages. ESCO itself is CC BY 4.0
# (https://esco.ec.europa.eu/en/about-esco/escopedia/escopedia/terms-use).
#
# Pinned to v1.1.1 to match the frozen position-paper artefacts; bump when we
# move to a newer ESCO release.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${ROOT}/data/raw/esco"
VERSION="v1.1.1"
BASE="https://github.com/tabiya-tech/tabiya-esco-datasets-and-tools/raw/main/datasets/esco/${VERSION}"

mkdir -p "${DEST}"

CLASSIFICATION_ZIP="ESCO dataset - ${VERSION} - classification - en - csv.zip"
RELATIONS_ZIP="ESCO dataset - ${VERSION} - classification -  - csv.zip"

fetch() {
  local relpath="$1"
  local outfile="$2"
  local url_path
  url_path=$(python3 -c "import urllib.parse as u,sys; print(u.quote(sys.argv[1]))" "$relpath")
  echo "Fetching ${relpath} -> ${outfile}"
  curl -fL "${BASE}/${url_path}" -o "${DEST}/${outfile}"
}

fetch "classification/${CLASSIFICATION_ZIP}" "classification_en.zip"
fetch "relations/${RELATIONS_ZIP}"          "relations.zip"

unzip -q -o "${DEST}/classification_en.zip" -d "${DEST}"
unzip -q -o "${DEST}/relations.zip"         -d "${DEST}"

echo "Done. Files in ${DEST}"
ls -1 "${DEST}"
