#!/usr/bin/env bash
# Download core O*NET 30.2 tab-delimited files for task/occupation analysis.
# License: O*NET 30.2 Database — see https://www.onetcenter.org/license_db.html (CC BY 4.0 with US DOL attribution requirements).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data/raw/onet"
BASE="https://www.onetcenter.org/dl_files/database/db_30_2_text"

mkdir -p "${DEST}"

fetch() {
  local name="$1"
  local enc base
  base="${name%.txt}"
  base="${base// /_}"
  enc=$(python3 -c "import urllib.parse as u,sys; print(u.quote(sys.argv[1]))" "$name")
  echo "Fetching ${name} ..."
  curl -fL "${BASE}/${enc}" -o "${DEST}/${base}.txt"
}

fetch "Task Statements.txt"
fetch "Task Ratings.txt"
fetch "Occupation Data.txt"
fetch "Job Zones.txt"
fetch "Job Zone Reference.txt"
fetch "Scales Reference.txt"

echo "Done. Files in ${DEST}"
