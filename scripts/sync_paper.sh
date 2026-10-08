#!/usr/bin/env bash
# Pull the Overleaf mirror (github.com/pikonguwu/Duo-WAM, synced from Overleaf by the paper owner) and show what the ledger
# would change in its tables. Usage: bash scripts/sync_paper.sh [path_to_Duo-WAM_clone]
set -euo pipefail
cd "$(dirname "$0")/.."
PAPER=${1:-$HOME/Duo-WAM}
[[ -d "$PAPER/.git" ]] || git clone https://github.com/pikonguwu/Duo-WAM.git "$PAPER"
git -C "$PAPER" pull --ff-only
python ledger/build.py
for f in paper/tables/*.tex; do
  b=$(basename "$f"); echo "=== $b (ledger) vs Overleaf table/$b"; diff <(sed 's/[[:space:]]\+/ /g' "$PAPER/table/$b" 2>/dev/null) <(sed 's/[[:space:]]\+/ /g' "$f") || true
done
echo "To update Overleaf: copy paper/tables/<id>.tex over $PAPER/table/<id>.tex, commit there, and let the paper owner sync."
