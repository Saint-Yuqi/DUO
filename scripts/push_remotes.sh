#!/usr/bin/env bash
# Push the current branch to BOTH remotes (GitHub = origin, Codeup = codeup). Usage: bash scripts/push_remotes.sh [branch]
set -euo pipefail
cd "$(dirname "$0")/.."
GIT=${GIT:-git}
[[ -x /Library/Developer/CommandLineTools/usr/bin/git ]] && GIT=/Library/Developer/CommandLineTools/usr/bin/git   # macOS: Xcode git may be licence-blocked
BR=${1:-$($GIT rev-parse --abbrev-ref HEAD)}
for r in origin codeup; do
  echo "== push $BR -> $r"; $GIT push -u "$r" "$BR"
done
