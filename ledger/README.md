# ledger — single source of truth for every Duo experiment

| file | holds |
|---|---|
| `runs.yaml` | every run we trained/evaluated: backbone, head version, protocol, data, recipe, machine, code paths (cluster source + repo copy), weights (path / persistent? / deleted? / backup), eval state, results, notes |
| `variants.yaml` | every Duo head version in code, chronologically, with streams / coupling / gate / init and what changed from the previous one |
| `external.yaml` | numbers we did not produce (leaderboard snapshots, colleagues' baselines) with source + snapshot date |
| `tables.yaml` | the paper tables (mirroring the Overleaf `table/*.tex`), each cell pointing at a run or external id |
| `protocols.yaml` | what train/eval protocol a number was obtained under; never mix protocols in one table |
| `per_task/<run>.json` | per-task success rates (written by `eval/robotwin/summarize.py`) |
| `build.py`, `template.html` | generator → `docs/index.html`, `docs/artifact.html`, `docs/ledger.json`, `paper/tables/*.tex` |

Workflow: edit YAML → `python ledger/build.py` (validates every cross-reference and fails loudly) → commit → `bash scripts/push_remotes.sh`.
Conventions: success rates in %, `overall = mean(clean, random)`; `*_n` keeps successes/episodes; dates ISO; a run is never deleted
(`status: superseded`); anything on `/tmp` or `/root` of a PAI-DSW container is `persistent: false`.
