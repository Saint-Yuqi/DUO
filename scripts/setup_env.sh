#!/usr/bin/env bash
# Minimal environment for the duo library + ledger tools (CPU is enough for tests and the ledger).
# Backbone-specific environments (FLOWER / lerobot-π0.5 / OpenWAM / Fast-WAM) are documented in duo/adapters/README.md.
# Usage: bash scripts/setup_env.sh [venv_dir]      MIRROR=aliyun|tuna|none (default: auto — aliyun inside the Quic cluster)
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${1:-.venv}
MIRROR=${MIRROR:-auto}
if [[ "$MIRROR" == auto ]]; then
  if [[ -d /mnt/cpfs ]]; then MIRROR=aliyun; elif [[ -d /root/autodl-tmp ]]; then MIRROR=tuna; else MIRROR=none; fi
fi
case "$MIRROR" in
  aliyun) export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple PIP_TRUSTED_HOST=mirrors.aliyun.com;;
  tuna)   export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple;;
esac
python3 -m venv "$VENV"
"$VENV/bin/pip" install -U pip
"$VENV/bin/pip" install -e ".[dev]"
"$VENV/bin/python" tests/test_duo_cpu.py
echo "ok: source $VENV/bin/activate"
