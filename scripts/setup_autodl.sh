#!/usr/bin/env bash
# One-shot bootstrap of an AutoDL instance for Duo evaluations. Idempotent. Run as root inside the instance:
#   source /etc/network_turbo 2>/dev/null; git clone https://github.com/Saint-Yuqi/DUO.git /root/DUO && bash /root/DUO/scripts/setup_autodl.sh
# What it does:   1. academic acceleration (GitHub/HF reachable)  2. data disk layout under /root/autodl-tmp
#                 3. duo library venv + tests                      4. ossutil (asks for AK/SK once; region cn-beijing)
#                 5. prints the next steps per backbone (eval environments are NOT installed here — see duo/adapters/README.md)
set -euo pipefail
DATA=${DATA:-/root/autodl-tmp}
mkdir -p "$DATA"/{data,ckpt,runs,envs}
source /etc/network_turbo 2>/dev/null || true
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
cd "$(dirname "$0")/.."
MIRROR=tuna bash scripts/setup_env.sh "$DATA/envs/duo"
# ossutil
if ! command -v ossutil >/dev/null; then
  curl -fsSL https://gosspublic.alicdn.com/ossutil/install.sh | bash || echo "ossutil install failed — install manually: https://help.aliyun.com/zh/oss/developer-reference/install-ossutil"
fi
if [[ ! -f ~/.ossutilconfig ]]; then
  cat <<EOT
Configure ossutil once (AK/SK from Quic ops; region MUST be set or everything is 403):
  ossutil config -e https://oss-cn-beijing.aliyuncs.com --region cn-beijing -i <AK> -k <SK>
then:  bash scripts/pull_data.sh flower_cache $DATA/data      # 224² clean2500 cache for FLOWER / DINO-DiT heads
       bash scripts/pull_data.sh robotwin_assets $DATA/data   # simulator assets
EOT
fi
cat <<EOT
Next: eval needs the RoboTwin 2.0 simulator (SAPIEN 3, curobo) + the backbone's own environment:
  git clone https://github.com/RoboTwin-Platform/RoboTwin $DATA/RoboTwin && cd $DATA/RoboTwin && bash scripts/_install.sh
  unzip the asset zips into $DATA/RoboTwin/assets ; cp -r $PWD/eval/robotwin/xpolicylab_duo XPolicyLab/policy/Duo
  backbone envs: duo/adapters/README.md ; results -> python eval/robotwin/summarize.py <root> --run-id <id> ; python ledger/build.py
EOT
