#!/usr/bin/env bash
# Pull the shared data/assets from Quic's OSS onto a new machine (AutoDL etc.). Needs ossutil configured (scripts/setup_autodl.sh).
# Usage: bash scripts/pull_data.sh <what> <dest_dir>
#   what = flower_cache   clean2500 224² frame cache used by the FLOWER / DINO-DiT heads   (oss://quic-research/yangyq/data/cache/flower_robotwin_clean2500/)
#          robotwin_assets background_texture / objects / embodiments zips for the simulator (oss://quic-pre-train/data/RoboTwin2.0/)
#          robotwin_clean  the 50 aloha-agilex clean_50 demo zips (23.8 GB)                 (oss://quic-pre-train/data/RoboTwin2.0/dataset/<task>/aloha-agilex_clean_50.zip)
#          ckpt <name>     a published checkpoint package                                   (oss://quic-research/yangyq/checkpoints/<name>/)
# OSS region is cn-beijing; the pre-train bucket may answer 403 to ossutil for some keys (it did on the cluster) — then copy
# from the /mnt/oss mount on a cluster machine instead.
set -euo pipefail
OSSUTIL=${OSSUTIL:-ossutil}
what=${1:?what}; dest=${2:?dest}; mkdir -p "$dest"
case "$what" in
  flower_cache) $OSSUTIL cp -r oss://quic-research/yangyq/data/cache/flower_robotwin_clean2500/ "$dest/flower_robotwin_clean2500/" --update;;
  robotwin_assets) for z in background_texture objects embodiments; do $OSSUTIL cp oss://quic-pre-train/data/RoboTwin2.0/$z.zip "$dest/" --update; done;;
  robotwin_clean) while read -r t; do [[ -z "$t" ]] && continue; $OSSUTIL cp "oss://quic-pre-train/data/RoboTwin2.0/dataset/$t/aloha-agilex_clean_50.zip" "$dest/$t/" --update; done < "$(dirname "$0")/../eval/robotwin/tasks50.txt";;
  ckpt) $OSSUTIL cp -r "oss://quic-research/yangyq/checkpoints/${3:?name}/" "$dest/${3}/" --update;;
  *) echo "unknown: $what"; exit 2;;
esac
