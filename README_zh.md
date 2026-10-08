# DUO 仓库 — 团队工作流（中文）

这个仓库维护三样东西：**Duo 动作头本体**（`duo/`，可插到任何 backbone）、**每个 backbone 里实际跑过的原码**（`duo/adapters/`，逐字拷贝 + 出处），
以及**实验台账**（`ledger/*.yaml` → `docs/index.html` 网页 + `paper/tables/*.tex` 论文表）。两个远端同步：GitHub `origin` 与 Codeup `codeup`。

## 日常
```bash
git pull                                    # 本机 ~/QuicData/DUO；Mac 用 /Library/Developer/CommandLineTools/usr/bin/git
# 改代码 / 改台账
python ledger/build.py                      # 校验所有引用、重建 docs/ 与 paper/tables/
bash scripts/push_remotes.sh                # 同时推 GitHub 和 Codeup
```
台账网页：`docs/index.html`（仓库里随时可下载；GitHub 仓库 Settings → Pages → `main` / `docs` 打开后有公网地址），
也发布为 claude.ai artifact：https://claude.ai/artifact/LYUcUNwvGWM3XQeWZgQHSR（私有，分享给同事后他们才能打开；每次 build 后重新发布同一文件即可更新）。改表格只改 YAML，不要直接改 HTML。

## 记一个新 run
1. `ledger/runs.yaml` 复制一条，填：backbone、`variant`（头版本，见 `variants.yaml`）、`protocol`、数据、训练配方、机器、
   `code`（集群源码路径 + 本仓库拷贝路径）、`weights`（路径 / 是否持久 / 是否已删 / 备份）、`eval`、`results`、`notes`。
2. 评测完：`python eval/robotwin/summarize.py <run_root> --run-id <id>` 直接打印 `results:` 块并写 `ledger/per_task/<id>.json`。
3. `python ledger/build.py` 不报错 → commit → push。表格对应关系在 `tables.yaml`（哪个格子用哪个 run / 外部数）。

## 换 backbone
按 `duo/adapters/README.md` 的两种模式：有逐 token 的 DiT/flow block 列表的宿主（FLOWER、DINO-DiT、OpenWAM）→ 两份流 + `CrossArmSet.apply`；
MoT 一次联合注意力的宿主（Fast-WAM、OpenWAM native）→ 动作 token 翻倍 + `block_cross_stream_attention` 遮掉跨流块 + `CrossArm`。
动作切分用 `Partition`（arm_semantic / mixed / eef_openwam80 / joint），预训练单流权重用 `warm_start_from_single_stream` 切列/切行初始化。
策略服务统一走 `eval/robotwin/policy_server.py` 的协议，评测、汇总、记账三步不变。

## AutoDL
```bash
source /etc/network_turbo; git clone https://github.com/Saint-Yuqi/DUO.git /root/DUO && bash /root/DUO/scripts/setup_autodl.sh
```
脚本做：学术加速、`/root/autodl-tmp` 目录、duo 库 venv + CPU 测试、装 ossutil 并提示配 `cn-beijing`（漏 region 全 403），
然后 `scripts/pull_data.sh flower_cache|robotwin_assets|robotwin_clean|ckpt <name> <dest>` 从 OSS 拉数据。
仿真评测环境（RoboTwin 2.0 + SAPIEN）和各 backbone 的训练环境不在脚本里，按 `eval/robotwin/README.md` 与 `duo/adapters/README.md` 装。
`eval/robotwin/xpolicylab_duo/` 是官方 XPolicyLab 路线的桥接（**未测**），先用 `--test_num 3` 跑一个任务核对观测键再信结果。

## 协议纪律
只有 `robotwin2_official`（clean2500 训、50×2×100 评）能和榜单比；`zzm500`、`fastwam_subset10` 是内部协议，网页和表格都分开放。
权重在 PAI-DSW 的 `/tmp`、`/root` 上的都是 `persistent: false`，容器重启就没了——台账里标红的先备份到 OSS。
