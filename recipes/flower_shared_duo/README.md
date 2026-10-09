# FLOWER + 共享权重 Duo（把 π0.5-Duo 的做法搬到 FLOWER 上）

集群目录：`/mnt/workspace/yangyq/flower_sharedduo_robotwin`（quic5000/5001 都能看到，CPFS 持久）。本目录是它在 DUO 仓库里的副本。

## 为什么要跑它
之前的 FLOWER-Duo（`flower_duodit_robotwin`，68.30 / 26.62）是**两份独立权重**的 FlowBlock 流 + 零初始化 tanh 增益的跨臂注意力。
10-10 读它的 `last.pt`：8 个跨臂增益 |tanh(g)| ≤ 0.044，通道没打开，实际是两个各管一条臂的 FLOWER 头，参数还多了 283M（721M）。
π0.5-Duo（80.36 / 46.36）是**一份共享的动作专家**同时处理两臂 token，门开时两臂直接互相注意，没有要“学着打开”的增益。
这个 run 让 FLOWER 也按 π0.5-Duo 的方式来，回答“共享权重的 Duo 在 FLOWER 上是多少分”。

## 模型（`flower_sharedduo.py`，`FlowerSharedDuo`）
| | 旧 FlowBlock-Duo | **本 run（共享 Duo）** | π0.5-Duo（官方 run） |
|---|---|---|---|
| 动作专家 | 2 份 FlowBlock（右 = 左的 deepcopy） | **1 份**作者预训练 FlowBlock（12×1024） | 1 份预训练 expert |
| token 序列 | 每流 50 个动作 token | `[s_L, a_L×50, s_R, a_R×50]`，RoPE 位置两臂都是 0..50 | `[s_L, a_L×50, s_R, a_R×50]` |
| 每臂输入/输出投影 | 新建 MLP | 预训练 16 维编码器 fc1 按臂切列、解码器按臂切行 | 预训练投影按臂切 |
| 本臂状态 | 进 adaLN | 一个状态 token（新 Linear，零初始化）+ 存在位 | 一个状态 token（零初始化） |
| 跨臂通道 | CrossArm 模块 + tanh 增益（训完≈0） | **自注意力掩码**：门开 = 两臂互相可见；无额外参数 | 后缀注意力掩码，门开 |
| 注意力因果 | 每流内因果 | 两臂共用时间轴的因果：时刻 p 只看时刻 ≤ p 的 token（FLOWER 预训练就是因果的） | 后缀内双向 |
| 观测 | Florence 跑两次（head + 本臂腕） | Florence 跑**一次**：head + 左腕 + 右腕 + 提示 + 指令，两臂共享 | VLM 前缀共享 |
| 参数 | 721.2M | **440.0M**（可训练 416.8M），≈ 单流 FLOWER | ≈ π0.5 |

`--gate closed` 可以跑“共享权重但两臂互不可见”的消融（不是这次要跑的）。

## 配方（与旧 FlowBlock-Duo 完全一致，只换了模型）
Florence-2-base + 作者预训练头 `360000_model_weights.pt`（240/264 个 DiT 张量加载，`cond_linear`/`cond_norm` 因 base/large 维度不同重新初始化，和旧 run 一样）；
数据 clean2500 帧缓存 `/mnt/workspace/yangyq/data/flower_robotwin_clean2500`（50 任务 × 50 clean，224²，三相机）；
4 卡 × batch 2 × accum 4 = 全局 32；最多 60k 步，15k 步后验证连续 6 次（每 2000 步一次）不降就停；head lr 1e-4、VLM lr 2e-5、AdamW wd 0.01、warmup 500 + cosine 到 10%；bf16；
评测用验证集最好的 `training/best.pt`（与旧 run 同口径）。

## 怎么跑
```bash
cd /mnt/workspace/yangyq/flower_sharedduo_robotwin
# 1) 训练：4 张卡，约 19 h（旧 run 57.7k 步 1146 min；本模型只跑一次 Florence、一次 DiT，预计不慢于它）
GPUS=0,1,2,3 nohup bash train.sh > train.nohup 2>&1 &
#    中断后续训：RESUME=1 GPUS=0,1,2,3 nohup bash train.sh > train.nohup 2>&1 &
#    看进度：tail -f training/train.log ；验证曲线 training/val.jsonl ；训完会写 training/DONE
# 2) 评测：官方协议 50 任务 × clean/randomized × 100，4 张卡 × 6 个仿真 worker，可断点续跑
EVAL_GPUS=4,5,6,7 nohup bash eval.sh > eval.nohup 2>&1 &
#    结果：eval_results/<task>/<setting>/result.json ；全部完成写 eval_summary.json 和 finished
```
卡数不是 4 张也行：`train.sh` 按 `GPUS` 个数自动设 accum，保持全局 batch 32（1/2/4/8 卡）。

## 已验证（2026-10-10，CPU，真实 Florence-2-base + 真实预训练头 + 真实 clean2500 数据）
`smoke_cpu.py` 全部通过：预训练头加载报告与旧 run 一致；每臂投影是预训练层的精确切片；三相机一次过 Florence（197 个上下文 token）；
门关时左臂速度场与右臂动作/状态无关、门开时有关；跨两臂的时间因果正确；损失与梯度有限，新模块、DiT、VLM 都有梯度；
`predict` 输出 (B, 50, 14)；`policy_server.py` 能加载 checkpoint 并按 RoboTwin 协议返回 (50, 14) 动作。
`train.py` CPU 跑 2 步 + 验证 + 存盘 + `--resume` 续到 4 步，正常；`run_eval_fast.py --check` 正常。
**没在 GPU 上跑过**（当时 5000 八卡都在跑 OpenWAM 评测）：显存预计与旧 run 相当或更低（440M 对 721M，batch 2/卡）。

## 跑完以后
```bash
python /path/to/DUO/eval/robotwin/summarize.py /mnt/workspace/yangyq/flower_sharedduo_robotwin --run-id flower_sharedduo_clean2500
```
把输出的 `results:` 块填进 DUO 仓库 `ledger/runs.yaml` 里 `flower_sharedduo_clean2500` 那条，`python ledger/build.py`，推两个远端。
