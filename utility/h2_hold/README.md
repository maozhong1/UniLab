# H2 SONIC 定姿保持（hold）测试工具

## 用途

部署时 motion clip 播完后，deploy 会一直保持最后一帧作为参考。这组工具在 MuJoCo 里复现这个行为：
给每个 clip 末尾追加 N 秒的最后一帧（速度清零），用确定性 actor（CPU、1 env、无早停、无观测噪声、
无 DR）跑完整个 clip + hold 段，逐步记录关节力矩、PD 误差、CoM / CoP / 足底接触，然后统计
hold 段是否稳定（会不会前倾、抬脚、踝关节打满、膝盖顶到限位）。

所有命令在仓库根目录执行，**必须用 `uv run --no-sync`**（普通 `uv run` 会替换 XPU 版 torch）。
rollout 只用 CPU（`CUDA_VISIBLE_DEVICES=`，`OMP_NUM_THREADS=2`），可以和训练并行。

## 文件

| 文件 | 作用 |
|------|------|
| `make_hold_clips.py` | 生成 `*_hold.npz`（clip + hold 段）和 `manifest.json`（clip -> 原始帧数） |
| `run_hold.py` | 单个 clip 的 hold rollout，输出逐步记录的 npz；可选 `--seed/--pose-noise/--vel-noise` 扰动 |
| `analyze.py` | `load()` / `hold_stats()` 可 import；CLI 打印一个结果目录的完整报告（可带变体目录） |
| `compare.py` | 两个结果目录逐 clip 对比关键指标（`A>B`） |
| `trace.py` | 单个结果在 clip 结束前后的逐步轨迹 |
| `trend.py` | 训练日志关键指标按 iteration 分箱的趋势表 |
| `make_xml_variants.py` | 生成接触模型 A/B 用的 xml 变体（dt2 / solref / box），仓库 xml 不改 |
| `quick_hold.sh` | 4 个代表性 clip 并行快测，带 baseline 时调用 compare |
| `run_suite.sh` | manifest 中全部 clip，可选每个 clip K 个扰动 seed |
| `wait_ckpt.sh` | 等某个 run 目录出现 `model_<iter>.pt`，或训练进程退出 |

## 典型流程

```bash
# 1. 生成 hold clip（只需一次；默认源目录为 gear_sonic_deploy/h2/npz_new）
uv run --no-sync python utility/h2_hold/make_hold_clips.py --dst /tmp/h2hold/clips \
    --hold-s 6 --stand src/unilab/assets/motions/h2/stand.npz --stand-hold-s 3

# 2a. 快测（stand / macarena / washing / walk_forward_loop），可选与 baseline 结果目录对比
CLIP_DIR=/tmp/h2hold/clips utility/h2_hold/quick_hold.sh <ckpt> /tmp/h2hold/q_new [/tmp/h2hold/q_old]

# 2b. 全量 + 每个 clip 3 个扰动 rollout（默认 POSE_NOISE=0.02, VEL_NOISE=0.05）
CLIP_DIR=/tmp/h2hold/clips utility/h2_hold/run_suite.sh <ckpt> /tmp/h2hold/suite_new 3

# 3. 对比 / 报告 / 单步轨迹
uv run --no-sync python utility/h2_hold/compare.py /tmp/h2hold/q_old /tmp/h2hold/q_new old new
uv run --no-sync python utility/h2_hold/analyze.py /tmp/h2hold/suite_new /tmp/h2hold/suite_new/seeds
uv run --no-sync python utility/h2_hold/trace.py /tmp/h2hold/q_new/stand.npz -25 200 5

# 单个 rollout（扰动示例）
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 uv run --no-sync python utility/h2_hold/run_hold.py \
    /tmp/h2hold/clips/stand_hold.npz /tmp/h2hold/pert/stand__s1.npz --ckpt <ckpt> \
    --seed 1 --pose-noise 0.02 --vel-noise 0.05

# 训练日志趋势（每 10 iter 一行）
uv run --no-sync python utility/h2_hold/trend.py <train.log> 10

# 等 checkpoint 再测（进程匹配只读 pgrep，不会杀进程）
utility/h2_hold/wait_ckpt.sh logs/rsl_rl_ppo/H2SonicMotionTracking/<run> 11000 "train_rsl_rl.py.*h2" \
    && utility/h2_hold/quick_hold.sh logs/rsl_rl_ppo/H2SonicMotionTracking/<run>/model_11000.pt /tmp/h2hold/q11000

# 接触模型 A/B：生成 xml 变体后用 --scene（dt2 还需 --simdt 0.002），结果命名 <clip>__<tag>.npz
uv run --no-sync python utility/h2_hold/make_xml_variants.py /tmp/h2hold/xml
EXTRA="--scene /tmp/h2hold/xml/xml_box/scene_flat.xml" utility/h2_hold/quick_hold.sh <ckpt> /tmp/h2hold/q_box
```

`analyze.py <dir> <variant_dir>` 会把 `<variant_dir>/<clip>__<tag>.npz` 与 `<dir>/<clip>.npz`
并排打印，扰动 seed（`run_suite.sh` 的 `seeds/`）和 xml 变体都用这个格式。

## 指标含义（默认统计 hold 段）

- **pitch err**：骨盆 pitch 与参考最后一帧的差（deg，mean / 绝对值最大处），正值表示比参考更前倾。
- **CoM ahead of mid-sole**：CoM 在朝向坐标系下相对两脚足底中心中点的前向偏移（cm），越大越靠脚尖。
- **CoP toe margin（CoP-toe）**：按足底力加权的 CoP 到脚尖边缘的距离（cm）；接近 0 表示压在脚尖上，
  足长约 21 cm，中足约 10 cm。
- **ankle pitch / roll saturation（apSat / arSat）**：踝关节力矩达到 `actuatorfrcrange` 98% 的步数占比（%）。
- **apErr / apAct / arAct**：踝关节 PD 误差（rad）和策略原始动作，持续偏大说明在“顶”。
- **knee lock %**：膝关节角 < -0.08 rad（贴近 -0.087 的过伸限位）的步数占比。
- **lifts**：任一只脚法向力从 ≥20 N 掉到 <20 N 的次数（抬脚 / 迈步次数）。
- **foot disp**：hold 段内脚相对 hold 起点的最大水平位移（cm），> 几 cm 即发生了挪步。
- 其他：drift（骨盆水平漂移）、fall（骨盆高度 < 0.5 m）、trip（超过训练终止阈值）、
  接触细节（前半脚掌受力比例、脚跟离地比例、外侧边受力、穿透、接触抖动、滑移）。

## 通过标准

一个 checkpoint 在 hold 测试中算“稳”，需要所有 clip 同时满足：

1. **不抬脚**：hold 段 lifts = 0，foot disp 接近 0；
2. **|pitch err| ≤ 3°**（mean 和 max 都看）；
3. **CoP 在中足**：CoP-toe 明显离开脚尖（不是 0~3 cm），CoM 不明显前移；
4. **没有关节顶在限位**：踝 pitch / roll 饱和 ≈ 0%，knee lock ≈ 0%；
5. **多次扰动 rollout 都满足**：用 `run_suite.sh ... K`（K ≥ 3）跑扰动 seed，不只看确定性的一次。
