# UniLab 上的 Unitree H2 SuperMimic（世界模型 + BPTT）

本仓库把 SuperMimic 参考实现（`SuperMimic-Reproduced`，基于 mjlab / MuJoCo Warp）移植到了 UniLab，用于验证。机器人是头部固定的 29 自由度 H2，模型文件为 `src/unilab/assets/robots/h2/h2.xml`，与 SuperMimic 使用的 H2 资产完全相同。

## 代码结构

| 模块 | 位置 |
|---|---|
| 环境 `H2SuperMimic`（mujoco backend） | `src/unilab/envs/motion_tracking/h2/supermimic.py` |
| 参考动作库 + 基于回报的重置采样器（NumPy） | `src/unilab/envs/motion_tracking/h2/supermimic_library.py` |
| 采用 SuperMimic 求解器设置的场景（pyramidal cone，impratio 1，迭代 10 / 线搜索 20） | `src/unilab/assets/robots/h2/scene_flat_supermimic.xml` |
| 算法（torch）：FK、状态表示、世界模型、解析头、replay、想象 rollout、训练器、runner | `src/unilab/algos/torch/supermimic/` |
| 训练入口（Hydra 配置根目录 `conf/supermimic`） | `scripts/train_supermimic.py` |
| XPU 后台训练启动脚本（nohup） | `scripts/supermimic/train_h2_xpu_nohup.sh` |
| 评测（checkpoint 或 ONNX）/ ONNX 导出 | `scripts/supermimic/{eval_benchmark,export_onnx}.py` |
| 测试 | `tests/algos/test_supermimic.py` |

环境逐项复现了 SuperMimic 的 mjlab 环境：

- **维度**：obs 930（10 帧 × 93，term-major 历史），goal 680（10 帧 × 68），state 71，上下文 xi 33。
- **动作**：`q0 + (a + eps) * 0.25 * effort / kp - encoder_bias`。
- **奖励**：BeyondMimic 跟踪奖励，乘以 dt。
- **终止与截断**：骨盆高度或倾斜超阈值为终止；clip 结束和 10 s 超时按截断处理。
- **重置**：参考状态初始化，带位姿、速度、关节噪声。
- **目标帧时序**：与 mjlab 相同，重置到帧 `f` 后目标为 `f + 1`。
- **body 运动学**：奖励和终止使用滞后一个物理子步的 body 运动学，与 mjlab 一致。

算法通过环境的 `sm_*` 属性读取 `s`、`xi`、实际执行的动作和目标帧。

未移植的内容：

- 参考实现的 commit `24b8b61`，即 robot-aware 可行性过滤和 `normalize_bin_return`；
- 多 GPU；
- `anchor_obs`。

clip 只按名字过滤：`motion_exclude_keywords: [table, _box_]`。参考实现的 H2 过滤也只去掉了这类 clip。

## 运行

```bash
# CPU 冒烟测试（只检查流程：32 个 env、20 个 clip、小网络）
uv run scripts/train_supermimic.py task=h2_supermimic/mujoco_cpu_smoke

# CUDA 全量训练（与参考 H2 命令一致：4096 env，policy batch 2048，5 万次迭代，每 3000 次存 ckpt）
uv run scripts/train_supermimic.py training.device=cuda algo.supermimic.wm_trainer.compile=default \
    algo.supermimic.policy_trainer.compile=default

# Intel XPU（torch 2.7.0+xpu）：先执行一次 `make sync-xpu`，之后始终用 `uv run --no-sync`。
# mujoco_xpu 配置：4096 env，MLP 矩阵乘用 bf16，开启 torch.compile，数据与超参同参考 H2 训练。
uv run --no-sync scripts/train_supermimic.py task=h2_supermimic/mujoco_xpu

# 评测（评测协议：无噪声、无 DR，摩擦 1，从每个 clip 的第一帧开始）
uv run scripts/supermimic/eval_benchmark.py --checkpoint logs/supermimic/H2SuperMimic/<run>/ckpt_latest.pt \
    --npz-dir ~/work/sonic_vla_infer/bones-seed/selections/upright_nzp_h2_29dof_new \
    --npz-dir ~/work/sonic_vla_infer/bones_seed_3k_new/npz_h2_29dof_new --sample 400 --out eval.json
uv run scripts/supermimic/export_onnx.py --checkpoint logs/supermimic/H2SuperMimic/<run>/ckpt_latest.pt
```

### 后台训练（SSH 断开后继续运行）

启动脚本用 `setsid nohup` 让训练脱离终端会话，SSH 或 VSCode 断开、关闭终端都不影响训练。日志写到 `logs/supermimic/train_h2_xpu_<时间>.log`，PID 写到 `logs/supermimic/train_h2_xpu.pid`。

```bash
cd /home/maozhong/work/tmp/myUnilab
scripts/supermimic/train_h2_xpu_nohup.sh                                              # 全量：5 万次迭代
scripts/supermimic/train_h2_xpu_nohup.sh algo.supermimic.schedule.total_iterations=10000  # 先跑短一点

tail -f logs/supermimic/train_h2_xpu_*.log | grep -v RuntimeWarning   # 查看进度（重连后也可以看）
tensorboard --logdir logs/supermimic/H2SuperMimic                      # 查看曲线
kill $(cat logs/supermimic/train_h2_xpu.pid)                            # 停止训练
scripts/supermimic/train_h2_xpu_nohup.sh algo.resume=logs/supermimic/H2SuperMimic/<run>/ckpt_latest.pt  # 断点续训
```

训练开始的前一分钟左右在做 `torch.compile` 编译，这段时间没有日志输出是正常的。

**数据**：`conf/supermimic/task/h2_supermimic/mujoco.yaml`（以及 `mujoco_xpu.yaml`）中的 `env.motion_dirs` 指向两个 BONES-SEED H2 目录：

- 共 1946 + 3515 个 clip，两个目录之间没有重复；
- 每个 npz 含 33 个 body，按 MuJoCo 顺序排列；
- 帧率 50 fps。

## 目前的验证结果

- **原版 ONNX 跑通**：SuperMimic 训练出的 `h2_mimic_full.onnx` 不做任何修改，直接在 UniLab 环境中运行。
  - 从两个目录随机抽 100 个 clip：平均归一化回报 0.939，按步加权 0.898，0 次摔倒。
  - 参考实现在 mjlab 上 400 个 clip 的结果是 0.894 / 0.869。
  - 这从端到端验证了 obs / goal 布局、动作映射、物理、奖励、终止和帧时序。
- **torch 侧与仿真侧一致**（单测）：
  - FK 与 MuJoCo 的 `xpos/xquat/objectVelocity` 一致；
  - 解析 observation / goal 与环境 observation 一致；
  - torch 奖励与环境奖励一致（8 项全部）；
  - npz 中的参考 body 状态与参考状态的 FK 结果一致；
  - 零初始化的世界模型等价于匀速外推一步。
- **CPU 冒烟训练**：世界模型的 1 步和 8 步误差在下降，想象 rollout 的存活率在上升。
- **仿真吞吐**（12 核笔记本 CPU，mujoco backend）：4096 env 每步 0.118 s，1024 env 每步 0.035 s。
  - 环境仿真的是只保留碰撞部分的场景（`strip_visual_meshes: true`）：去掉可视网格和天空盒，碰撞网格只保留凸包顶点。
  - 这样做的原因：摩擦 / COM 随机化会让 backend 给每个 env 复制一份模型，单份模型从 187 MB 降到了 2 MB。
  - 需要渲染时把该项设为 false。
- **Intel XPU**（Core Ultra X7 358H 核显，`mujoco_xpu` 配置，全量数据，4096 env）：
  - 速度 0.74 it/s。每次迭代仿真 0.15 s；每次 WM 更新 0.89 s，每次 policy 更新 2.1 s，每次迭代做 0.4 次更新。
  - 5 万次迭代约 19 小时。
  - 系统内存占用约 24 GB / 30 GB（核显与 CPU 共享内存）。
  - 在相同 replay 数据上对比 MLP 矩阵乘的精度：bf16 与 fp32 的 WM 评测 nse1 为 0.0810 vs 0.0813，nse8 为 0.165 vs 0.164，bf16 快 2.7 倍。
  - 前 300 次迭代的变化：

    | 指标 | 起点 | 300 次迭代后 |
    |---|---|---|
    | WM nse1 | 0.49 | 0.10 |
    | imag_ret | 0.09 | 0.43 |
    | train_ret | 0.42 | 0.51 |
