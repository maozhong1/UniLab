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
| 录制 rollout / 离屏渲染视频 | `scripts/supermimic/{record_rollout,render_rollout}.py` |
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

clip 只按名字过滤：`motion_exclude_keywords: [table, _box_]`。参考实现的 H2 过滤也只去掉了这类 clip。

## 运行

```bash
# Intel XPU（torch 2.7.0+xpu）：先执行一次 `make sync-xpu`，之后始终用 `uv run --no-sync`。
# mujoco_xpu 配置：4096 env，batch 2048, MLP 矩阵乘用 bf16，开启 torch.compile，数据与超参同参考 H2 训练。
uv run --no-sync scripts/train_supermimic.py task=h2_supermimic/mujoco_xpu

# 评测（评测协议：无噪声、无 DR，摩擦 1，从每个 clip 的第一帧开始）。
# --sample N 随机抽 N 个 clip（种子 0）；--clip-list FILE（每行一个 npz 文件名，不带后缀）固定 clip 集合，
# 方便不同 checkpoint 在同一批 clip 上对比。评 ONNX 时把 --checkpoint 换成 --onnx <文件>。
uv run scripts/supermimic/eval_benchmark.py --checkpoint logs/supermimic/H2SuperMimic/<run>/ckpt_latest.pt \
    --npz-dir ~/work/sonic_vla_infer/bones-seed/selections/upright_nzp_h2_29dof_new \
    --npz-dir ~/work/sonic_vla_infer/bones_seed_3k_new/npz_h2_29dof_new --sample 400 --out eval.json
uv run scripts/supermimic/export_onnx.py --checkpoint logs/supermimic/H2SuperMimic/<run>/ckpt_latest.pt
```

### 后台训练

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


### 续训
`algo.resume=<ckpt>` 会恢复世界模型、policy、critic 及它们的 Adam 状态，以及重置采样器的统计量和迭代计数。replay buffer 不保存，续训后从当前迭代开始重新装数据；前 `schedule.warmup_iterations` 次迭代（默认 40）只采数据、不更新。

- **断点续训**（训练中途停了）：用同一个 task，加 `algo.resume=.../ckpt_latest.pt`（见上一节）。学习率继续按原来的 cosine 曲线走，曲线按绝对迭代次数计算。
- **训完后接着训**（例如 5 万 → 10 万）：用续训专用的 task。如果只把 `total_iterations` 调大，cosine 曲线会被拉长，学习率会从 3% 的底部一下跳到约 30%。`h2_supermimic/mujoco_xpu_continue100k` 的做法是：
  1. 先只采数据 1000 次迭代，把 replay buffer 装满；
  2. 学习率在 2000 次迭代内从基础学习率的 3% 线性升到 25%；
  3. 然后按 cosine 降回 3%（约 8.97 万次），保持到 10 万次。

```bash
# 传入 task=... 会替换启动脚本默认的 task
scripts/supermimic/train_h2_xpu_nohup.sh task=h2_supermimic/mujoco_xpu_continue100k \
    algo.resume=logs/supermimic/H2SuperMimic/<run>/ckpt_0050000.pt
```

| 参数 | 含义 |
|---|---|
| `algo.supermimic.schedule.lr_restart_iteration` | 学习率重启开始的迭代数（一般就是续训的起点） |
| `algo.supermimic.schedule.lr_restart_peak_ratio` | 重启后的峰值，按基础学习率的比例 |
| `algo.supermimic.schedule.lr_restart_warmup_iterations` | 线性升温的长度，从重启后第一次更新开始算 |
| `algo.supermimic.schedule.warmup_iterations` | （重新）开始后只采数据的迭代数；续训时建议 1000 左右，避免世界模型在几乎空的 buffer 上训练 |

控制台日志会打印当前学习率（`lr ...`）。学习率重启后的高位阶段，评测分数会先下降（峰值附近约 -0.02），学习率降下来后再回升。判断重启是否有效，要看学习率降下来之后的 checkpoint，不要看中途的。

### 录制和渲染视频

`record_rollout.py` 按评测协议把每个 clip 播放一遍（checkpoint 或 ONNX 都可以），记录每一步机器人和参考动作的 `qpos`。`render_rollout.py` 把记录离屏渲染成每个 clip 一个 mp4，可以再串成一个视频：

- 灰色实体是策略的实际动作，半透明蓝色是参考动作；
- 左上角标注 clip 名、归一化回报和时间。


```bash
NPZ=~/work/sonic_vla_infer/GR00T-WholeBodyControl-ov/gear_sonic_deploy/h2/npz_new   # 8 条展示用 clip
uv run --no-sync scripts/supermimic/record_rollout.py \
    --checkpoint logs/supermimic/H2SuperMimic/<run>/ckpt_0100000.pt --npz-dir $NPZ \
    --out media/h2_rollouts.npz                       # 或 --onnx <文件>；用 --clips <名字> ... 指定 clip
MUJOCO_GL=osmesa uv run --no-sync scripts/supermimic/render_rollout.py \
    --traj media/h2_rollouts.npz --out-dir media/h2_out --tag "ckpt 100000" --concat media/h2_all.mp4
```

渲染选项：`--seconds N`（每个 clip 只渲染前 N 秒）、`--clips <名字> ...`、`--distance`（默认 3.2）、`--azimuth`（140）、`--elevation`（-14）、`--width/--height`（1280x720）、`--fps`（25）、`--no-ghost`（不画参考动作）。

参考动作按它在世界坐标中的位置绘制，所以机器人和蓝影之间的错位就是累积的全局漂移。

```bash
mkdir -p media/sbs; for f in media/a/*.mp4; do n=$(basename $f); ffmpeg -y -loglevel error -i media/a/$n -i media/b/$n \
  -filter_complex "[0:v]scale=960:540[a];[1:v]scale=960:540[b];[a][b]hstack=inputs=2" \
  -c:v libx264 -pix_fmt yuv420p -crf 20 media/sbs/$n; done
```

**数据**：`conf/supermimic/task/h2_supermimic/mujoco.yaml`（以及 `mujoco_xpu.yaml`）中的 `env.motion_dirs` 指向两个 BONES-SEED H2 目录：

- 共 1946 + 3515 个 clip，两个目录之间没有重复；
- 每个 npz 含 33 个 body，按 MuJoCo 顺序排列；
- 帧率 50 fps。

## 目前的验证结果

- **原版 ONNX 跑通**：SuperMimic 训练出的 `h2_mimic_full.onnx` 不做任何修改，直接在 UniLab 环境中运行。
  - 从两个目录随机抽 100 个 clip：平均归一化回报 0.939，按步加权 0.898，0 次摔倒。
  - 这从端到端验证了 obs / goal 布局、动作映射、物理、奖励、终止和帧时序。
- **训练结果**（`mujoco_xpu`，bf16）：从过滤后的 5408 个 clip 中随机抽 400 个（种子 0，下表各行用同一批 clip；参考 README 的 400 个 clip 是另一次抽样）：

  | 策略 | 平均 | 按步加权 | 摔倒 |
  |---|---|---|---|
  | UniLab XPU，5 万次迭代 | 0.885 | 0.868 | 0 |
  | UniLab XPU，续训到 10 万次（`mujoco_xpu_continue100k`） | **0.906** | **0.889** | 0 |
  | `h2_mimic_full.onnx`（5 万多个 clip，20 万次迭代；上限） | 0.939 | 0.922 | 0 |

  - 5 万 → 10 万：按 clip 配对平均 +0.022（400 个 clip 中 333 个变好）；train_ret 0.742 → 0.762，WM nse1 0.0071 → 0.0063，imag_ret 2.666 → 2.738。
  - 最弱的 clip 是快节奏 dance（hiphop、vouge、slide；10 万次时 0.56–0.66），与参考实现一致。
  - 把 MLP 矩阵乘从 bf16 换成 fp32 做续训，结果与 bf16 相同，说明 bf16 在这里不是精度瓶颈。
- **torch 侧与仿真侧一致**（单测）：
  - FK 与 MuJoCo 的 `xpos/xquat/objectVelocity` 一致；
  - 解析 observation / goal 与环境 observation 一致；
  - torch 奖励与环境奖励一致（8 项全部）；
  - npz 中的参考 body 状态与参考状态的 FK 结果一致；
  - 零初始化的世界模型等价于匀速外推一步。
- **仿真吞吐**（12 核笔记本 CPU，mujoco backend）：4096 env 每步 0.118 s，1024 env 每步 0.035 s。
  - 环境仿真的是只保留碰撞部分的场景（`strip_visual_meshes: true`）：去掉可视网格和天空盒，碰撞网格只保留凸包顶点。
  - 这样做的原因：摩擦 / COM 随机化会让 backend 给每个 env 复制一份模型，单份模型从 187 MB 降到了 2 MB。
  - 需要渲染时把该项设为 false。
- **Intel XPU**（Core Ultra X7 358H 核显，`mujoco_xpu` 配置，全量数据，4096 env）：
  - 完整训练下的速度为 0.65–0.71 it/s。每次迭代仿真 0.15 s；每次 WM 更新约 1.1 s，每次 policy 更新约 2.1 s，每次迭代做 0.4 次更新。
  - 5 万次迭代约 21 小时。MLP 矩阵乘改用 fp32 时约 0.39 it/s。
  - 系统内存占用约 24 GB / 30 GB（核显与 CPU 共享内存）。
  - 在相同 replay 数据上对比 MLP 矩阵乘的精度：bf16 与 fp32 的 WM 评测 nse1 为 0.0810 vs 0.0813，nse8 为 0.165 vs 0.164，bf16 快 2.7 倍。
