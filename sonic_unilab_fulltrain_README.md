# SONIC Full Training in UniLab

This guide describes from-scratch SONIC PPO motion-tracking training in UniLab. It uses
Fourier GR2 as the main example and covers environment setup, motion retargeting,
curriculum training, checkpoint management, and split ONNX export.

The same SONIC core is also used by G1 and H2, with robot-specific dimensions:

| Robot | Encoder input | Proprio | Actor input | Actions | Critic |
|---|---:|---:|---:|---:|---:|
| G1 | 640 | 930 | 1570 | 29 | 1645 |
| H2 head-welded | 640 | 930 | 1570 | 29 | 1645 |
| GR2 fixed-head | 600 | 870 | 1470 | 27 | 1545 |

H2's head is welded (no head joints/actuators), so its joint set matches Unitree G1
exactly and all SONIC dimensions are identical to G1. H2-specific setup and the (small)
differences from the GR2 flow are in [Section 8](#8-h2-training-differences-from-gr2).

GR2 and H2 split ONNX export is supported and numerically checked against the PyTorch
model. All commands below run from the UniLab repository root:

```bash
cd /home/maozhong/work/my_sonic/UniLab
```

Always run Python tools through `uv run --no-sync`. A plain `uv run` or `uv sync` may
replace the locally selected accelerator-specific PyTorch build.

## 1. GR2 Training Contract

The source GR2 model has 29 scalar joints, including head yaw and pitch. The SONIC
training profile fixes the head and exposes 27 controlled joints:

```text
waist yaw                         1
left/right arms, 7 each          14
left/right legs, 6 each          12
                                  --
total                             27
```

The following widths must remain equal throughout the pipeline:

```text
MuJoCo scalar joints = actuators = NPZ joints = action scale = policy actions = 27
```

The training asset is:

```text
src/unilab/assets/robots/gr2/scene_sonic_27dof.xml
```

The GR2 actor contract is:

```text
future encoder input: 10 x (27 joint positions + 27 joint velocities + 6 anchor) = 600
proprio history:      10 x (3 gyro + 27 jpos + 27 jvel + 27 action + 3 gravity) = 870
actor input:          600 + 870 = 1470
decoder input:        token 64 + proprio 870 = 934
action output:        27
privileged critic:    1545
```

GR2 is trained from scratch. Do not load the G1 `last.pt`: its encoder input, decoder
input, output width, and joint meanings are incompatible.

## 2. Environment Setup

### 2.1 Install `uv` and system prerequisites

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Ubuntu or Debian with a system Python, install the native build prerequisites used
by the MuJoCo extra:

```bash
sudo apt-get install build-essential python3-dev
```

A Python installed with `uv python install` already includes the required headers.

### 2.2 Install UniLab dependencies

For an Intel Arc GPU or Intel integrated GPU:

```bash
make sync-xpu
```

This installs the MuJoCo/Motrix extras, then installs the Intel XPU PyTorch build. Other
supported setup paths are:

```bash
make setup       # Linux CUDA, macOS, or Windows
```

Verify the Intel training environment:

```bash
uv run --no-sync python -c \
  "import torch; print(torch.__version__); print('xpu:', torch.xpu.is_available())"
```

For XPU training, `torch.xpu.is_available()` must print `True`.

### 2.3 Run focused contract tests

```bash
uv run --no-sync pytest tests/test_gr2_sonic_27dof_contract.py \
  tests/envs/test_motion_loader.py -q
```

These tests cover the GR2 asset dimensions/order, fixed head, motion width, actor shape,
and local IMU velocity behavior.

## 3. Prepare GR2 Motion Data

### 3.1 Retarget G1 BONES-SEED CSV to GR2 27-DoF NPZ

The converter accepts one CSV, a directory, or the repository-supported input layout.
Always select the 27-DoF target model explicitly:

```bash
CSV_DIR=/path/to/bones-seed/g1/csv
NPZ_DIR=/path/to/npz_gr2_27dof
GR2_MODEL=src/unilab/assets/robots/gr2/scene_sonic_27dof.xml

uv run --no-sync python scripts/motion/g1_csv_to_gr2_npz.py \
  --input "$CSV_DIR" \
  --output "$NPZ_DIR" \
  --gr2_model_xml "$GR2_MODEL" \
  --input_fps 120 \
  --output_fps 50 \
  --skip_existing
```

The retargeter performs task-space IK, applies an explicit G1-to-GR2 joint/body map,
drops G1 waist roll/pitch, handles the GR2 elbow sign convention, grounds the feet, and
writes `joint_names` and `body_names` metadata.

Before a large conversion, validate the model and output resolution cheaply:

```bash
uv run --no-sync python scripts/motion/g1_csv_to_gr2_npz.py \
  --input "$CSV_DIR" \
  --output "$NPZ_DIR" \
  --gr2_model_xml "$GR2_MODEL" \
  --limit 1 --dry-run
```

For a visually inspectable sample, omit `--dry-run`, then replay it:

```bash
uv run --no-sync python scripts/motion/replay_npz.py \
  --npz_file "$NPZ_DIR/<clip>_gr2.npz" \
  --model_file "$GR2_MODEL"
```

Expected data properties are 27-wide `joint_pos`/`joint_vel`, finite arrays, exact
`joint_names` agreement with the MuJoCo scalar joint order, and no joint-limit violation.

### 3.2 Inspect and materialize a calm subset

Stage 1 should start with low-dynamic motions. Preview the default Stage 1 filter:

```bash
uv run --no-sync python scripts/motion/filter_calm_gr2.py \
  --src "$NPZ_DIR" --dry-run
```

The default limits are:

| Metric | Stage 1 limit |
|---|---:|
| body angular velocity p99 | 6.5 rad/s |
| absolute joint velocity p99 | 5.5 rad/s |
| base linear velocity p99 | 1.35 m/s |
| base angular velocity p99 | 2.0 rad/s |
| base height range | 0.15 m |
| maximum foot height | 0.35 m |

To create a persistent symlinked subset and CSV report:

```bash
uv run --no-sync python scripts/motion/filter_calm_gr2.py \
  --src "$NPZ_DIR" \
  --dst /path/to/npz_gr2_27dof_calm
```

The GR2 launcher performs this filtering automatically for curriculum stages 1 through
3. Use the standalone command when inspecting distributions or curating a fixed dataset.

## 4. Training Configuration

The owner configuration is:

```text
conf/ppo/task/gr2_motion_tracking/sonic_full_train.yaml
```

Important defaults and launcher overrides are:

| Parameter | Value | Source |
|---|---:|---|
| task | `GR2SonicMotionTracking` | YAML |
| simulator | MuJoCo | YAML |
| actor | `SonicGR2ActorModel` | YAML |
| pretrained actor | none | YAML |
| actor observation | 1470 | env/actor contract |
| critic observation | 1545 | env/critic contract |
| actions | 27 | fixed-head asset |
| number of environments | 4096 | launcher |
| rollout steps per environment | 24 | launcher |
| maximum iterations | 20000 | launcher |
| checkpoint interval | 100 | launcher |
| initial action std | 0.50 | launcher; checkpoint value is restored on resume |
| learning rate | stage-dependent (`7.5e-6` for Stage 2.8) | launcher |
| entropy coefficient | stage-dependent (`0.001` for Stage 2.8) | launcher |
| desired KL | `0.01` | launcher |
| adaptive maximum learning rate | `2e-4` | launcher |

The initial termination overrides are deliberately permissive for a random policy:

```yaml
env:
  strict_foot_pos_threshold: 0.5
  strict_anchor_ori_error_sq: 0.8
  strict_height_threshold: 0.45
```

These are reference-tracking errors, not absolute robot heights. Tighten them only after
training becomes stable, while recording termination reason frequencies. A typical order
is height error first, then root orientation, then foot position.

### 4.1 PD control, action scale, and torque

GR2 uses MuJoCo position actuators. The policy output is a residual target around the
effective default pose, not a direct torque:

$$
q_{target}\coloneqq q_{default}+a\,s, \qquad
	au\coloneqq K_p(q_{target}-q)-K_d\dot q
$$

Here $a$ is the policy action and $s$ is the per-joint action scale. The current GR2
profile uses:

$$
s\coloneqq 0.25\frac{\tau_{limit}}{K_p}
$$

This makes the action range depend on both the actuator effort limit and stiffness. The
actuator, action-scale, scalar-qpos, and NPZ orders must remain exactly aligned.

Current PD values, in each side's actuator order, are:

| Group | $K_p$ | $K_d$ |
|---|---|---|
| waist yaw | `200` | `10` |
| arm | `[300, 300, 100, 100, 50, 50, 50]` | `[10, 10, 5, 5, 5, 5, 5]` |
| leg | `[90, 180, 120, 90, 30, 60]` | `[19, 10, 9, 19, 5, 3.5]` |

These are the current manufacturer-derived simulation values, not a completed hardware
calibration. The `forcerange` values also need confirmation as continuous versus peak
torque limits before a real-robot transfer.

PD changes are high-impact:

- Raising $K_p$ improves pose tracking but can increase torque saturation, impact force,
  chatter, and sensitivity to latency or model error.
- Lowering $K_p$ may make the robot compliant but unable to support its weight or follow
  fast references.
- Too little $K_d$ permits oscillation; too much $K_d$ can make motion sluggish and
  produce large velocity-dependent torques.
- Changing $K_p$ without recomputing action scale changes both the closed-loop dynamics
  and the policy's effective target-position range.
- A policy trained with one PD/action-scale contract must not silently deploy with
  another. Treat the gains, effort limits, control rate, default pose, and action scale as
  one versioned control contract.

### 4.2 Contact, collision, and friction

Two different mechanisms exist and must not be confused:

1. **Physical MuJoCo collision.** The visual foot meshes do not collide. Each foot uses
   one sole box with `condim=3`, `friction="1 0.005 0.0001"`, and
   `solref="0.005 1"`. This geometry came from a Fourier 21-DoF reference model and is a
   simulation baseline, not final hardware calibration.
2. **Touch sensors.** `lf-touch` and `rf-touch` are attached to `lf-tc` and `rf-tc` sites.
   The current SONIC reward and termination paths do not consume these sensors.

Do not tune friction, contact geometry, PD, action scale, and reward penalties at the same
time.

## 5. Curriculum Training

The launcher is:

```text
scripts/sonic/full_train_gr2.sh
```

It creates a temporary Hydra motion list, avoiding Linux command-line length limits for
thousands of clips. It launches training through `setsid nohup`, so the process survives
the terminal closing.

### 5.1 Validate configuration without training

```bash
SOURCE_NPZ_DIR="$NPZ_DIR" CURRICULUM_STAGE=1 DRY_RUN=1 \
  ./scripts/sonic/full_train_gr2.sh
```

This filters the Stage 1 motions and composes the complete Hydra job without starting PPO.

### 5.2 Start Stage 1 from scratch

```bash
SOURCE_NPZ_DIR="$NPZ_DIR" CURRICULUM_STAGE=1 \
  ./scripts/sonic/full_train_gr2.sh
```

Stage 1 rejects `LOAD_RUN` by design. The default filtered dataset is materialized under:

```text
/tmp/sonic_train/gr2_stage1
```

The launcher prints the log path, for example:

```text
/tmp/sonic_train/full_train_gr2_stage1_YYYYMMDD_HHMMSS.log
```

### 5.3 Monitor training

```bash
tail -f /tmp/sonic_train/full_train_gr2_stage1_*.log
```

TensorBoard:

```bash
uv run --no-sync tensorboard \
  --logdir logs/rsl_rl_ppo/GR2SonicMotionTracking/
```

Checkpoints are written to:

```text
logs/rsl_rl_ppo/GR2SonicMotionTracking/<timestamp>_mujoco/model_<iteration>.pt
```

Monitor more than total reward. In particular, inspect termination frequencies, action
standard deviation, per-term unweighted rewards, weighted torque contribution, joint
limits, foot contact-height gates, and actuator saturation. GR2 is substantially heavier
than G1, so G1 torque-penalty magnitudes are not directly transferable.

### 5.4 Continue with Stage 2, Stage 2.5, Stage 2.8, Stage 3, and full data

For every stage after Stage 1, provide the previous run directory name through `LOAD_RUN`.
The launcher finds and resumes its numerically latest `model_*.pt`.

```bash
# Replace with the directory name under logs/rsl_rl_ppo/GR2SonicMotionTracking/.
STAGE1_RUN=2026-09-24_10-53-35_mujoco

SOURCE_NPZ_DIR="$NPZ_DIR" CURRICULUM_STAGE=2 LOAD_RUN="$STAGE1_RUN" \
  ./scripts/sonic/full_train_gr2.sh
```

Then continue from the Stage 2 run:

```bash
STAGE2_RUN=2026-09-24_22-40-40_mujoco
CURRICULUM_STAGE=2.5 LOAD_RUN="$STAGE2_RUN" \
  ./scripts/sonic/full_train_gr2.sh
```

Stage 2.5 merges the base motion directory with the upright-manipulation directory and
applies the Stage 2 filter to the combined set. Its lower learning rate and entropy
coefficient reduce policy drift while introducing the new motion distribution. Because
the full actor state is restored, the resumed action standard deviation starts from the
checkpoint value (about `0.91` for `model_2800.pt`), not the construction-time
`init_std=0.50`.

Stage 2.5 also replaces the raw action-difference penalty with
`target_rate_l2=-0.05`, which penalizes changes in the physical PD position target
`action_scale * (action_t - action_t-1)`, and enables
`upper_body_joint_pos=0.5` for the 14 shoulder, elbow, and wrist joints. Neither term
changes policy inputs, outputs, action scaling, or checkpoint shapes, so an existing
27-DoF checkpoint remains resume-compatible.

TensorBoard rollout logs include overlapping per-step termination rates under
`termination/*` and total/upper-body requested-PD-torque saturation estimates under
`diagnostic/*`. The saturation values compare requested PD torque with the limits
implied by the fixed GR2 action-scale contract; they are estimates, not measured
actuator-force telemetry.

Stage 2.8 merges the same base and upright-manipulation sources, then admits moderately
more dynamic clips than Stage 2.5. To resume the exact checkpoint instead of the newest
checkpoint in the run directory:

```bash
CURRICULUM_STAGE=2.8 \
LOAD_RUN=2026-09-30_22-45-21_mujoco CHECKPOINT=7900 \
STRICT_FOOT_POS_THRESHOLD=0.50 \
STRICT_ANCHOR_ORI_ERROR_SQ=0.80 \
STRICT_HEIGHT_THRESHOLD=0.45 \
MAX_ITERATIONS=5000 ./scripts/sonic/full_train_gr2.sh
```

Its defaults are `learning_rate=7.5e-6`, `adaptive_lr_max=1e-5`,
`entropy_coef=0.001`, `motion_global_root_pos=0.75`, `action_rate_l2=0`,
`target_rate_l2=-0.02`, `upper_body_joint_pos=1.0`, and
`anti_shake_ang_vel=0`. The reduced smoothness penalties avoid suppressing the faster
wrist references admitted by this stage. The generated subset is stored under
`/tmp/sonic_train/gr2_stage2_8`.

Stage 2.8 also defaults to `FOCUS_PATTERN='*macarena*.npz'` and
`FOCUS_REPEAT=5`. In the current filtered subset this matches 34 clips and raises their
frame-weighted probability under mixed sampling from `1.65%` to `7.73%`. Repetition
duplicates paths only in the generated Hydra motion list; it does not copy NPZ files.
Set `FOCUS_REPEAT=1` to disable focus oversampling.

Stage 2.8 keeps the permissive Stage 2.5 termination gates while the policy adapts to
the broader distribution. Stage 3 defaults to `strict_foot_pos_threshold=0.45`,
`strict_anchor_ori_error_sq=0.70`, and `strict_height_threshold=0.40`. The orientation
value is squared geodesic error: `0.80` and `0.70` correspond to approximately `51.2`
and `47.9` degrees. These gates prevent collapsed tracking; they are not precision
targets.

Continue Stage 3 from the Stage 2.8 run:

```bash
STAGE2_8_RUN=<stage-2.8-run-directory>
SOURCE_NPZ_DIR="$NPZ_DIR" CURRICULUM_STAGE=3 LOAD_RUN="$STAGE2_8_RUN" \
STRICT_FOOT_POS_THRESHOLD=0.45 STRICT_ANCHOR_ORI_ERROR_SQ=0.70 \
STRICT_HEIGHT_THRESHOLD=0.40 \
  ./scripts/sonic/full_train_gr2.sh
```

Finally train on all retargeted clips:

```bash
STAGE3_RUN=<stage-3-run-directory>
SOURCE_NPZ_DIR="$NPZ_DIR" CURRICULUM_STAGE=full LOAD_RUN="$STAGE3_RUN" \
  ./scripts/sonic/full_train_gr2.sh
```

The automatic filter limits are:

| Stage | body ang p99 | joint vel p99 | base lin p99 | base ang p99 | base Z range | foot max |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 6.5 | 5.5 | 1.35 | 2.0 | 0.15 | 0.35 |
| 2 | 8.0 | 7.0 | 1.8 | 2.5 | 0.25 | 0.45 |
| 2.5 | 8.0 | 7.0 | 1.8 | 2.5 | 0.25 | 0.45 |
| 2.8 | 10.0 | 8.5 | 2.0 | 3.5 | 0.30 | 0.55 |
| 3 | 12.0 | 9.5 | 2.2 | 4.0 | 0.35 | 0.65 |
| full | no filter | no filter | no filter | no filter | no filter | no filter |

### 5.5 Override training scale or use an existing dataset

The launcher accepts these environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `SOURCE_NPZ_DIR` | repository-local development path | complete 27-DoF motion source |
| `MANIPULATION_NPZ_DIR` | repository-local upright-operations path | motions added during Stages 2.5 and 2.8 |
| `NPZ_DIR` | unset | bypass automatic filtering and use this directory directly |
| `CURRICULUM_STAGE` | `1` | `1`, `2`, `2.5`, `2.8`, `3`, or `full` |
| `LOAD_RUN` | unset | prior run directory; required after Stage 1 |
| `CHECKPOINT` | unset | exact checkpoint number/name; newest checkpoint is used when unset |
| `LEARNING_RATE` | stage-dependent | `7.5e-6` for Stage 2.8 |
| `ADAPTIVE_LR_MAX` | stage-dependent | `1e-5` for Stage 2.8; otherwise `2e-4` |
| `ENTROPY_COEF` | stage-dependent | `0.001` for Stage 2.8 |
| `ROOT_POS_SCALE` | stage-dependent | `0.75` for Stages 2.5 and 2.8 |
| `ACTION_RATE_L2_SCALE` | stage-dependent | `0` for Stages 2.5 and 2.8 |
| `TARGET_RATE_L2_SCALE` | stage-dependent | `-0.02` for Stage 2.8 |
| `UPPER_BODY_JOINT_POS_SCALE` | stage-dependent | `1.0` for Stage 2.8 |
| `ANTI_SHAKE_ANG_VEL_SCALE` | stage-dependent | `0` for Stage 2.8; otherwise `-2.5e-3` |
| `FOCUS_PATTERN` | stage-dependent | `*macarena*.npz` for Stage 2.8; unset otherwise |
| `FOCUS_REPEAT` | stage-dependent | `5` for Stage 2.8; `1` disables focus oversampling |
| `STRICT_FOOT_POS_THRESHOLD` | stage-dependent | `0.50` for Stage 2.8; `0.45` for Stage 3 |
| `STRICT_ANCHOR_ORI_ERROR_SQ` | stage-dependent | `0.80` for Stage 2.8; `0.70` for Stage 3 |
| `STRICT_HEIGHT_THRESHOLD` | stage-dependent | `0.45` for Stage 2.8; `0.40` for Stage 3 |
| `INIT_STD` | `0.50` | construction value; restored actor std wins on resume |
| `NUM_ENVS` | `4096` | parallel environments |
| `NUM_STEPS` | `24` | rollout steps per environment |
| `MAX_ITERATIONS` | `20000` | PPO iterations in this invocation |
| `SAVE_INTERVAL` | `100` | checkpoint interval |
| `WORK_DIR` | `/tmp/sonic_train` | filtered subsets, Hydra overrides, and logs |
| `DRY_RUN` | `0` | set to `1` for Hydra composition only |

Example small smoke run on a prepared dataset:

```bash
NPZ_DIR=/path/to/small_gr2_subset \
CURRICULUM_STAGE=1 NUM_ENVS=4 NUM_STEPS=2 MAX_ITERATIONS=1 SAVE_INTERVAL=1 \
  ./scripts/sonic/full_train_gr2.sh
```

## 6. Export Split ONNX

Use `export_full_deploy_onnx.py` for from-scratch checkpoints trained with both joint-order
permutation flags disabled, as in the GR2/H2 full-train configurations.

Select a checkpoint and export GR2:

```bash
RUN=logs/rsl_rl_ppo/GR2SonicMotionTracking/<timestamp>_mujoco
CKPT="$RUN/model_500.pt"

uv run --no-sync python scripts/sonic/export_full_deploy_onnx.py \
  --robot gr2 \
  --ckpt "$CKPT" \
  --out-dir "$RUN/deploy_onnx" \
  --verify
```

The GR2 export produces:

```text
model_encoder_gr2.onnx  obs_dict[1,600] -> encoded_tokens[1,64]
model_decoder.onnx      obs_dict[1,934] -> action[1,27]
```

`--verify` compares ONNX Runtime outputs with the mapped PyTorch core. Export should not
be accepted when verification fails.

For H2 and GR2, the exported split graphs preserve the current single MuJoCo joint order.
They have been numerically verified against their training checkpoints, but have not yet
been cross-tested with the external SONIC deploy program. Integrating either robot into
that runtime still requires an explicit observation layout, joint-order, action-order,
and hardware control contract review.

For G1 only, `--merged-1762` can additionally generate the existing deploy-buffer bridge:

```bash
uv run --no-sync python scripts/sonic/export_full_deploy_onnx.py \
  --robot g1 --ckpt "$G1_CKPT" --out-dir "$G1_OUT" \
  --merged-1762 --verify
```

The raw SONIC `last.pt` is G1-only. H2 and GR2 export requires an RSL-RL `model_*.pt`
checkpoint containing `actor_state_dict['core.*']`.

## 7. Validation and Troubleshooting

Run the focused checks after changing the GR2 asset, converter, environment, actor, or
training configuration:

```bash
uv run --no-sync pytest tests/test_gr2_sonic_27dof_contract.py \
  tests/envs/test_motion_loader.py -q

uv run --no-sync pytest tests/config/test_config_system.py -q
```

| Symptom | Check |
|---|---|
| XPU unavailable | Run `make sync-xpu`, then always use `uv run --no-sync`. |
| Motion width is 29 instead of 27 | Regenerate with `--gr2_model_xml .../scene_sonic_27dof.xml`. |
| Launcher reports no NPZ clips | Check `SOURCE_NPZ_DIR`/`NPZ_DIR` and ensure files end in `.npz`. |
| Stage 1 rejects the command | Unset `LOAD_RUN`; Stage 1 is intentionally from scratch. |
| Stage 2/3/full rejects the command | Set `LOAD_RUN` to an existing GR2 run directory containing `model_*.pt`. |
| Hydra command is too long | Use `full_train_gr2.sh`; it writes the motion list to a temporary config. |
| Reward is unstable | Inspect termination reasons, action std, torque contribution, limits, and saturation before changing aggregate reward scales. |
| ONNX load has size mismatches | Confirm `--robot` matches the checkpoint robot and action width. |
| `--merged-1762` is rejected | This buffer layout is G1-only; export H2/GR2 split graphs without it. |

## 8. H2 Training (differences from GR2)

H2 uses the same SONIC core, curriculum launcher, and from-scratch philosophy as GR2.
This section lists only what differs; everything else (environment setup, monitoring,
curriculum stages, checkpoint/resume, torque diagnostics, split ONNX export) follows the
GR2 sections above unchanged.

### 8.1 Joint contract

H2's head is welded (no head joints/actuators; the head bodies stay for rendering only),
so H2 exposes **29 controlled joints**, identical to Unitree G1:

```text
left/right legs, 6 each          12
waist yaw, roll, pitch            3
left/right arms, 7 each          14
                                  --
total                             29
```

This differs from GR2's 27-DoF profile, which keeps only `waist_yaw` (1) and drops waist
roll/pitch. The widths that must stay equal throughout the H2 pipeline are:

```text
MuJoCo scalar joints = actuators = NPZ joints = action scale = policy actions = 29
```

H2 actor contract: encoder `640 = 10·(2·29 + 6)`, proprio `930 = 10·(3 + 3·29 + 3)`,
actor `1570`, decoder `token 64 + proprio 930 = 994`, action `29`, privileged critic
`1645`. These equal G1; they are **not** the pre-weld 31-DoF values (680/990/1670/31/1745).

The training asset is `src/unilab/assets/robots/h2/scene_flat.xml` (29-DoF; head welded).
Do not load the G1 `last.pt` — H2 trains from scratch like GR2.

### 8.2 Robot-specific names and constants (vs GR2)

| Aspect | GR2 | H2 |
|---|---|---|
| Controlled joints | 27 (waist_yaw only) | 29 (waist yaw/roll/pitch) |
| Anchor body | `base` | `pelvis` |
| Foot body | `*_foot_roll_link` | `*_ankle_roll_link` |
| Training asset | `gr2/scene_sonic_27dof.xml` | `h2/scene_flat.xml` |
| Retarget script | `g1_csv_to_gr2_npz.py` | `g1_csv_to_h2_npz2.py` |
| Calm filter | `filter_calm_gr2.py` | `filter_calm_h2.py` |
| Launcher | `full_train_gr2.sh` | `full_train_h2.sh` |
| Run directory | `logs/rsl_rl_ppo/GR2SonicMotionTracking/` | `logs/rsl_rl_ppo/H2SonicMotionTracking/` |
| NPZ `body_names` metadata | present | absent (filter resolves indices from the model) |
| `feet_contact_height` | 0.08 | 0.15 |
| Actuator effort | manufacturer GR2 | ≈ 3× G1 (watch `joint_torque_l2`) |

### 8.3 Retarget G1 CSV to H2 29-DoF NPZ

H2 retargeting is task-space IK (not an explicit joint map like GR2). It scales the body
targets anisotropically — leg-length ratio horizontally, pelvis-height ratio vertically —
**plus a separate arm-length ratio for the distal arm bodies** (so hand reach follows H2's
proportions, important for arm-heavy clips like macarena), and temporally smooths the
floor-projection lift so the exported root vertical velocity has no per-frame spikes.

```bash
CSV_DIR=/path/to/bones-seed/g1/csv
NPZ_DIR=/path/to/npz_h2_29dof

uv run --no-sync python scripts/motion/g1_csv_to_h2_npz2.py \
  --input "$CSV_DIR" --output "$NPZ_DIR" \
  --input_fps 120 --output_fps 50 --skip_existing
```

Scales (`arm_scale`, `h_scale`, `v_scale`) are auto-derived from the two robots' standing
kinematics; `--ground_smooth` (default 5) sets the lift-smoothing window. Validate one clip
with `scripts/motion/replay_npz.py` and the H2 model before a large run. Output is 29-wide
`joint_pos`/`joint_vel`; the body arrays are indexed by MuJoCo body id (no `body_names`
key — the H2 filter resolves pelvis/foot indices from the model).

The default standing motion `src/unilab/assets/motions/h2/stand.npz` is also 29-DoF; if you
re-weld or change the model, regenerate it from the keyframe.

### 8.4 Curriculum launcher

`scripts/sonic/full_train_h2.sh` mirrors `full_train_gr2.sh` (same stages, filter limits,
manipulation merge, resume, DRY_RUN, and env-var overrides from Section 5.5), driving
`H2SonicMotionTracking` and `filter_calm_h2.py`. Defaults:

```text
SOURCE_NPZ_DIR         ~/work/sonic_vla_infer/bones_seed_3k_new/npz_h2_29dof
MANIPULATION_NPZ_DIR   ~/work/sonic_vla_infer/bones-seed/selections/upright_nzp_h2_29dof
```

```bash
# Stage 1 (from scratch)
SOURCE_NPZ_DIR="$NPZ_DIR" CURRICULUM_STAGE=1 ./scripts/sonic/full_train_h2.sh
# later stages require LOAD_RUN, exactly as GR2
SOURCE_NPZ_DIR="$NPZ_DIR" CURRICULUM_STAGE=2 LOAD_RUN=<stage1-run> ./scripts/sonic/full_train_h2.sh
```

Two launcher behaviors differ from GR2:

- **macarena focus is advisory, not fatal.** Stage 2/2.5/3/full default to
  `FOCUS_PATTERN='*macarena*.npz'`, `FOCUS_REPEAT=5`; Stage 1 disables it. Because macarena
  clips have `body_ang_p99 >= 8.25`, the calm filters drop them entirely at Stage 1 (6.5)
  and Stage 2/2.5 (8.0); they re-enter only at Stage 2.8 (>=10, ~38/86) and Stage 3/full
  (>=12, all). When a focus pattern matches nothing the launcher prints a **warning and
  continues** (GR2 errors out), so Stage 1/2 run cleanly with no macarena to oversample.
- The smoothness terms `joint_acc_l2 = -2.5e-7` and `joint_torque_l2 = -1.0e-6` are passed
  as on GR2, but H2's effort is ≈3× G1, so `joint_torque_l2`'s raw magnitude is ≈9× larger;
  monitor `reward/joint_torque_l2` and lower it (≈ `-1e-7`) if it approaches the tracking
  terms.

### 8.5 Export split ONNX

Identical to GR2 but with `--robot h2`; H2 uses the single MuJoCo joint order (identity
permutation), producing `model_encoder_h2.onnx` (`[1,640] -> [1,64]`) and
`model_decoder.onnx` (`[1,994] -> [1,29]`). `--merged-1762` remains G1-only.