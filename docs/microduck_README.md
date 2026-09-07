<h1 align="center"> Microduck on UniLab </h1>

<h3 align="center">
Bipedal walking RL for a ~0.8 kg / 25 cm Dynamixel-XL330 robot, ported to UniLab
</h3>

<p align="center">
  <em>CPU MuJoCo rollout · XPU/GPU learner · PPO (rsl_rl) · sim2real-frozen 61D obs</em>
</p>

Microduck is a small bipedal robot (14 Dynamixel XL330 servos) originally trained
in [`microduck_rl`](https://github.com/pollen-robotics/microduck) on **mjlab**
(MuJoCo-Warp). This directory is its port onto **UniLab**'s heterogeneous runtime:
simulation runs CPU-parallel through `mujoco_uni` and policy learning runs on an
accelerator (here Intel **XPU**). Policies are trained at 50 Hz, exported to ONNX,
and cross-checked back in `microduck_rl`'s `infer_policy.py` via a **shared 61D
observation contract**, which is what keeps the two frameworks sim2real-consistent.

---

## 1. Environment setup

The recommended toolchain is [`uv`](https://docs.astral.sh/uv/) (same as upstream
UniLab). Microduck adds one **optional extra**, `microduck`, which pulls the BAM
actuator model.

```bash
# from the UniLab repo root
uv sync                                   # base UniLab environment

# Intel XPU: any `uv sync` drops the XPU torch build — reinstall it afterwards
uv pip install torch==2.7.0 --torch-backend xpu

# (optional, if HuggingFace Hub is blocked in your region)
export HF_ENDPOINT=https://hf-mirror.com
```

Microduck commands run with two extras and `--no-sync` (so the manual XPU torch is
not clobbered):

```bash
uv run --extra mujoco --extra microduck --no-sync <command…>
```

- **`mujoco`** — the `mujoco_uni` CPU simulation backend.
- **`microduck`** — the BAM actuator (`better-actuator-models`, import name `bam`).
  It is an opt-in extra so it never perturbs UniLab's default dependency
  resolution. Note the upstream pin `requires-python < 3.13`, so the venv must be
  **Python 3.12** (on 3.13 the extra resolves empty and env construction fails).

---

## 2. What the port actually is (key points)

### 2.1 `MicroduckBaseEnv` — the shared base for the whole task family

Every microduck task (velocity/walk today; standup, sitstand, roulade, rollers,
and their backlash twins later) subclasses **`MicroduckBaseEnv`**
(`src/unilab/envs/locomotion/microduck/base.py`). Everything that MUST stay
identical across the family — so trained policies are **hot-swappable** in the
deployed runtime — lives there exactly once:

- **61D actor observation contract** (frozen, shared, sim2real-critical):

  ```
  [ base_ang_vel(3), projected_gravity(3), joint_pos(14), joint_vel(14),
    last_action(14),  twist(3), head_pose(4), body_pose(6) ]
    └────────── 48 proprioception ──────────┘ └──── 13 command block ────┘
  ```

  The critic appends privileged `base_lin_vel(3)` → **64D**. A task that does not
  use a command slot **zero-pads** it (keeps the obs term, samples a tiny
  keep-alive range) — a slot is never deleted, or every other policy stops loading.

- **Servo joint indices are resolved by name**, never hardcoded — identity on the
  plain walk model, still correct on rollers/backlash models where passive joints
  interleave.

- The base owns obs, termination, DR, BAM wiring and the reward dispatch;
  subclasses only implement `_init_reward_functions` (fill `self._reward_fns`) and
  set `self._reward_cfg`. With no reward config it still steps (reward = 0), so the
  obs/actuator/termination machinery is testable before any task exists.

Layering: `ABEnv → NpEnv → LocomotionBaseEnv → MicroduckBaseEnv → MicroduckVelocityEnv`.

### 2.2 BAM actuator adaptation

At this scale the **actuator is most of the sim2real gap**, so microduck is trained
against the **BAM XL330 voltage/friction model**, not an ideal PD. `microduck_rl`
uses `bam.mjlab.BamActuator` inside MuJoCo-Warp; UniLab's CPU backend has **NO** such
hook, so the port splits BAM into two halves
(`src/unilab/envs/locomotion/microduck/actuator.py`):

- **Electrical dynamics → per-step control callback.** `MicroduckBamActuator` is
  mounted via `backend.set_pre_step_control(...)`. Every physics substep it reads
  the servo state, runs BAM's firmware voltage-control law + DC-motor back-EMF
  torque equation (vectorized over `(num_envs, 14)` in numpy), and writes the
  result to the `<motor>` actuators' `data.ctrl`. Per-env randomization of battery
  voltage (`vin`), load-dependent voltage sag, back-EMF and Kp lives here.
  The policy action is a **HOME-relative joint-position target**.

- **Friction → baked into the motor MJCF (solver-side).** BAM's friction budget
  (Coulomb + viscous + gearbox) is what provides *static holding* (stiction) for a
  0.8 kg biped on low-Kp servos. It **cannot** be folded into the returned torque —
  the callback never sees the gravity/contact load, so the robot creeps and topples.
  Instead a nominal-load friction constant is written as `dof_frictionloss` /
  `dof_damping` into a generated `*_motor.xml`
  (`scripts/microduck_make_motor_xml.py`), and MuJoCo's constraint solver does the
  stiction.

Because UniLab retrains from scratch against **this** model, train and eval stay
self-consistent; the exported ONNX is what gets cross-checked in `microduck_rl`.

### 2.3 Foot-gait rewards (walking recipe)

The velocity task (`velocity.py` / `rewards.py`) adds foot-clearance and gait
shaping on top of the IMU-only base reward, driven by foot sensors added to the
MJCF (they shape **reward only** — never the 61D obs, so the ONNX contract is
untouched). The current recipe uses an explicit **duty-cycle gait clock**: a
`feet_contact_schedule` reward ties each foot's *contact state* (planted in
stance, airborne in swing) to the clock, which is what forces a real walking
cadence instead of a fast shuffle. See the port plan for the reward-tuning history.

---

## 3. Training environment prep

The task is registered as **`MicroduckVelocityFlat`** (Hydra owner config
`conf/ppo/task/microduck_velocity_flat/mujoco.yaml`).

**Run the tests** (CPU, no accelerator needed) to confirm the install and the
port invariants (61D obs, reward signs, foot sensors, BAM):

```bash
HF_ENDPOINT=https://hf-mirror.com uv run --extra mujoco --extra microduck --no-sync \
  pytest tests/envs/locomotion/microduck/ -q
```

**Smoke test** (always run before a long job — catches most config errors in
seconds): 64 envs, 5 iterations.

```bash
HF_ENDPOINT=https://hf-mirror.com uv run --extra mujoco --extra microduck --no-sync \
  train --algo ppo --task microduck_velocity_flat --sim mujoco \
  algo.num_envs=64 algo.max_iterations=5 training.device=xpu training.no_play=true
```
---

## 4. Training Microduck

Full training run (detached, so it survives the shell). 4096 envs on XPU learner:

```bash
cd ~/work/microduck/UniLab
setsid env HF_ENDPOINT=https://hf-mirror.com \
  uv run --extra mujoco --extra microduck --no-sync \
  train --algo ppo --task microduck_velocity_flat --sim mujoco \
  algo.num_envs=4096 algo.max_iterations=2000 \
  training.device=xpu training.no_play=true \
  > train_microduck.log 2>&1 < /dev/null &
```

**Render an evaluation video** from the latest run (record mode also exports
`policy.onnx` alongside the video):

```bash
HF_ENDPOINT=https://hf-mirror.com uv run --extra mujoco --extra microduck --no-sync \
  eval --algo ppo --task microduck_velocity_flat --sim mujoco \
  --load-run -1 --render-mode record \
  training.device=xpu algo.num_envs=16 training.play_steps=1500
# → logs/rsl_rl_ppo/MicroduckVelocityFlat/<timestamp>_mujoco/play_video.mp4
```

Resume a run with `--load-run <path> --checkpoint <N>`; use `--load-run -1` to pick
the most recent run directory.

---

## Repo map (microduck)

| Path | What |
|------|------|
| `src/unilab/envs/locomotion/microduck/base.py` | `MicroduckBaseEnv` — shared 61D obs / BAM wiring / DR / termination |
| `src/unilab/envs/locomotion/microduck/actuator.py` | BAM XL330 voltage actuator adapter (per-step control callback) |
| `src/unilab/envs/locomotion/microduck/velocity.py` | `MicroduckVelocityEnv` — the walking task + reward config |
| `src/unilab/envs/locomotion/microduck/rewards.py` | microduck reward functions (tracking, foot-gait, duty-cycle) |
| `src/unilab/assets/robots/microduck/` | MJCF models (walk / groundcontact / rollers) + scenes |
| `scripts/microduck_make_motor_xml.py` | generate `*_motor.xml` (bakes friction into the MJCF) |
| `conf/ppo/task/microduck_velocity_flat/mujoco.yaml` | Hydra owner config (env + reward + backend) |
