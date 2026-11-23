# SuperMimic (world model + BPTT) on Unitree H2 in UniLab

Validation port of the SuperMimic reference implementation
(`SuperMimic-Reproduced`, mjlab / MuJoCo Warp) to UniLab, for the fixed-head 29-DoF H2
(`src/unilab/assets/robots/h2/h2.xml`, identical to SuperMimic's H2 asset).

## Layout

| Part | Location |
|---|---|
| Env `H2SuperMimic` (mujoco backend) | `src/unilab/envs/motion_tracking/h2/supermimic.py` |
| Reference library + return-based reset sampler (NumPy) | `src/unilab/envs/motion_tracking/h2/supermimic_library.py` |
| Scene with SuperMimic solver settings (pyramidal cone, impratio 1, 10/20 iterations) | `src/unilab/assets/robots/h2/scene_flat_supermimic.xml` |
| Algorithm (torch): FK, state spec, world model, analytic heads, replay, imagination, trainers, runner | `src/unilab/algos/torch/supermimic/` |
| Training entry (Hydra root `conf/supermimic`) | `scripts/train_supermimic.py` |
| Detached XPU training launcher (nohup) | `scripts/supermimic/train_h2_xpu_nohup.sh` |
| Benchmark (checkpoint or ONNX) / ONNX export | `scripts/supermimic/{eval_benchmark,export_onnx}.py` |
| Rollout recording / offscreen video rendering | `scripts/supermimic/{record_rollout,render_rollout}.py` |
| bf16 vs fp32 policy-gradient check | `scripts/supermimic/check_bf16_grad.py` |
| Tests | `tests/algos/test_supermimic.py` |

The env reproduces the SuperMimic mjlab env: obs 930 (10 x 93, term-major history), goal 680
(10 x 68), state 71, context xi 33, action `q0 + (a + eps) * 0.25 * effort / kp - encoder_bias`,
BeyondMimic reward x dt, pelvis height / tilt terminations, clip end and 10 s as truncations,
reference-state init with pose / velocity / joint noise, mjlab target-frame timing (reset to `f`
-> target `f + 1`), and the one-substep-stale body kinematics mjlab's reward / terminations see.
The algorithm reads `s`, `xi`, executed actions and target frames through the env's `sm_*`
attributes. Not ported: commit `24b8b61` of the reference (robot-aware feasibility filter,
`normalize_bin_return`), multi-GPU, `anchor_obs`. Clips are filtered by name only
(`motion_exclude_keywords: [table, _box_]`, the only clips the reference H2 filter dropped).

## Run

```bash
# CPU smoke run (pipeline check: 32 envs, 20 clips, small nets)
uv run scripts/train_supermimic.py task=h2_supermimic/mujoco_cpu_smoke

# Full H2 run (reference H2 command: 4096 envs, policy batch 2048, 50k iterations, ckpt every 3000)
uv run scripts/train_supermimic.py training.device=cuda algo.supermimic.wm_trainer.compile=default \
    algo.supermimic.policy_trainer.compile=default

# Intel XPU (torch 2.7.0+xpu): install once with `make sync-xpu`, then always `uv run --no-sync`.
# Profile: 4096 envs, bf16 MLP matmuls, torch.compile, same data / schedule as the reference H2 run.
uv run --no-sync scripts/train_supermimic.py task=h2_supermimic/mujoco_xpu

# Benchmark (eval protocol: no noise / DR, friction 1, start at the first frame).
# --sample N picks N random clips (seed 0); --clip-list FILE (one npz stem per line) fixes the set,
# so different checkpoints are compared on the same clips. --onnx <file> instead of --checkpoint.
uv run scripts/supermimic/eval_benchmark.py --checkpoint logs/supermimic/H2SuperMimic/<run>/ckpt_latest.pt \
    --npz-dir ~/work/sonic_vla_infer/bones-seed/selections/upright_nzp_h2_29dof_new \
    --npz-dir ~/work/sonic_vla_infer/bones_seed_3k_new/npz_h2_29dof_new --sample 400 --out eval.json
uv run scripts/supermimic/export_onnx.py --checkpoint logs/supermimic/H2SuperMimic/<run>/ckpt_latest.pt
```

### Background training (survives SSH disconnects)

The launcher detaches the run from the terminal session with `setsid nohup`, so SSH / VSCode
disconnects or closing the terminal do not stop it. The log goes to
`logs/supermimic/train_h2_xpu_<time>.log`, the PID to `logs/supermimic/train_h2_xpu.pid`.

```bash
cd /home/maozhong/work/tmp/myUnilab
scripts/supermimic/train_h2_xpu_nohup.sh                                                  # full run: 50k iterations
scripts/supermimic/train_h2_xpu_nohup.sh algo.supermimic.schedule.total_iterations=10000  # shorter first run

tail -f logs/supermimic/train_h2_xpu_*.log | grep -v RuntimeWarning   # progress (also after reconnecting)
tensorboard --logdir logs/supermimic/H2SuperMimic                      # curves
kill $(cat logs/supermimic/train_h2_xpu.pid)                            # stop
scripts/supermimic/train_h2_xpu_nohup.sh algo.resume=logs/supermimic/H2SuperMimic/<run>/ckpt_latest.pt  # resume
```

The first minute or so goes into `torch.compile`; no log output during that time is expected.

### Resume and continue training

`algo.resume=<ckpt>` restores the world model, policy, critic, their Adam states, the
reset-sampler statistics and the iteration counter. The replay buffer is not saved; it is
refilled from the resumed iteration on, and no update runs during the first
`schedule.warmup_iterations` iterations (40 by default).

* **Resume an interrupted run**: same task, `algo.resume=.../ckpt_latest.pt` (see above). The LR
  follows the original cosine schedule, which is indexed by the absolute iteration.
* **Continue a finished run** (e.g. 50k -> 100k): use a fine-tune task. Raising only
  `total_iterations` would stretch the cosine schedule and make the LR jump from its 3% floor to
  ~30% in one step. `h2_supermimic/mujoco_xpu_continue100k` instead refills the replay buffer
  for 1000 data-only iterations, ramps the LR from 3% to 25% of the base LR over 2000 iterations,
  decays it by cosine back to 3% (~89.7k) and keeps it there until 100k:

```bash
# A task=... argument replaces the launcher's default task.
scripts/supermimic/train_h2_xpu_nohup.sh task=h2_supermimic/mujoco_xpu_continue100k \
    algo.resume=logs/supermimic/H2SuperMimic/<run>/ckpt_0050000.pt
```

The runner writes `ckpt_<iteration>.pt` only at multiples of `checkpoint_interval`; the final
state goes to `ckpt_latest.pt` only (e.g. 50000 with interval 3000). Copy it to
`ckpt_0050000.pt` before starting another run from it, because `ckpt_latest.pt` is overwritten.

Fine-tuning knobs (all off by default, base runs are unchanged):

| Key | Meaning |
|---|---|
| `algo.supermimic.schedule.lr_restart_iteration` | iteration where the LR warm restart starts (normally the resume iteration) |
| `algo.supermimic.schedule.lr_restart_peak_ratio` | restart peak as a fraction of the base LR |
| `algo.supermimic.schedule.lr_restart_warmup_iterations` | linear ramp length, counted from the first update after the restart |
| `algo.supermimic.schedule.warmup_iterations` | data-only iterations after (re)start; use ~1000 when resuming, so the WM is not fitted on a near-empty buffer |
| `env.sampler.boost_keywords`, `env.sampler.boost_factor` | multiply the reset probability of bins of clips whose name contains a keyword (e.g. `[dance]`, `3.0`) |

The console line prints the current LR (`lr ...`). During the high-LR part of a restart the
benchmark dips (about -0.02 around the peak) and recovers once the LR decays; judge a restart
by the checkpoints after the decay, not in the middle of it.

### Record and render rollouts

`record_rollout.py` plays each clip once under the benchmark protocol (checkpoint or ONNX) and
stores per-step robot and reference `qpos`; `render_rollout.py` renders them offscreen to one mp4
per clip (solid robot = policy, translucent blue ghost = reference; label = clip name, normalised
return, time) and can concatenate them. Rendering runs on the CPU (`MUJOCO_GL=osmesa`, the
default; EGL does not initialise on the XPU box), so it does not compete with training on the iGPU.

```bash
NPZ=~/work/sonic_vla_infer/GR00T-WholeBodyControl-ov/gear_sonic_deploy/h2/npz_new   # 8 showcase clips
uv run --no-sync scripts/supermimic/record_rollout.py \
    --checkpoint logs/supermimic/H2SuperMimic/<run>/ckpt_0100000.pt --npz-dir $NPZ \
    --out media/h2_rollouts.npz                       # or --onnx <file>; --clips <stem> ... to pick clips
MUJOCO_GL=osmesa uv run --no-sync scripts/supermimic/render_rollout.py \
    --traj media/h2_rollouts.npz --out-dir media/h2_out --tag "ckpt 100000" --concat media/h2_all.mp4
```

Render options: `--seconds N` (trim each clip), `--clips <stem> ...`, `--distance` (3.2),
`--azimuth` (140), `--elevation` (-14), `--width/--height` (1280x720), `--fps` (25), `--no-ghost`.
The ghost is drawn at the reference's world pose, so an offset between robot and ghost shows the
accumulated root drift. Two policies side by side (same clips, rendered into two folders):

```bash
mkdir -p media/sbs; for f in media/a/*.mp4; do n=$(basename $f); ffmpeg -y -loglevel error -i media/a/$n -i media/b/$n \
  -filter_complex "[0:v]scale=960:540[a];[1:v]scale=960:540[b];[a][b]hstack=inputs=2" \
  -c:v libx264 -pix_fmt yuv420p -crf 20 media/sbs/$n; done
```

Data: `conf/supermimic/task/h2_supermimic/mujoco.yaml` (and `mujoco_xpu.yaml`) points `env.motion_dirs` at the two
BONES-SEED H2 folders (1946 + 3515 clips, no overlap, 33 bodies in MuJoCo order, 50 fps).

## Validation so far

* SuperMimic-trained `h2_mimic_full.onnx` (unchanged) in the UniLab env, 100 random clips of
  both folders: mean normalised return 0.939, step-weighted 0.898, 0 falls. This checks obs /
  goal layout, action mapping, physics, reward, terminations and frame timing end to end. Note
  that this ONNX comes from a larger run (50k+ clips, 200k iterations), not from the reference
  README's 50k run; use it as a ceiling, not as the baseline of a 50k run.
* Training results, `mujoco_xpu` (bf16), 400 random clips of the filtered 5408 (seed 0, same set
  for every row; the reference README's 400-clip set is a different draw):

  | Policy | Mean | Step-weighted | Falls |
  |---|---|---|---|
  | Reference README, 50k iterations (CUDA, mjlab) | 0.894 | 0.869 | 2 |
  | UniLab XPU, 50k iterations | 0.885 | 0.868 | 0 |
  | UniLab XPU, continued to 100k (`mujoco_xpu_continue100k`) | **0.906** | **0.889** | 0 |
  | `h2_mimic_full.onnx` (50k+ clips, 200k iterations; ceiling) | 0.939 | 0.922 | 0 |

  50k -> 100k: +0.022 per clip on average (333 of 400 clips better), train_ret 0.742 -> 0.762,
  WM nse1 0.0071 -> 0.0063, imag_ret 2.666 -> 2.738. The weakest clips are fast dances (hiphop,
  vouge, slide; 0.56-0.66 at 100k), as in the reference. A fine-tune with fp32 instead of bf16 MLP
  matmuls gave the same results as bf16, so bf16 is not a precision limit here.
* Torch versus simulator side (tests): FK = MuJoCo `xpos/xquat/objectVelocity`, analytic
  observation / goal = env observation, torch reward = env reward (all 8 terms), npz reference
  bodies = FK of the reference state, zero-initialised world model = constant-velocity step.
* CPU smoke training: world-model 1-step / 8-step errors fall, imagined survival rises.
* Sim throughput (12-core laptop CPU, mujoco backend): 4096 envs 0.118 s/step, 1024 envs
  0.035 s/step. The env simulates a collision-only copy of the scene
  (`strip_visual_meshes: true`: no visual meshes / skybox, collision meshes reduced to their
  convex-hull vertices; 187 MB -> 2 MB per env model, which the backend clones per env because
  of the friction / COM randomisation). Set it to false for rendering.
* Intel XPU (Core Ultra X7 358H iGPU, `mujoco_xpu` profile, full data, 4096 envs): 0.65-0.71 it/s
  over a full run (sim 0.15 s, world-model update ~1.1 s, policy update ~2.1 s, 0.4 updates per
  iteration), about 21 h for 50k iterations; ~24 GB of the 30 GB system RAM in use (iGPU shares
  it). With fp32 MLP matmuls: ~0.39 it/s. bf16 versus
  fp32 MLP matmuls on identical replay data: WM eval nse1 0.0810 vs 0.0813, nse8 0.165 vs 0.164,
  2.7x faster. First 300 iterations: WM nse1 0.49 -> 0.10, imag_ret 0.09 -> 0.43, train_ret 0.42 -> 0.51.
