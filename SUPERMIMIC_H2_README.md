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
| Tests | `tests/algos/test_supermimic.py` |

The env reproduces the SuperMimic mjlab env: obs 930 (10 x 93, term-major history), goal 680
(10 x 68), state 71, context xi 33, action `q0 + (a + eps) * 0.25 * effort / kp - encoder_bias`,
BeyondMimic reward x dt, pelvis height / tilt terminations, clip end and 10 s as truncations,
reference-state init with pose / velocity / joint noise, mjlab target-frame timing (reset to `f`
-> target `f + 1`), and the one-substep-stale body kinematics mjlab's reward / terminations see.
The algorithm reads `s`, `xi`, executed actions and target frames through the env's `sm_*`attributes.

## Run

```bash
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

### Background training

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

| Key | Meaning |
|---|---|
| `algo.supermimic.schedule.lr_restart_iteration` | iteration where the LR warm restart starts (normally the resume iteration) |
| `algo.supermimic.schedule.lr_restart_peak_ratio` | restart peak as a fraction of the base LR |
| `algo.supermimic.schedule.lr_restart_warmup_iterations` | linear ramp length, counted from the first update after the restart |
| `algo.supermimic.schedule.warmup_iterations` | data-only iterations after (re)start; use ~1000 when resuming, so the WM is not fitted on a near-empty buffer |

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
  | UniLab XPU, 50k iterations | 0.885 | 0.868 | 0 |
  | UniLab XPU, continued to 100k (`mujoco_xpu_continue100k`) | **0.906** | **0.889** | 0 |
  | `h2_mimic_full.onnx` (50k+ clips, 200k iterations; ceiling) | 0.939 | 0.922 | 0 |

  50k -> 100k: +0.022 per clip on average (333 of 400 clips better), train_ret 0.742 -> 0.762,
  WM nse1 0.0071 -> 0.0063, imag_ret 2.666 -> 2.738. The weakest clips are fast dances (hiphop,
  vouge, slide; 0.56-0.66 at 100k), as in the reference. A fine-tune with fp32 instead of bf16 MLP
  matmuls gave the same results as bf16, so bf16 is not a precision limit here.
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
  2.7x faster. 

### XPU throughput notes (Arc Pro B70, 8192 envs, bf16, Xeon w5-3435X)

The updates are bound by CPU-side kernel launches (~25 us per Triton launch on XPU), not by GPU
FLOPs: update time was flat from batch 1024 to 2048. Changes, measured on the same data
(400 iterations, `mujoco_xpu`, 8192 envs): 0.84 it/s -> 1.70 it/s (sequential) -> 2.41 it/s
(`pipeline`). With 4096 envs the optimized run reaches ~2.98 it/s (about 10 GB of VRAM).

* `kinematics.py`: FK no longer compares the pybind `mjtJoint` enum inside the body loop (it made
  `torch.compile` graph-break ~200 times per step) and runs one batched step per tree depth
  instead of one per body.
* `imagination.py`: policy forward, history update and sensor noise are compiled together with the
  world-model step; goal windows, root-error features and survival are computed for all steps at
  once. `runner.py` enables inductor `combo_kernels` on XPU (launches per step 574 -> 182).
* `replay_buffer.py`: the window / start sampling mask is built once per buffer state instead of
  once per batch (it scanned all 8192 x 1000 rows ten times per update).
* `schedule.pipeline: true` (on in `mujoco_xpu`): updates run in a learner thread while the main
  thread simulates, so the CPU simulation overlaps the GPU updates. Collection acts with a policy
  at most `schedule.pipeline_lag` (1) updates old; the update-to-data ratio is unchanged. The
  console `t sim/wm/pi` timers are then concurrent, not additive. Set `pipeline=false` to get the
  strictly sequential loop back.

### Platform comparison (Panther Lake iGPU, Arc Pro B70, RTX 5090)

Batch 2048, UTD 0.4, H = 32 for all columns. Panther Lake is the unoptimized baseline; the B70 runs
include the optimizations above. The 5090 column is the GR2 run from `GR2_TRAINING_README.md` of
SuperMimic-Reproduced (mjlab / MuJoCo Warp, GPU simulation, 51.7 k clips); the two XPU columns run
H2 with MuJoCo on the CPU and 5.4 k clips, so this is not a like-for-like comparison.

| | Panther Lake iGPU | Arc Pro B70, 4096 envs | Arc Pro B70, 8192 envs | RTX 5090, 4096 envs |
|---|---|---|---|---|
| Simulation step | 0.15 s (12-core CPU) | 0.15 s (32-thread Xeon, overlapped) | 0.25 s (0.33 s overlapped) | on GPU, not split out |
| World-model update | ~1.1 s | ~0.39 s | ~0.40 s (was ~0.50) | n/a |
| Policy + critic update | ~2.1 s | ~0.44 s | ~0.45 s (was ~1.87) | n/a |
| Throughput | 0.65-0.71 it/s | 2.98 it/s | 2.41 it/s (was 0.84) | ~4.4 it/s |
| Env steps / s | ~2.8 k | ~12.2 k | ~19.7 k (was 6.9 k) | ~18.0 k |
| 50k iterations | ~21 h | ~4.7 h | ~5.8 h | ~3.2 h |
| Memory | ~24 of 30 GB (shared) | ~10 GB VRAM | ~12 GB VRAM (before opt.) | 17.7 GB VRAM |

At 4096 envs the B70 is ~4.4x Panther Lake and ~68% of the 5090 in it/s; 8192 envs trade it/s for
~60% more env steps/s. 50k-iteration times for the B70 and the 5090 are extrapolated from it/s (the
5090 run took 14.8 h for 200k iterations).

bf16 vs fp32 MLP matmuls (measured on Panther Lake, 4096 envs, same replay data): updates 2.7x faster and end-to-end
~0.65-0.71 vs ~0.39 it/s (~1.7x), with no loss of accuracy (WM eval nse1 0.0810 vs 0.0813, nse8 0.165 vs 0.164; an fp32 fine-tune
matched the bf16 results). The fp32 run was not repeated on the B70, so its bf16 gain there is not measured.
