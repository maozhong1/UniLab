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

# Benchmark (eval protocol: no noise / DR, friction 1, start at the first frame)
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

Data: `conf/supermimic/task/h2_supermimic/mujoco.yaml` (and `mujoco_xpu.yaml`) points `env.motion_dirs` at the two
BONES-SEED H2 folders (1946 + 3515 clips, no overlap, 33 bodies in MuJoCo order, 50 fps).

## Validation so far

* SuperMimic-trained `h2_mimic_full.onnx` (unchanged) in the UniLab env, 100 random clips of
  both folders: mean normalised return 0.939, step-weighted 0.898, 0 falls (reference: 0.894 /
  0.869 on 400 clips with mjlab). This checks obs / goal layout, action mapping, physics,
  reward, terminations and frame timing end to end.
* Torch versus simulator side (tests): FK = MuJoCo `xpos/xquat/objectVelocity`, analytic
  observation / goal = env observation, torch reward = env reward (all 8 terms), npz reference
  bodies = FK of the reference state, zero-initialised world model = constant-velocity step.
* CPU smoke training: world-model 1-step / 8-step errors fall, imagined survival rises.
* Sim throughput (12-core laptop CPU, mujoco backend): 4096 envs 0.118 s/step, 1024 envs
  0.035 s/step. The env simulates a collision-only copy of the scene
  (`strip_visual_meshes: true`: no visual meshes / skybox, collision meshes reduced to their
  convex-hull vertices; 187 MB -> 2 MB per env model, which the backend clones per env because
  of the friction / COM randomisation). Set it to false for rendering.
* Intel XPU (Core Ultra X7 358H iGPU, `mujoco_xpu` profile, full data, 4096 envs): 0.74 it/s
  (sim 0.15 s, world-model update 0.89 s, policy update 2.1 s, 0.4 updates per iteration), about
  19 h for 50k iterations; ~24 GB of the 30 GB system RAM in use (iGPU shares it). bf16 versus
  fp32 MLP matmuls on identical replay data: WM eval nse1 0.0810 vs 0.0813, nse8 0.165 vs 0.164,
  2.7x faster. First 300 iterations: WM nse1 0.49 -> 0.10, imag_ret 0.09 -> 0.43, train_ret 0.42 -> 0.51.
