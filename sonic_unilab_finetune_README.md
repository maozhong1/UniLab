# SONIC G1 — UniLab Warm-Start Finetune & Deploy-Split ONNX

Operational guide for finetuning the sonic G1 encoder+decoder on Intel XPU in UniLab,
and exporting the deploy-split ONNX (encoder / decoder) for NPU deployment.

All commands run from the UniLab repo root:
```bash
cd /home/maozhong/work/my_sonic/UniLab
```
Python always via `uv run --no-sync python …` (the `--no-sync` is important — a bare
`uv run`/`uv sync` reverts torch to the `+cu128` build and XPU disappears).

---

## 0. What this trains

- **Env** `G1SonicMotionTracking` (`src/unilab/envs/motion_tracking/g1/tracking_sonic.py`):
  emits a 1570-dim actor obs = encoder-input(640) ++ proprio(930), byte-compatible with
  sonic `last.pt` (frame-major enc, MuJoCo→IsaacLab joint permutation, action remap).
- **Actor** `SonicG1ActorModel` = G1 encoder(640→64) + FSQ(2×32) + g1_dyn decoder(994→29),
  warm-started from `sonic_release/last.pt`. Critic = stock 286-dim MLP (fresh).
- **Config** `conf/ppo/task/g1_motion_tracking/sonic.yaml` (warm-start is the default there).

Byte-compat is numerically verified (`scripts/sonic/parity_harness.py`): encoder max|Δ|=0,
decoder ~3e-6 vs the deployed release ONNX.

---

## 1. Prerequisites

- Intel GPU (`/dev/dri/renderD128`), oneAPI at `/opt/intel/oneapi` (sourcing NOT needed —
  the torch `+xpu` wheel bundles its runtime).
- `train_warm.sh` auto-restores the `+xpu` wheel if the venv reverted. Manual check:
  ```bash
  uv run --no-sync python -c "import torch; print(torch.__version__, torch.xpu.is_available())"
  # want: 2.7.0+xpu True    — if False:
  uv pip install "torch==2.7.0+xpu" "pytorch-triton-xpu" --index-url https://download.pytorch.org/whl/xpu
  ```

---

## 2. Data prep (only when adding/changing clips)

7 walk clips already exist in `/home/maozhong/work/sonic_vla_infer/bones_seed_subset/npz/`.
The g1 CSVs are already extracted at `bones-seed/g1/csv/` — **do NOT re-run `tar` on g1.tar.gz.**

```bash
# a) pick clips (clean forward walks = basename starts walk_ff, NOT *_M.csv):
mkdir -p /home/maozhong/work/sonic_vla_infer/bones_seed_subset/csv
cp /home/maozhong/work/sonic_vla_infer/bones-seed/g1/csv/*/walk_ff_loop*.csv \
   /home/maozhong/work/sonic_vla_infer/bones_seed_subset/csv/

# b) CSV → NPZ (120→50 Hz, cm→m):
uv run --no-sync python scripts/motion/bones_seed_csv_to_npz.py \
  --input  /home/maozhong/work/sonic_vla_infer/bones_seed_subset/csv \
  --output /home/maozhong/work/sonic_vla_infer/bones_seed_subset/npz \
  --input_fps 120 --output_fps 50

# c) alignment self-check (robocasa-style guard; expect VERDICT: STABLE, pelvis ~0.7 m):
uv run --no-sync python scripts/sonic/openloop_replay.py \
  --motion_file /home/maozhong/work/sonic_vla_infer/bones_seed_subset/npz/<clip>.npz --steps 300
```

**How much data:** warm-start from an already-general `last.pt` needs far less than from
scratch. Single skill 20–100 clips; skill family (walk/turn/start-stop/speeds) 200–1000;
domain adaptation 1000–5000. Always pass clips through the STABLE check first.

---

## 3. Train

### Step 1 — Smoke (~10 s, always do first)
```bash
./scripts/sonic/train_warm.sh smoke
```
Healthy = `Mean action std: 0.05` and iter0 `Mean reward` clearly > 0 (warm-start starts
high). Confirms last.pt loads strict + obs/action alignment is consistent.

### Step 2 — Full run (durable, ~1 h for 512×1000)
**Always launch with `setsid`** so it survives the terminal/session closing:
```bash
LOG=/tmp/sonic_train/warm_$(date +%Y%m%d_%H%M%S).log
setsid nohup ./scripts/sonic/train_warm.sh run > "$LOG" 2>&1 < /dev/null &
echo "$LOG"
```
Defaults: 512 envs · 1000 iters · save every 200 · init_std 0.05 · lr 1e-4 · entropy 1e-3.

### Step 3 — Monitor
```bash
tail -f "$LOG"
# or:
uv run --no-sync tensorboard --logdir logs/rsl_rl_ppo/G1SonicMotionTracking/
```
Expected curve: high start → small cold-critic dip over the first ~25 iters (critic is NOT
warm-started) → recovers and climbs. If reward keeps sinking past ~150 iters, lower `lr` or
set `FREEZE_ENCODER=true`. Reference run: 1000 iters ≈ 61 min, final mean reward ~+0.64.

### Stop
```bash
pkill -f 'train_rsl_rl.py.*pretrained_ckpt'
```

---

## 4. Tuning knobs (env-var overrides to `train_warm.sh`)

| Var | Default | Meaning |
|---|---|---|
| `NPZ_DIR` | `bones_seed_subset/npz` | training motions (all `*.npz` in the dir) |
| `CKPT` | `sonic_release/last.pt` | warm-start weights; set a `model_N.pt` to resume |
| `NUM_ENVS` | 512 (smoke 16) | parallel envs; CPU has headroom → 1024 speeds up |
| `ITERS` | 1000 (smoke 2) | iterations; warm-start 500–2000 is plenty |
| `SAVE` | 200 | checkpoint interval |
| `FREEZE_ENCODER` | false | **true = finetune DECODER ONLY (freezes the 64-d FSQ token space → VLA-compatible)** |
| `DEVICE` | xpu | `cpu` to run without a GPU |

**Rollout:** `num_steps_per_env: 24` (`conf/ppo/config.yaml`) → 512 × 24 = **12,288 samples/iter**.

**Example — VLA-safe variant (decoder-only), more clips, 1500 iters:**
```bash
LOG=/tmp/sonic_train/warm_$(date +%Y%m%d_%H%M%S).log
FREEZE_ENCODER=true ITERS=1500 NUM_ENVS=1024 \
  setsid nohup ./scripts/sonic/train_warm.sh run > "$LOG" 2>&1 < /dev/null &
```

**Revert to from-scratch (no warm-start)** — override on the CLI (or use the from-scratch path):
```bash
uv run --no-sync python scripts/train_rsl_rl.py task=g1_motion_tracking/sonic \
  training.device=xpu training.no_play=true \
  algo.actor.pretrained_ckpt=null algo.actor.distribution_cfg.init_std=1.0 \
  env.action_output_isaaclab_to_mujoco=false \
  '+env.motion_file=[<npz1>,<npz2>,...]' algo.num_envs=512 algo.max_iterations=1000
```

---

## 5. Checkpoints & deploy-split ONNX export

Checkpoints land in `logs/rsl_rl_ppo/G1SonicMotionTracking/<timestamp>_mujoco/model_{0,200,…,N-1}.pt`.

Export the two deploy graphs from a finetuned checkpoint (or from `last.pt`):
```bash
RUN=$(ls -dt logs/rsl_rl_ppo/G1SonicMotionTracking/*/ | head -1)
uv run --no-sync python scripts/sonic/export_deploy_onnx.py \
  --ckpt "$RUN/model_999.pt" --out-dir "$RUN/deploy_onnx" --verify
```
Produces (opset 13, FLOAT, FSQ Round in-graph):
- `model_encoder_g1.onnx` — `obs_dict[1,640] → encoded_tokens[1,64]`  (**g1-only**, see note)
- `model_decoder.onnx`    — `obs_dict[1,994] → action[1,29]`  (994 = token64 ‖ proprio930; matches deploy exactly)

`--ckpt` auto-detects rsl_rl (`actor_state_dict['core.*']`) vs raw last.pt (`policy_state_dict`).
`--verify` checks the exported ONNX vs the torch core (< 1e-4) and, for last.pt, bit-exact vs
the deployed release ONNX.

---

## 6. Deploy notes / open items

- **Encoder is g1-only 640→64**, not the deployed merged 1762→64 (we lack teleop/smpl branches).
  The deploy side must feed the g1 640-obs slice — the env already assembles it. The decoder
  994→29 matches the deploy interface exactly.
- **VLA red line:** finetuning with `FREEZE_ENCODER=false` **redefines the 64-d FSQ token space**,
  so an existing VLA (trained on the old token space) will no longer drive the new decoder. Use
  `FREEZE_ENCODER=true` (decoder-only) to keep the token space fixed, or plan to retrain the VLA.
- **g1_recon aux loss:** the `g1_kin` head + `SonicG1ActorModel.kin_recon()` exist but no PPO loop
  consumes them; enabling requires a HIM-PPO-style fork. Not required for a first deploy.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.xpu.is_available()` False / trains on CPU | venv reverted to `+cu128`; re-install the `+xpu` wheel (§1). Always use `uv run --no-sync`. |
| Hydra `Key 'motion_file' is not in struct` | use `+env.motion_file=[...]` (the `+`); `train_warm.sh` already does. |
| Run dies when terminal/session closes | launch with `setsid nohup … &` (§3 Step 2), not plain `nohup`. |
| Warm-start reward starts low (~from-scratch) | check `mujoco_to_isaaclab_perm=True` + `env.action_output_isaaclab_to_mujoco=true` + `init_std=0.05`. |
| Reward keeps sinking past ~150 iters | lower `lr`, or `FREEZE_ENCODER=true`. |

Related docs: `~/work/my_sonic/sonic_v2_bytecompat.md`, `sonic_onnx_interface_spec.md`,
`sonic_g1_to_unilab_mapping.md`; scripts under `scripts/sonic/`.
