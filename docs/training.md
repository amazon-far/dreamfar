# Training

All launchers are `torchrun` wrappers; every extra argument is a Hydra override (configs under `configs/`).
Single node = the commands below; multi-node = the same command on every node with
`NUM_NODES=<n> NODE_RANK=<r> MASTER_ADDR=<rank-0 ip> MASTER_PORT=29500` (not smoke-tested beyond 1 node).
A run is named with `name=<run>`; checkpoints go to `outputs/checkpoints/<run>/` (`last.ckpt` + the latest
step) and **relaunching the same command auto-resumes** from `last.ckpt` (`resume=scratch` to ignore it,
`resume=<path>` to pick a file). Logging: `wandb.mode=online wandb.project=... wandb.entity=...`.

## Tokenizer  [smoke-tested: 8xA100, 50 steps]

```bash
GPUS=8 bash scripts/train_tokenizer.sh name=tok256 \
    dataset.raw_root=<raw_root> dataset.heldout_path=<heldout.json> \
    algorithm.encoder_checkpoint=<F16C16_M_256.safetensors> algorithm.decoder_checkpoint=<F16C16_M_256.safetensors> \
    algorithm.repa.lingbot_weights_dir=<lingbot small weights>
```

The released run: 112 H200, per-GPU batch 48 (global 5,376), 200k steps, RAdamScheduleFree (1e-4 new params / 1e-5
pretrained), bf16-mixed, grad clip 0.5, losses `diffusion 1 + lpips 0.25 + repa 0.1 (10k-step warm-up) + uniformity 0.1`,
random view + horizontal flip, SSDD `F16C16_M_256` init ([facebook/SSDD](https://github.com/facebookresearch/SSDD)).
The smoke test ran exactly this launcher with `algorithm.repa=null` (no teacher weights on the box), random init,
`experiment.training.max_steps=50 experiment.training.batch_size=8`.

## Dynamics  [smoke-tested: full-size DiT on 8xH200 (20 steps + resume), reduced DiT on 8xA100-40GB (see below)]

```bash
GPUS=8 bash scripts/train_dynamics.sh name=dreamfar_v9 \
    dataset.latent_root=<latent_root> dataset.contact_dir=<contact_dir> dataset.heldout_path=<heldout.json> \
    dataset.normalizer_path=<normalizer.pt> \
    algorithm.tokenizer_ckpt=tok256_step000200000.ckpt algorithm.tokenizer_config=configs/release/tok256_ssdd_final_v1.yaml \
    algorithm.ee_urdf_path=yam.urdf
```

Defaults (`configs/algorithm/dynamics.yaml`, `configs/experiment/dynamics.yaml`) are the released recipe:
DiT 2048 wide / 28 deep / 16 heads x 128, patch 2, joint spatio-temporal attention at layers [3, 9, 15, 21, 27],
window 50 frames at 10 Hz, memory 8 slots from a pool of 30 contact keyframes, MCP depth 3, AdamW lr 1e-3
(1k warm-up, wd 0.01 on matrices only, eps 1e-10), EMA 0.999, bf16-mixed, grad clip 0.5, per-GPU batch 8.
The released run used 256 GPUs (global batch 2048) warm-started from an earlier run
(`algorithm.init_dynamics_ckpt=<ckpt>`, weights only). The full-size smoke test (8x H200, random init) ran the command
above with `experiment.training.batch_size=2 experiment.training.max_steps=20` (about 8 min including compilation;
a full checkpoint with optimizer + EMA state is 35 GB); the A100-40GB smoke test used
`algorithm.dit.dim=1024 algorithm.dit.depth=12 algorithm.dit.attn_heads=8 algorithm.dit.full_attn_layers=[3,7,11]
experiment.training.batch_size=2`. Resume was tested on both by relaunching the same command with a larger
`experiment.training.max_steps`.

In-training validation logs the teacher-forced flow loss per noise bucket and, on a few batches, the latent MSE /
cosine of a cold-start streaming rollout.

## Released checkpoints

Download links and sha256s are in the README's [Download](../README.md#download) section.

| Model | Config | Checkpoint | Size | Notes |
|---|---|---|---|---|
| Tokenizer `tok256` (step 200k) | [`configs/release/tok256_ssdd_final_v1.yaml`](../configs/release/tok256_ssdd_final_v1.yaml) | `tok256_step000200000.ckpt` (386 MB, weights only) | 46M enc + 50M dec | 256x256 -> 16x16x16 unit-sphere latent; averaged (ScheduleFree) weights |
| Dynamics `v9` (step 54k) | [`configs/release/dreamfar_v9.yaml`](../configs/release/dreamfar_v9.yaml) | `dreamfar_v9_step000054000.ckpt` (4.4 GB) | 2.19B trained, 1.93B deployed | EMA weights, bf16; MCP heads dropped at inference |
| Action/state normalizer | - | `spec_ee_joint_normalizer.pt` | - | spec-derived (`dreamfar.data.spec_normalizer`); also baked into the dynamics checkpoint |

The dynamics checkpoint loads with no key remapping (`dreamfar.inference.build.build_model` checks the exact
tensor name/shape set). The `dreamfar_v9.yaml` paths (`tokenizer_ckpt`, `tokenizer_config`, `ee_urdf_path`,
`dataset.*`) are placeholders: pass the real ones on the command line (shown below).

You also need the **YAM robot URDF** (`yam.urdf`, from the robot vendor's description package) for the
end-effector forward kinematics used by memory retrieval, the demo and the latent precompute.
