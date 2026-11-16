# Inference, demo and tests


## Rollout on a held-out episode  [smoke-tested]

```bash
python -m dreamfar.inference.rollout --ckpt dreamfar_v9_step000054000.ckpt --config configs/release/dreamfar_v9.yaml \
    --tokenizer-ckpt tok256_step000200000.ckpt --tokenizer-config configs/release/tok256_ssdd_final_v1.yaml --urdf yam.urdf \
    --latent-root <latent_root> --contact-dir <contact_dir> --episode 0 --regime mid --frames 300 --out outputs/rollouts
```

Seeds the first 10 s of the episode (`--regime mid`; `cold` / `dup` = cold starts from frame 0), rolls the
remaining frames out with the paper's streaming settings (P = 16 Euler steps per frame, logit-normal flow-time grid,
committed-context flow-time 0.2404, memory 0.1, bf16 weights), decodes GT and prediction with one fixed decoder
noise, and writes `<out>/<asset>_<regime>.mp4` (`[prediction | ground truth]`, views stacked) plus a JSON with
PSNR / SSIM / LPIPS per view over the predicted frames and the achieved fps. `--config` accepts the model-zoo
config or a training run's `.hydra/config.yaml`.

## Programmatic use

```python
from dreamfar.inference.build import build_model, decode_noise, decode_frame
from dreamfar.inference.episodes import list_episodes, load_episode
from dreamfar.inference.seeding import build_regime

model = build_model("dreamfar_v9_step000054000.ckpt", "configs/release/dreamfar_v9.yaml", "cuda:0",
                    overrides={"dyn_flow_steps": 16, "infer_t_schedule": "logit_normal", "stream_ctx_noise": 0.2404252730845196,
                               "infer_weights_dtype": "bf16", "tokenizer_ckpt": ..., "tokenizer_config": ..., "ee_urdf_path": ...})
eps, meta = list_episodes("<latent_root>")
batch = load_episode(model, eps[0], meta, "cuda:0", cap=300, contact_dir="<contact_dir>")
z_gt = model.stack_views(batch["latent"]).transpose(0, 1)                     # (1, T, C, 3*16, 16)
kw = build_regime("mid", model, z_gt, model.cond_action(batch["action"]), batch["state"], batch["frame_idx"],
                  contact=batch.get("contact"), seed_sec=10, fps=10)
z_pred, _ = model.rollout(kw["z_hist"], kw["action"], kw["state"], kw["frame_idx"],
                          seed_latents=kw["seed_latents"], seed_contact=kw["seed_contact"], compiled=True)
rgb = decode_frame(model.tokenizer, z_pred[0, -1:], 3, decode_noise(model, "cuda:0"), steps=2)   # (3, 3, 256, 256) in [0,1]
```

`model.rollout` also drives the interactive demo: with `pending_fn` / `on_admit` / `wait_fn` / `on_commit` it admits a
frame only when an action is available, idles with the KV cache frozen, and hands every committed latent off as
soon as it exists.

## Demo  [smoke-tested headless]

A single-page web demo (keyboard and mouse control of the two arms; `dreamfar/demo/`):

```bash
export DEMO_CKPT=dreamfar_v9_step000054000.ckpt DEMO_CONFIG=configs/release/dreamfar_v9.yaml \
       TOKENIZER_CKPT=tok256_step000200000.ckpt TOKENIZER_CONFIG=configs/release/tok256_ssdd_final_v1.yaml \
       URDF_PATH=yam.urdf LATENT_ROOT=<latent_root> CONTACT_DIR=<contact_dir> HELDOUT_PATH=<heldout.json>
CUDA_VISIBLE_DEVICES=0 python -m uvicorn dreamfar.demo.server:app --host 127.0.0.1 --port 8000 \
    --ws-max-size 33554432 --ws-per-message-deflate false
```

Open http://localhost:8000 (tunnel the port from a remote machine). Start-up builds the model and compiles
+ warms the decoder and the dynamics (a few minutes). With two visible GPUs the decoder runs on the second one.
Keyboard: W/S A/D Q/E move the left end-effector, 1/Z 2/X 3/C rotate it, R/F open/close its gripper; I/K J/L U/O,
7/M 8/, 9/., P/; for the right arm. Mouse: left/right click picks an arm, the pointer over the top view sets its
table position, the wheel its height, Space toggles its gripper. Headless check (connects, holds keys, saves
frames): `python -m dreamfar.demo.ws_client --url ws://127.0.0.1:8000/ws --seconds 8 --out outputs/demo_check`.
Hand-tracking and leader-arm input modes of the paper's demo need extra hardware / trackers and are not included.
All environment-variable knobs and the websocket protocol: [`dreamfar/demo/README.md`](../dreamfar/demo/README.md).

## Repository structure

```
configs/                    Hydra configs: config.yaml, algorithm/{tokenizer,dynamics}.yaml, dataset/{frames,latents}.yaml,
                            experiment/{tokenizer,dynamics}.yaml, release/ (model-zoo configs)
dreamfar/tokenizer/         net.py (SsddTokenizer: encode / decode), encoder.py, decoder.py (thin wrappers over SSDD's encoder
                            and U-ViT decoder), losses.py (LPIPS, REPA), module.py (Lightning training), data.py (frame dataset)
dreamfar/dynamics/          dit.py (streaming DiT: training forward, stream_prefill / stream_decode KV cache), attention.py,
                            stream_layout.py (masks, dt-RoPE), mcp.py, module.py (loss, EMA, validation, KV-cached rollout), optim.py
dreamfar/data/              latent_dataset.py (shards, windows, contact-keyframe memory selection), contact.py, video.py (PyAV),
                            precompute_latents.py, ee_fk.py (URDF forward kinematics), rot6d.py, normalizer.py, spec_normalizer.py
dreamfar/inference/         build.py (checkpoint -> model, decode helpers), episodes.py, seeding.py (cold / dup / mid), rollout.py
dreamfar/train/             experiment.py (data + Trainer wiring), train.py (Hydra entrypoint), dump_config.py
dreamfar/demo/              server.py (FastAPI / websocket), index.html, modes.py, ik.py, ws_client.py, README.md
scripts/                    train_tokenizer.sh, train_dynamics.sh (torchrun launchers), install_ssdd.sh
tests/                      test_equivalence.py (bit-exactness against the original code's reference dump)
```

## Tests (bit-exactness)

`tests/test_equivalence.py` compares the release against reference tensors dumped from the original research code
(same checkpoints, same seeds, fp32 eager on one GPU): tokenizer encode / decode (fp32 and bf16) and training loss,
the dynamics state_dict name/shape set, one streaming loss evaluation (flow, state, 3 MCP terms, contact BCE), the
data loader's episode batch and a 120-frame mid-start rollout (eager and torch.compiled, memory selections and
keyframe bank included). All comparisons are `torch.equal` (run with SSDD installed at the pinned commit above).
The reference dump is not distributed.

Run it with:

```bash
python -m tests.test_equivalence --ref equiv_ref.pt --tok-ckpt tok256_step000200000.ckpt \
    --tok-config configs/release/tok256_ssdd_final_v1.yaml --dyn-ckpt dreamfar_v9_step000054000.ckpt \
    --dyn-config configs/release/dreamfar_v9.yaml --latent-root <latent_root> --contact-dir <contact_dir> --urdf yam.urdf
```
