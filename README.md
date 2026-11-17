<div align="center">

# DreamFAR: Real-Time and Long-Horizon Interaction with Diverse Objects

[Yiming Dou](https://www.yimingdou.com/)<sup>1,2</sup>,
[Botao He](https://bottle101.github.io/)<sup>1,3</sup>,
[Antonio Loquercio](https://antonilo.github.io/)<sup>4</sup>,
[Andrew Owens](https://andrewowens.com/)<sup>2</sup>,
[Jitendra Malik](https://people.eecs.berkeley.edu/~malik/)<sup>1</sup>,
[Sumedh Anand Sontakke](https://sumedh7.github.io/)<sup>1</sup>,
[Ruoshi Liu](https://ruoshiliu.github.io/)<sup>1,3</sup>

<sup>1</sup>Amazon FAR &nbsp; <sup>2</sup>Cornell University &nbsp; <sup>3</sup>University of Maryland, College Park &nbsp; <sup>4</sup>University of Pennsylvania

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2611.22001)
[![Project page](https://img.shields.io/badge/Project-page-blue.svg)](https://amazon-far.github.io/dreamfar/)
[![Code](https://img.shields.io/badge/Code-GitHub-black.svg)](https://github.com/amazon-far/dreamfar)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

<img src="assets/teaser.jpg" alt="DreamFAR teaser" width="90%"/>

</div>

DreamFAR is an action-conditioned world model for bimanual robot manipulation whose dynamics are trained from
scratch on ~20k hours of robot-native teleoperation (XDOF, 3 cameras). Given the commanded end-effector actions,
it predicts all three camera views in real time and stays consistent over minutes of interaction.

* **Real-time:** teleoperate inside the world model at 10 Hz on one GPU (streaming DiT with a KV cache).
* **Long-horizon:** minutes of autoregressive rollout; a contact-keyframe memory keeps objects consistent.
* **Diverse objects:** generalization across 1000+ tasks and transfer to unseen tasks.
* **Two stages:** an image tokenizer (a 256×256 image → 16×16×16 unit-sphere latent) and a 2.2B-parameter
  dynamics DiT over the three view-stacked latents.

## Installation

Python ≥ 3.11 and one NVIDIA A100, H100 or H200 (tested: PyTorch 2.7.1 + CUDA 12.6 on A100-40GB and H200).
Ada-generation GPUs (L40S, RTX 4090) are not supported: the model's Triton kernels need more shared memory per SM
than they have.

```bash
git clone https://github.com/amazon-far/dreamfar.git && cd dreamfar
python3.11 -m venv .venv && source .venv/bin/activate         # or: uv venv --seed --python 3.11 .venv
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126   # or cu128
pip install -r requirements.txt
bash scripts/install_ssdd.sh          # facebook/SSDD at a pinned commit, editable, into third_party/SSDD
pip install -e .
```

## Download

Checkpoints and sample scenes are on Hugging Face: [Amazon-FAR/dreamfar](https://huggingface.co/Amazon-FAR/dreamfar).
They are licensed under **CC BY-NC 4.0** (the tokenizer was initialized from SSDD's CC BY-NC weights; the scenes come
from the XDOF teleoperation data). The code in this repository is Apache-2.0.

| file | what | size | sha256 |
|---|---|---|---|
| `tok256_step000200000.ckpt` | image tokenizer: a 256×256 image → 16×16×16 unit-sphere latent | 386 MB | `1442f5a7e3deb647a6fb14c97b825388e372aa8d0a19828c6dd058f3f167cde0` |
| `dreamfar_v9_step000054000.ckpt` | dynamics model (paper checkpoint, EMA weights, bf16) | 4.4 GB | `7c258a722f40587a5ca39bb3f49a48b187c0e8d3588757140900392fa4c61365` |
| `spec_ee_joint_normalizer.pt` | action/state normalizer (also baked into the dynamics checkpoint; needed for training) | 9 KB | `d4a6733eebf7989d7fb8ef519ac4e8efd663da54de66af5a5f361ca4eb0adf21` |
| `yam.urdf` | YAM bimanual station kinematics (the XDOF YAM station model) for the end-effector actions | 14 KB | `6dbda3fc41aaf9dc7eba6481011a2bbf06a0bb85816995b6276300c5098e8eb8` |
| `scenes/` | 10 sample start scenes (latents, states, contact labels; see `scenes/README_scenes.md`) | 136 MB | `scenes/SHA256SUMS` |

```bash
pip install -U "huggingface_hub[cli]"
hf download Amazon-FAR/dreamfar --local-dir dreamfar_assets        # everything, ~5 GB
(cd dreamfar_assets && sha256sum -c SHA256SUMS && cd scenes && sha256sum -c SHA256SUMS)
# or a single file without the CLI:
wget https://huggingface.co/Amazon-FAR/dreamfar/resolve/main/tok256_step000200000.ckpt
```

## Quick start

Roll the model out from a sample scene (10 s of seed frames, then 10 s predicted from the recorded actions); writes
`[prediction | ground truth]` videos plus PSNR / SSIM / LPIPS per view:

```bash
A=dreamfar_assets
python -m dreamfar.inference.rollout --ckpt $A/dreamfar_v9_step000054000.ckpt --config configs/release/dreamfar_v9.yaml \
    --tokenizer-ckpt $A/tok256_step000200000.ckpt --tokenizer-config configs/release/tok256_ssdd_final_v1.yaml --urdf $A/yam.urdf \
    --latent-root $A/scenes/latents --contact-dir $A/scenes/contact --episode 0 --regime mid --frames 200 --out outputs/rollouts
```

Drive the model yourself in the browser (keyboard and mouse control of both arms):

```bash
A=dreamfar_assets
export DEMO_CKPT=$A/dreamfar_v9_step000054000.ckpt DEMO_CONFIG=configs/release/dreamfar_v9.yaml \
       TOKENIZER_CKPT=$A/tok256_step000200000.ckpt TOKENIZER_CONFIG=configs/release/tok256_ssdd_final_v1.yaml \
       URDF_PATH=$A/yam.urdf LATENT_ROOT=$A/scenes/latents CONTACT_DIR=$A/scenes/contact HELDOUT_PATH=$A/scenes/scenes.json
CUDA_VISIBLE_DEVICES=0 python -m uvicorn dreamfar.demo.server:app --host 127.0.0.1 --port 8000 \
    --ws-max-size 33554432 --ws-per-message-deflate false
# open http://localhost:8000 (tunnel the port from a remote machine); headless check:
python -m dreamfar.demo.ws_client --url ws://127.0.0.1:8000/ws --seconds 8 --out outputs/demo_check
```

Seeding regimes, programmatic use, keyboard controls and the demo's settings: [docs/inference.md](docs/inference.md).

## Training

Data layout (raw episodes, contact labels, latent shards): [docs/data_format.md](docs/data_format.md). Recipes,
multi-node launch, resume and logging: [docs/training.md](docs/training.md).

```bash
# 1. tokenizer (initialized from SSDD's F16C16_M_256 weights)
GPUS=8 bash scripts/train_tokenizer.sh name=tok256 \
    dataset.raw_root=<raw_root> dataset.heldout_path=<heldout.json> \
    algorithm.encoder_checkpoint=<F16C16_M_256.safetensors> algorithm.decoder_checkpoint=<F16C16_M_256.safetensors> \
    algorithm.repa.lingbot_weights_dir=<lingbot small weights>
# 2. latents: encode every episode once with the frozen tokenizer
python -m dreamfar.data.precompute_latents --raw-root <raw_root> --out <latent_root> \
    --tokenizer-ckpt tok256_step000200000.ckpt --tokenizer-config configs/release/tok256_ssdd_final_v1.yaml \
    --urdf yam.urdf --shard-episodes 512 --device cuda:0
# 3. dynamics
GPUS=8 bash scripts/train_dynamics.sh name=dreamfar_v9 \
    dataset.latent_root=<latent_root> dataset.contact_dir=<contact_dir> dataset.heldout_path=<heldout.json> \
    dataset.normalizer_path=<normalizer.pt> \
    algorithm.tokenizer_ckpt=tok256_step000200000.ckpt algorithm.tokenizer_config=configs/release/tok256_ssdd_final_v1.yaml \
    algorithm.ee_urdf_path=yam.urdf
```

## Citation

```bibtex
@article{dreamfar2026,
  title   = {DreamFAR: Real-Time and Long-Horizon Interaction with Diverse Objects},
  author  = {Dou, Yiming and He, Botao and Loquercio, Antonio and Owens, Andrew and Malik, Jitendra and Sontakke, Sumedh Anand and Liu, Ruoshi},
  journal = {arXiv preprint arXiv:2611.22001},
  year    = {2026}
}
```

## License

The code in this repository is licensed under the Apache License 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).
The released weights and sample scenes on Hugging Face are licensed under CC BY-NC 4.0.
The tokenizer requires [facebook/SSDD](https://github.com/facebookresearch/SSDD), which is not included here and is
licensed under CC BY-NC 4.0, as are the SSDD weights the tokenizer was initialized from. All third-party components:
[THIRD_PARTY.md](THIRD_PARTY.md).

## Acknowledgements

The tokenizer builds on [SSDD](https://github.com/facebookresearch/SSDD) (its encoder, U-ViT decoder and flow-matching
code are used as a dependency, and its pretrained weights as the initialization) and uses [LingBot-Vision](https://github.com/robbyant/lingbot-vision) as the REPA
teacher; the multi-chunk-prediction heads follow *Next Forcing*; the memory and streaming design draws on
Diffusion Forcing, WorldMem, LongLive and WEAVER; the project started from the
[Interactive World Simulator](https://github.com/WangYixuan12/interactive_world_sim) codebase.
