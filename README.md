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

[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](https://amazon-far.github.io/dreamfar/assets/paper/DreamFAR.pdf)
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

Python ≥ 3.11 and one NVIDIA GPU (tested: PyTorch 2.7.1 + CUDA 12.6 on A100-40GB and H200).

```bash
git clone https://github.com/amazon-far/dreamfar.git && cd dreamfar
python3.11 -m venv .venv && source .venv/bin/activate         # or: uv venv --seed --python 3.11 .venv
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126   # or cu128
pip install -r requirements.txt
bash scripts/install_ssdd.sh          # facebook/SSDD at a pinned commit, editable, into third_party/SSDD
pip install -e .
```

## Download

The checkpoints (tokenizer, dynamics, normalizer) and a small pack of sample start scenes will be published on
Hugging Face under [Amazon-FAR](https://huggingface.co/Amazon-FAR); download commands and sha256s will appear here.
You also need the YAM robot URDF (`yam.urdf`) for the end-effector kinematics.

## Quick start

Roll the model out from a sample scene (10 s of seed frames, then 10 s predicted from the recorded actions); writes
`[prediction | ground truth]` videos plus PSNR / SSIM / LPIPS per view:

```bash
python -m dreamfar.inference.rollout --ckpt dreamfar_v9_step000054000.ckpt --config configs/release/dreamfar_v9.yaml \
    --tokenizer-ckpt tok256_step000200000.ckpt --tokenizer-config configs/release/tok256_ssdd_final_v1.yaml --urdf yam.urdf \
    --latent-root <latent_root> --contact-dir <contact_dir> --episode 0 --regime mid --frames 300 --out outputs/rollouts
```

Drive the model yourself in the browser (keyboard and mouse control of both arms):

```bash
export DEMO_CKPT=dreamfar_v9_step000054000.ckpt DEMO_CONFIG=configs/release/dreamfar_v9.yaml \
       TOKENIZER_CKPT=tok256_step000200000.ckpt TOKENIZER_CONFIG=configs/release/tok256_ssdd_final_v1.yaml \
       URDF_PATH=yam.urdf LATENT_ROOT=<latent_root> CONTACT_DIR=<contact_dir> HELDOUT_PATH=<heldout.json>
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
  title   = {DreamFAR: TITLE},
  author  = {AUTHORS},
  journal = {arXiv preprint arXiv:PAPER-ID},
  year    = {2026}
}
```

## License

The code in this repository is licensed under the Apache License 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).
The tokenizer requires [facebook/SSDD](https://github.com/facebookresearch/SSDD), which is not included here and is
licensed under CC BY-NC 4.0, as are the SSDD weights the tokenizer was initialized from. All third-party components:
[THIRD_PARTY.md](THIRD_PARTY.md).

## Acknowledgements

The tokenizer builds on [SSDD](https://github.com/facebookresearch/SSDD) (its encoder, U-ViT decoder and flow-matching
code are used as a dependency, and its pretrained weights as the initialization) and uses [LingBot-Vision](https://github.com/robbyant/lingbot-vision) as the REPA
teacher; the multi-chunk-prediction heads follow *Next Forcing*; the memory and streaming design draws on
Diffusion Forcing, WorldMem, LongLive and WEAVER; the project started from the
[Interactive World Simulator](https://github.com/WangYixuan12/interactive_world_sim) codebase.
