# Third-party code and assets

The code in this repository is licensed under Apache-2.0 (see `LICENSE`). It depends on the following
third-party work, which carries its own terms. No third-party source code is vendored.

## External source dependency (not vendored)

| Package | Origin | License | Used by |
|---|---|---|---|
| `ssdd` | [facebook/SSDD](https://github.com/facebookresearch/SSDD) at commit `c8542960d69107535aeb1b673676786a1544c5cf`, installed by `scripts/install_ssdd.sh` | **CC BY-NC 4.0** (non-commercial) | `dreamfar/tokenizer/encoder.py` subclasses `ssdd.models.vq_encoder.VQEncoder`; `dreamfar/tokenizer/decoder.py` subclasses `ssdd.models.ssdd.uvit.UViTDecoder`; `dreamfar/tokenizer/net.py` uses `ssdd.flow` (flow-matching trainer and Euler sampler). Importing `ssdd` also imports `diffusers`. |

## Code adapted from permissively licensed projects

| Path | Origin | License | Notes |
|---|---|---|---|
| `dreamfar/dynamics/attention.py` (rotary helpers), `dreamfar/dynamics/embeddings.py` | Adapted from the [Interactive World Simulator (iws)](https://github.com/WangYixuan12/interactive_world_sim) reference implementation (MIT, Copyright (c) 2024 Boyuan Chen; license text below) and [lucidrains/rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch) usage patterns (used as a pip dependency) | MIT | Heavily modified. |
| `dreamfar/dynamics/embeddings.py` (`get_timestep_embedding`, `Timesteps`, `TimestepEmbedding`) | Adapted from [huggingface/diffusers](https://github.com/huggingface/diffusers) `src/diffusers/models/embeddings.py` (Copyright 2024 The HuggingFace Team) | Apache-2.0 | Modified: accepts 1-D or 2-D index tensors; projection without conditioning inputs. |
| `dreamfar/dynamics/mcp.py` | Our implementation of the multi-chunk-prediction heads of *Next Forcing* (arXiv 2606.11187) | — | No code copied. |
| `dreamfar/data/contact.py` | Our code | — | Derives contact keyframes from the contact sidecars; the sidecars themselves were produced by a fine-tuned VLM labeler that is **not** part of this release (format documented in the README). |

## Python dependencies (not vendored)

| Package | License | Role |
|---|---|---|
| torch, torchvision | BSD-3 | everything |
| lightning | Apache-2.0 | training loop / DDP |
| hydra-core, omegaconf | MIT | configuration |
| einops | MIT | tensor layout |
| rotary-embedding-torch | MIT | 2-D spatial RoPE in the DiT |
| safetensors, huggingface_hub | Apache-2.0 | SSDD init weights |
| diffusers | Apache-2.0 | imported by the `ssdd` package |
| pyyaml | MIT | imported by the `ssdd` package |
| wandb | MIT | optional experiment logging |
| schedulefree | Apache-2.0 | RAdamScheduleFree (tokenizer optimizer) |
| lpips | BSD-2 | LPIPS loss / metric (downloads the VGG/AlexNet weights) |
| torchmetrics | Apache-2.0 | PSNR / SSIM |
| av (PyAV) | BSD-3 | MP4 decoding (bundles FFmpeg, LGPL/GPL components) |
| imageio, imageio-ffmpeg | BSD-2 | MP4 writing |
| opencv-python-headless | Apache-2.0 | JPEG encoding in the demo |
| scipy | BSD-3 | spec normalizer (Sobol sampling) |
| urdfpy (WangYixuan12 fork) | MIT | YAM URDF forward kinematics |
| fastapi, uvicorn, websockets | MIT / BSD-3 / BSD-3 | demo server |
| [lingbot-vision](https://github.com/robbyant/lingbot-vision) + its `small` weights | Apache-2.0 | frozen REPA teacher for tokenizer training only (optional at inference) |

## Assets referenced

* `facebook/SSDD` `F16C16_M_256.safetensors` (CC BY-NC 4.0): tokenizer initialization (training only; the
  released tokenizer checkpoint was fine-tuned from these weights).
* LingBot-Vision `small` backbone weights (Apache-2.0): REPA teacher.
* YAM robot URDF (`yam.urdf`): required for the end-effector forward kinematics; obtain it from the robot
  vendor's description package (not included).

## License texts

### Interactive World Simulator (iws)

```
MIT License

Copyright (c) 2024 Boyuan Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software, and the following
paragraph shall be included in REAMD.md or other instruction files.

"This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template
[repo](https://github.com/buoyancy99/research-template). By its license,
you must keep the above sentence in `README.md` and the `LICENSE` file to
credit the author."

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
