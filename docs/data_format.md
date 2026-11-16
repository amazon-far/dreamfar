# Data format

What the code reads, and how to turn your own episodes into the latent shards the dynamics model trains and runs on.

## Raw episodes

```
<raw_root>/index.json                          {"fps": 30, "episodes": [{"id": "<episode_id>", "task_name": "...", "size": <n_frames>}, ...]}
<raw_root>/<episode_id>/top.mp4                 `size` frames, time-aligned across views (any resolution; letterboxed to 256x256)
<raw_root>/<episode_id>/left.mp4
<raw_root>/<episode_id>/right.mp4
<raw_root>/<episode_id>/states_actions.bin      float64, C-order, shape (size, 28): [:, :14] joint state, [:, 14:] joint action
                                                YAM layout per 14: [L_j1..L_j6, L_grip, R_j1..R_j6, R_grip]; grippers in [0,1] (1 = open)
```

Row `k` of `states_actions.bin` is the state at frame `k` and the position target commanded at frame `k` (30 Hz).

## Contact labels (per episode, optional for inference, required for dynamics training)

`<contact_dir>/<episode_id>.json` (non-alphanumeric characters of the id replaced by `_`):

```json
{"frames": [0, 3, 6, ...],                       // native frame indices labeled (10 Hz grid)
 "contact": {"left": [0, 0, 1, ...], "right": [0, 1, 1, ...]},   // per-arm contact bit per labeled frame
 "keyframes": [87, 213, ...]}                    // native frames where either arm's contact flips
```

The paper's labels came from a fine-tuned vision-language contact labeler that is not part of this release;
any labeler producing this format works (`dreamfar/data/contact.py` turns it into per-frame bits + keyframes).

## Held-out split and normalizer

* `heldout.json`: `{"asset_ids": ["<episode_id>", ...]}`, the validation episodes (everything else trains).
* Normalizer: `python -m dreamfar.data.spec_normalizer --urdf yam.urdf --out normalizer.pt` (bounds from the
  URDF joint limits + the FK-reachable workspace; the released one is in the model zoo).

## Latent precompute (frozen tokenizer -> latent shards)  [smoke-tested]

```bash
python -m dreamfar.data.precompute_latents --raw-root <raw_root> --out <latent_root> \
    --tokenizer-ckpt tok256_step000200000.ckpt --tokenizer-config configs/release/tok256_ssdd_final_v1.yaml \
    --urdf yam.urdf --shard-episodes 512 --device cuda:0
```

Several processes can split the shards (`--shard-index i --num-shards n`). Output per shard:

```
<latent_root>/shard_XXX/latents.bin            float16 (N, 3, 16, 16, 16): view-stacked unit-sphere latents (top, left, right)
<latent_root>/shard_XXX/states_actions.bin     float32 (N, 28) joint state | action
<latent_root>/shard_XXX/states_actions_ee.bin  float32 (N, 40) end-effector state | action: [L_pos(3) L_rot6d(6) L_grip R_pos(3) R_rot6d(6) R_grip] x2
<latent_root>/shard_XXX/index.json             {"views", "latent_shape", "latent_dtype", "fps", "resolution", "episodes": [{"asset_id","start","size","task_name"}]}
```
