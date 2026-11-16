# Interactive demo (keyboard + mouse)

`dreamfar/demo/` serves one page on which you drive the world model live: with the KEYBOARD (end-effector
translation / rotation / gripper keys -> differential IK -> joint clamp -> FK action) or with the MOUSE (pointer
over the top view = the selected arm's table position, wheel = height, Q/W A/S Z/X = orientation, Space / middle
click = gripper), switchable per session. The model steps only while an input is pending; with nothing queued it
idles with its KV cache frozen. One GPU is enough; with two visible GPUs the decoder runs on the second
(fps = 1/max(predict, decode) instead of 1/(predict + decode)).


## Inputs (environment variables, read at import)

Required:

    DEMO_CKPT      dynamics checkpoint (.ckpt)
    DEMO_CONFIG    its config: a model-zoo config (configs/release/*.yaml) or the run's .hydra/config.yaml
    TOKENIZER_CKPT / TOKENIZER_CONFIG   the tokenizer files (override the paths stored in DEMO_CONFIG)
    LATENT_ROOT    precomputed latent shards (shard_*/index.json + latents / states_actions.bin): the seed scenes
    URDF_PATH      the YAM URDF (forward kinematics; also written into the config as ee_urdf_path)

Optional:

    HELDOUT_PATH   {"asset_ids": [...]} restricting the scenes offered (one scene per task is kept, in this order)
    CONTACT_DIR    per-arm contact labels: with them the mid-start seeds the keyframe memory from the labels
    DEMO_TASKS     `;`-separated case-insensitive prefixes on the task name (further restricts the scenes)
    DEMO_SEED_MODE mid (default, take over after 10 s of real frames) | dup (cold start from frame 0)
    DEMO_SEED_FRAMES  mid-start length in 10 Hz frames (100)
    DEMO_CTRL_MODE keyboard (default) | mouse: the input mode a new session starts in (the page switches live)
    DEMO_EP        -1 (default: random scene) | index into the scene list, pinned for Reset / New episode
    DEMO_TASK_LIST_MAX  tasks listed in the page's picker (400)
    DEMO_PIPELINE  auto (default: decode on a 2nd GPU when visible) | 0 | 1
    DEMO_COMPILE_DECODER / DEMO_WARM_DYNAMICS   1 (default): torch.compile + warm both at startup
    DEMO_RECOMPILE_LIMIT  torch._dynamo recompile limit (64)
    DEMO_TOTAL     frames one generation can commit before it reseeds (8000)
    DEMO_DECODE_STEPS  decoder Euler steps (2)
    DEMO_ACTION_HZ / DEMO_QUEUE_MAX / DEMO_JPEG_Q / DEMO_VIEW_ORDER   streaming knobs (10 / 600 / 85 / left,top,right)
    DEMO_GRIP_STEP / DEMO_EE_POS_STEP / DEMO_EE_ROT_STEP / DEMO_IK_DAMP   keyboard steps per frame (0.1 / 0.01 m / 0.03 rad / 0.05)
    DEMO_MOUSE_*   mouse servo knobs (MAX_DP 0.04 m, MAX_DW 0.10 rad, ROT_STEP 0.05, ROT_LEAD 0.30 rad, DZ 0.01 m, SMOOTH 0.5, DGRIP 0.15,
                   EPS_P 0.003, EPS_W 0.01, MAX_AQ 2, AXES "-y,-x", BOX_L / BOX_R, GRACE_S 1.0, SPAWN 1, SPAWN_*)
    DEMO_HAND_TCP / DEMO_HAND_IK_* / DEMO_HAND_W_* / DEMO_HAND_MAX_DQ / DEMO_HAND_REACH_GUARD / DEMO_HAND_EE_BOX_MARGIN
                   iterative-IK knobs of the mouse admission (tool offset 0,0,0.12 m; 6 iters; weights 50/1/0.02/0.02)
    DYN_FLOW_STEPS / STREAM_CTX_NOISE / INFER_T_SCHEDULE / INFER_WEIGHTS_DTYPE
                   inference settings applied over the config (16 / 0.2404252730845196 / logit_normal / bf16)

## Launch

    export DEMO_CKPT=dreamfar_v9_step000054000.ckpt DEMO_CONFIG=configs/release/dreamfar_v9.yaml \
           TOKENIZER_CKPT=tok256_step000200000.ckpt TOKENIZER_CONFIG=configs/release/tok256_ssdd_final_v1.yaml \
           URDF_PATH=yam.urdf LATENT_ROOT=<latent_root> CONTACT_DIR=<contact_dir> HELDOUT_PATH=<heldout.json>
    # one GPU:
    CUDA_VISIBLE_DEVICES=0   python -m uvicorn dreamfar.demo.server:app --host 127.0.0.1 --port 8000 \
                                 --ws-max-size 33554432 --ws-per-message-deflate false
    # two GPUs (cuda:0 predicts, cuda:1 decodes):
    CUDA_VISIBLE_DEVICES=0,1 python -m uvicorn dreamfar.demo.server:app --host 127.0.0.1 --port 8000 \
                                 --ws-max-size 33554432 --ws-per-message-deflate false

`--ws-per-message-deflate false` matters: a JPEG frame does not compress and the deflate would run on the event
loop for every frame. Startup builds the model, compiles + warms the decoder and the dynamics (a few minutes); the
page is served only after that. On a remote machine tunnel the port (`ssh -N -L 8000:localhost:8000 <host>`) and
open http://localhost:8000. The page's websocket follows its own host:port. One viewer at a time: a new tab evicts
the previous session.

## Endpoints

    GET /          the page            GET /catalog   modes + tasks (JSON)      GET /state   the live session's hello + state
    WS  /ws        the session: binary frames (52-byte header + JPEG panel) + JSON control messages
                   client -> server: keys, mouse, reset {new_episode, task, seed_mode}, mode {mode, task, new_episode}, ping
                   server -> client: hello, state, stats, mouse_state, mode, pong, error
