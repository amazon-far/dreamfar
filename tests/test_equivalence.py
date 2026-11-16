"""Bit-exactness of the release against reference tensors dumped from the original codebase.

  python -m tests.test_equivalence --ref equiv_ref.pt --tok-ckpt ... --tok-config ... --dyn-ckpt ... \
      --dyn-config configs/... --latent-root ... --contact-dir ...

Checks: tokenizer encode/decode (fp32 + bf16 autocast) and training loss; dynamics state_dict names/shapes;
one streaming loss evaluation on the reference synthetic batch; the reference episode batch from the data
loader; a mid-start rollout (eager and compiled). Every comparison is torch.equal (bit-identical)."""
from __future__ import annotations

import argparse
import json

import torch
from einops import rearrange

DEV = "cuda:0"
FAIL = []


def check(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    a, b = a.detach().cpu(), b.detach().cpu()
    same = a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    diff = (a.float() - b.float()).abs().max().item() if a.shape == b.shape else float("nan")
    print(f"  {'OK ' if same else 'XX '} {name:28s} shape={tuple(a.shape)} max|diff|={diff:.3e}", flush=True)
    if not same:
        FAIL.append(name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--tok-ckpt", required=True)
    p.add_argument("--tok-config", required=True)
    p.add_argument("--dyn-ckpt", required=True)
    p.add_argument("--dyn-config", required=True)
    p.add_argument("--latent-root", required=True)
    p.add_argument("--contact-dir", required=True)
    p.add_argument("--urdf", required=True)
    p.add_argument("--tokenizer-only", action="store_true")
    a = p.parse_args()
    torch.set_float32_matmul_precision("high")
    ref = torch.load(a.ref, map_location="cpu", weights_only=False)

    # ---- tokenizer ------------------------------------------------------------------------------
    from dreamfar.tokenizer.net import load_tokenizer
    print("[tokenizer]")
    # The reference was dumped from a TRAINABLE net. `F.linear` on the decoder's 4-D (B,H,W,C) context
    # embedding takes a different GEMM path when the weight requires no grad (flattened addmm vs matmul),
    # which changes fp32 rounding by ~1e-3 -- so match the reference's autograd state here.
    tok = load_tokenizer(a.tok_ckpt, a.tok_config).to(DEV).eval().requires_grad_(True)
    r = ref["tok"]
    x, noise = r["x"].to(DEV), r["noise"].to(DEV)
    with torch.no_grad():
        z = tok.encode(x)
        xr = tok.decode(z, steps=2, noise=noise)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z_bf = tok.encode(x)
            xr_bf = tok.decode(z_bf, steps=2, noise=noise)
    check("state_dict keys", torch.tensor([sorted(tok.state_dict()) == r["keys"]]), torch.tensor([True]))
    check("encode fp32", z, r["z"])
    check("decode fp32", xr, r["xr"])
    check("encode bf16", z_bf, r["z_bf"])
    check("decode bf16", xr_bf, r["xr_bf"])
    tok.train()
    torch.manual_seed(0)
    with torch.no_grad():
        o = tok(x)
    tok.eval()
    check("train diffusion loss", o.losses["diffusion"], r["diff_loss"])
    check("train x0_pred", o.x0_pred, r["x0_pred"])
    del tok
    torch.cuda.empty_cache()
    if a.tokenizer_only:
        print("\nRESULT:", "ALL BIT-IDENTICAL" if not FAIL else f"FAILED: {FAIL}")
        raise SystemExit(1 if FAIL else 0)

    # ---- dynamics -------------------------------------------------------------------------------
    from dreamfar.inference.build import build_model
    from dreamfar.inference.episodes import list_episodes, load_episode
    from dreamfar.inference.seeding import build_regime
    print("[dynamics]")
    s = ref["dyn_settings"]
    model = build_model(a.dyn_ckpt, a.dyn_config, DEV, ema=True,
                        overrides={"dyn_flow_steps": s["P"], "infer_t_schedule": s["sched"], "stream_ctx_noise": s["ctx"],
                                   "tokenizer_ckpt": a.tok_ckpt, "tokenizer_config": a.tok_config,
                                   "ee_urdf_path": a.urdf, "infer_weights_dtype": "fp32"})
    live = {k: tuple(v.shape) for k, v in model.dynamics.state_dict().items()}
    check("dynamics key/shape set", torch.tensor([live == ref["dyn_keys"]]), torch.tensor([True]))
    if live != ref["dyn_keys"]:
        print("   only release:", sorted(set(live) - set(ref["dyn_keys"]))[:5],
              "only original:", sorted(set(ref["dyn_keys"]) - set(live))[:5])
    b = {k: v.to(DEV) for k, v in ref["dyn_batch"].items()}
    model.train()
    z = model.stack_views(b["latent"])
    action = rearrange(model.cond_action(b["action"]), "b t a -> t b a")
    state = rearrange(b["state"].float(), "b t a -> t b a")
    z_mem = model.stack_views(b["latent_memory"])
    mem_state = rearrange(b["state_memory"].float(), "b m a -> m b a")
    mem_dt = rearrange(b["mem_dt"].float(), "b m -> m b")
    mem_valid = rearrange(b["mem_valid"], "b m -> m b")
    contact_gt = rearrange(b["contact"].float(), "b t k -> t b k")
    fut = {"z": model.stack_views(b["latent_future"]),
           "state": rearrange(b["state_future"].float(), "b k a -> k b a"),
           "action": rearrange(b["action_future"].float(), "b k a -> k b a"),
           "valid": rearrange(b["future_valid"], "b k -> k b")}
    torch.manual_seed(0)
    with torch.no_grad():
        flow = model._flow_loss_stream(z, action, state, z_mem, model.normalizer["state"].normalize(mem_state),
                                       mem_dt, mem_valid, mem_frame_mask=b["mem_frame_mask"], contact_gt=contact_gt, fut=fut)
    loss = flow + model.state_pred_weight * model._last_state_loss
    for i, ml in enumerate(model._last_mcp_losses):
        loss = loss + float(model.mcp_loss_weights[i]) * ml
    loss = loss + model.contact_pred_weight * model._last_contact_bce
    model.eval()
    rl = ref["dyn_loss"]
    check("flow loss", flow, rl["flow"])
    check("state loss", model._last_state_loss, rl["state"])
    for i, ml in enumerate(model._last_mcp_losses):
        check(f"mcp loss k{i + 1}", ml, rl["mcp"][i])
    check("contact bce", model._last_contact_bce, rl["bce"])
    check("total loss", loss, rl["total"])
    check("x0 mse", model._last_x0_mse, rl["x0_mse"])

    # ---- episode loader + rollout ---------------------------------------------------------------
    print("[rollout]")
    re_ = ref["ep"]
    eps, meta = list_episodes(a.latent_root)
    ep = next(e for e in eps if e.asset_id == re_["asset_id"])
    batch = load_episode(model, ep, meta, DEV, cap=re_["cap"], contact_dir=a.contact_dir)
    for k in ("latent", "action", "state", "frame_idx", "contact"):
        check(f"batch {k}", batch[k], re_["batch"][k])
    z_bt = rearrange(model.stack_views(batch["latent"]), "t b c h w -> b t c h w")
    kw = build_regime("mid", model, z_bt, model.cond_action(batch["action"]), batch["state"], batch["frame_idx"],
                      contact=batch["contact"], seed_sec=re_["seed_sec"], fps=10)
    for mode in ("eager", "compiled"):
        with torch.no_grad():
            zp, mem_sel = model.rollout(kw["z_hist"], kw["action"], kw["state"], kw["frame_idx"],
                                        seed_latents=kw["seed_latents"], seed_contact=kw["seed_contact"],
                                        compiled=(mode == "compiled"))
        rr = ref["rollout"][mode]
        check(f"{mode} z_pred", zp, rr["z_pred"])
        check(f"{mode} state_pred", model._last_state_pred, rr["state_pred"])
        ok = json.dumps(mem_sel) == rr["mem_sel"] and list(model._last_kf_frames) == list(rr["kf"]) \
            and int(model._last_snap_miss) == int(rr["snap_miss"])
        check(f"{mode} memory/keyframes/harvest", torch.tensor([ok]), torch.tensor([True]))
    print("\nRESULT:", "ALL BIT-IDENTICAL" if not FAIL else f"FAILED: {FAIL}")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
