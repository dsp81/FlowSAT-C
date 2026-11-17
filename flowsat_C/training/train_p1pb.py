"""
train_p1pb.py — trainer for the P1-perblock variant (models/transformer_p1pb.py).

Copy of training/train.py (branch arch/no-p3). What differs, and only this:
  * model: build_flowsat_c_p1pb (P1 = shared trunk + per-block zero-init heads,
    no to_temb / alphas / to_global); flags --p1-glob-norm, --p1-rank.
  * optimiser: P1 trunk + heads at --head-lr (1e-4), wd 0; encoder / pool / P2
    token path as in train.py (--lr); P2 gate at --gate-lr, default 1e-3 (= stage15_v3).
  * --freeze-backbone is ON by default (all FLUX.2 params incl. mod_t, norm_out).
  * --p2-path-dropout p: per sample, P2 fully nulled with probability p
    (--p2-path-dropout-schedule const | anneal).
  * --p1-path-dropout p: per sample, independently of P2, P1 gets the all-null
    metadata glob with probability p (symmetric pathway dropout; default 0 = off).
  * checkpoints every --save-every (500), NEVER pruned (no --keep-last).
  * every --log-every: P1 per-record / constant d_mu/mu (D1 method), gate,
    fraction of samples with P2 dropped; --stop-on-plateau.
  * DataLoader num_workers 4 and torch.set_num_threads(4) (shared-machine limit).
  * --caption-mix full,terse,null (ported from DJ_Sat_s2 train.py, stage 2).
  * --snapshot-every N: weights-only bf16 state dict at snapshots/step-N.pt, never
    rotated, so a 50k stage-2 run keeps every 1k for the final pick (~8.4 GB each,
    vs ~54 GB for a full ZeRO-2 checkpoint).
  * verification hooks: --verify-precision, --grad-report-steps.
The original docstring follows.

train_flowsat_c.py — staged trainer for FlowSAT-C.

Stages:
  0  assert bit-identical to stock Flux2Transformer2DModel  (gate, seconds)
  1  full backbone fine-tune, metadata frozen shut -> ESTABLISH FID/CLIP HERE
  2  encoder + graft + tokens open, backbone at 0.1x LR
  3  optional joint polish at very low LR

Measuring FID at the stage-1/2 boundary makes any quality regression
ATTRIBUTABLE and DETECTABLE rather than confounded with domain adaptation.

FID 31.10 does not transfer across backbones. It is re-established in stage 1.
FLUX.2's base quality is well above Sana-0.6B so the ceiling is higher, but
budget the recovery run rather than assuming it.

A completed progress bar and a falling FID are compatible with controllability
going to zero. That is exactly what happened in v1. Watch the probes.

INTERPRETABILITY: --keep-stage-checkpoint preserves the end-of-stage state as a
permanent artifact. The most informative post-hoc result available is
COMPARATIVE -- how the representation of location changes between stage 1
(metadata shut) and stage 2 (open) -- and that comparison is impossible without
the stage-1 weights. Nothing here affects gradients or final weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from datetime import timedelta

import torch
from accelerate import Accelerator
from accelerate.utils import (InitProcessGroupKwargs, broadcast_object_list,
                               send_to_device, set_seed)
from torch.utils.data import DataLoader
from tqdm import tqdm

from flowsat_C.training.diagnostics import DiagnosticLogger
from flowsat_C.training.flow_matching import (loss_by_timestep_bucket, rectified_flow_loss,
                             sample_timesteps)
from flowsat_C.models.transformer_p1pb import build_flowsat_c_p1pb
from flowsat_C.models.encoder_p1pb import GLOB_NORMS
from flowsat_C.training.flux2_io import load_flux2_io, prepare_batch
from flowsat_C.data.dataset import (CaptionPolicy, CounterfactualPairSampler,
                          FMoWCaptionDataset, make_collate)
from flowsat_C.training.validation import BestTracker, FixedNoiseBank, validation_loss
from flowsat_C.models.encoder import EncoderConfig, MetadataSpec


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-id", default="black-forest-labs/FLUX.2-klein-base-4B")
    p.add_argument("--stage", type=int, required=True, choices=[0, 1, 2, 3])
    p.add_argument("--resume", default=None)
    p.add_argument("--init-from", default=None,
                   help="load MODEL WEIGHTS ONLY from a prior stage. Use this for "
                        "stage1 -> stage2; --resume is for continuing the SAME stage.")
    p.add_argument("--keep-stage-checkpoint", action=argparse.BooleanOptionalAction,
                   default=True)

    p.add_argument("--resolution", type=int, default=512)
    # GLOBAL batch is fixed; grad-accum is DERIVED from the GPU count. Switching
    # between 1 / 2 / 4 GPUs mid-project then leaves the optimisation identical
    # instead of silently changing the effective batch size.
    p.add_argument("--per-gpu-batch", type=int, default=8)
    p.add_argument("--global-batch", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--p1-glob-norm", choices=GLOB_NORMS, default="ln_mag",
                   help="P1 input: ln_mag = [LayerNorm(pool) ; magnitude] (3073-d), "
                        "none = raw pooled vector (3072-d). No to_global bottleneck")
    p.add_argument("--p1-rank", type=int, default=128,
                   help="rank r of every per-block head U_k(D_k(h)): 1024 -> r -> mod size")
    p.add_argument("--head-lr", type=float, default=1e-4,
                   help="LR of the P1 trunk and all per-block / norm_out heads (wd 0)")
    p.add_argument("--p2-path-dropout", type=float, default=0.3,
                   help="per sample, null P2 completely (metadata unreachable by "
                        "every query) with this probability; P1 still sees metadata")
    p.add_argument("--p1-path-dropout", type=float, default=0.0,
                   help="per sample, independently of --p2-path-dropout, feed P1 the "
                        "all-null metadata glob (P1 keeps its constant delta, loses the "
                        "per-record one) with this probability; P2 still sees metadata")
    p.add_argument("--p2-path-dropout-schedule", choices=["const", "anneal"],
                   default="const",
                   help="anneal (for stage 2): 0.5 for the first 33%% of --max-steps, "
                        "cosine to 0.1, held at 0.1 for the last 15%%")
    p.add_argument("--stop-on-plateau", action="store_true",
                   help="stop when the P1 per-record d_mu/mu (block average) AND the P2 "
                        "gate have both changed < --plateau-tol (relative) over the last "
                        "--plateau-window steps")
    p.add_argument("--plateau-window", type=int, default=1000)
    p.add_argument("--plateau-tol", type=float, default=0.05)
    p.add_argument("--verify-precision", action="store_true",
                   help="verification only (§4.3): randomise the heads small, run one "
                        "batch under the live DeepSpeed config, dump + assert the dtypes "
                        "at the P1 addition, then exit")
    p.add_argument("--grad-report-steps", type=int, default=0,
                   help="verification only: print per-group grad norms for the first N "
                        "optimizer steps (U_k / D_k / trunk / encoder / gate / backbone)")
    p.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True,
                   help="stage 1.5: train ONLY the metadata pathways. The "
                        "backbone must not drift while captions are nulled -- an "
                        "unfrozen backbone trained caption-free learns to stop "
                        "expecting structure from text, which is exactly what we "
                        "are preserving")
    p.add_argument("--caption-null-frac", type=float, default=None,
                   help="override the train CaptionPolicy null (text-dropout) "
                        "fraction. 1.0 = metadata-only phase: no caption ever "
                        "reaches the model, so metadata is the only thing that "
                        "can explain the residual. Remaining mass keeps the "
                        "default 65:25 full:terse ratio")
    p.add_argument("--caption-mix", default=None,
                   help="full,terse,null train caption fractions (sum 1), e.g. "
                        "0.9,0,0.1. Overrides --caption-null-frac. terse = first "
                        "clause of the metadata-agnostic caption")
    p.add_argument("--grid-null-text", action="store_true",
                   help="also save a metadata-only sample grid (captions nulled) "
                        "beside the text-conditioned one; that is the grid where "
                        "a collapse to the conditional mean is visible")
    p.add_argument("--gate-lr", type=float, default=1e-3,
                   help="LR for the P2 attention gate bias. 1e-3 = stage15_v3 (runs/stage15_v3/"
                        "launch.sh). A scalar under Adam moves ~lr per step, so 1e-4 would keep "
                        "the gate shut; P1 is protected by --p2-path-dropout, not a slow gate")
    p.add_argument("--gate-init", type=float, default=None,
                   help="stage >= 2: reset the P2 gate bias to this value after "
                        "--init-from (the -20 init exists for the stage-0 identity gate)")
    p.add_argument("--ctrl-sites", type=int, default=4,
                   help="val sites for the per-field controllability sweeps at FID steps")
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--halt-at-step", type=int, default=0,
                   help="end the run at this step (after its save/snapshot); the cosine LR still follows --max-steps, "
                        "so a run halted at N matches a --max-steps run at step N")
    p.add_argument("--ema", action="store_true",
                   help="stage >= 2 only. Off by default: AveragedModel copies all "
                        "4B params (+8 GB) and is untested under DeepSpeed")
    p.add_argument("--ema-decay", type=float, default=0.999)

    # structure is committed at HIGH noise, so geography is a high-t decision.
    # Keep the backbone default in stage 1; shift mean positive in stage 2.
    p.add_argument("--t-mean", type=float, default=0.0)
    p.add_argument("--t-std", type=float, default=1.0)
    p.add_argument("--t-shift", type=float, default=3.0)
    p.add_argument("--high-noise-frac", type=float, default=0.0)

    p.add_argument("--max-text-tokens", type=int, default=256)
    p.add_argument("--sh-degree", type=int, default=16)
    p.add_argument("--lambda-collin", type=float, default=0.0,
                   help="redundancy penalty between conditioning pathways")
    p.add_argument("--pair-frac", type=float, default=0.5)
    p.add_argument("--val-manifest", default=None)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--val-batches", type=int, default=32)
    p.add_argument("--grid-every", type=int, default=2000,
                   help="16-image sample grid from fixed val prompts + seed (~1-2 min); "
                        "skipped on FID steps, which save their own grid")
    p.add_argument("--grid-until", type=int, default=0,
                   help="only make --grid-every grids up to this step (0 = always); "
                        "e.g. --grid-every 500 --grid-until 2000 for dense early samples")
    p.add_argument("--probe-every", type=int, default=2000)
    p.add_argument("--fid-every", type=int, default=4000)
    p.add_argument("--fid-samples", type=int, default=512)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--snapshot-every", type=int, default=0,
                   help="also write a weights-only bf16 snapshot every N steps to "
                        "snapshots/step-N.pt (never rotated; 0 = off). Loadable with "
                        "--init-from or as a sampler checkpoint")
    p.add_argument("--keep-last", type=int, default=0,
                   help="keep only the newest N ckpt-<step> dirs (0 = keep all). best/ "
                        "and the stage-final checkpoint are never rotated")
    p.add_argument("--best-on", choices=["score", "val_loss"], default="score",
                   help="score = FID - lambda*ctrl at --fid-every steps (BestTracker); "
                        "val_loss = lowest fixed-noise val/loss at --val-every steps. "
                        "best/ is replaced atomically (written to best.tmp, then renamed)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default="flowsat-c")
    p.add_argument("--run-name", default=None,
                   help="default: stage{stage}-{timestamp}")
    p.add_argument("--wandb-mode", default="offline",
                   choices=["online", "offline", "disabled"])
    return p.parse_args()


def log_image_wandb(acc, key, path, step, caption=""):
    """Attach an image file to the W&B run. Main process only (other ranks have
    no run), and never raises: an image-logging failure must not kill a run."""
    if not acc.is_main_process or not os.path.exists(path):
        return
    try:
        import wandb
        acc.get_tracker("wandb").log(
            {key: wandb.Image(path, caption=caption or f"step {step}")}, step=step)
    except Exception as e:
        print(f"wandb image log skipped: {e}", flush=True)


_T_START = time.time()


def phase(msg: str) -> None:
    """Timestamped progress line, rank 0 only (tqdm.write keeps any active bar intact)."""
    if int(os.environ.get("RANK", "0")) == 0:
        el = time.time() - _T_START
        tqdm.write(f"[{time.strftime('%H:%M:%S')} +{el / 60:5.1f}m] {msg}")


@torch.no_grad()
def pathway_grad_norms(um) -> dict:
    """Per-module gradient norms under DeepSpeed ZeRO (p.grad is None there).
    safe_get_full_grad all-reduces, so every rank must call this.

    These are the health signal for the two pathways: if grad/encoder or a
    grad/p1_* group goes to zero while loss falls, that pathway is being ignored.
    p1_U = every U_k (block heads), p1_D = every D_k, p1_trunk, p1_normout = the
    norm_out head (D and U). Frozen backbone tensors have no grad and count 0."""
    from deepspeed.utils import safe_get_full_grad

    def gn(params):
        tot = 0.0
        for p in params:
            g = safe_get_full_grad(p) if p.requires_grad else None
            if g is not None:
                tot += float(g.float().norm()) ** 2
        return tot ** 0.5
    heads = list(um.mod_md_img) + list(um.mod_md_txt) + list(um.mod_md_sgl)
    return {"grad/encoder": gn(list(um.encoder.parameters())),
            "grad/p1_trunk": gn(list(um.p1_trunk.parameters())),
            "grad/p1_D": gn([p for h in heads for p in h.D.parameters()]),
            "grad/p1_U": gn([p for h in heads for p in h.U.parameters()]),
            "grad/p1_normout_D": gn(list(um.mod_md_out.D.parameters())),
            "grad/p1_normout_U": gn(list(um.mod_md_out.U.parameters())),
            "grad/gate_bias": gn([um.gate_bias])}


@torch.no_grad()
def head_grad_table(um) -> dict:
    """Per-head weight grad norms (verification, §4.4): every U_k / D_k by name,
    plus how many backbone tensors hold any gradient at all (must be 0 frozen)."""
    from deepspeed.utils import safe_get_full_grad

    def n(p):
        g = safe_get_full_grad(p) if p.requires_grad else None
        return 0.0 if g is None else float(g.float().norm())
    rows = {}
    for kind in ("img", "txt", "sgl"):
        for i, h in enumerate(getattr(um, f"mod_md_{kind}")):
            rows[f"{kind}_{i}"] = (n(h.U.weight), n(h.D.weight))
    rows["normout"] = (n(um.mod_md_out.U.weight), n(um.mod_md_out.D.weight))
    bb = sum(1 for p in um.tf.parameters()
             if p.requires_grad or getattr(p, "grad", None) is not None)
    return {"heads": rows, "backbone_tensors_with_grad": bb}


def p2_drop_prob(args, step: int) -> float:
    """const: --p2-path-dropout. anneal: 0.5 for the first 33% of --max-steps,
    cosine 0.5 -> 0.1 until 85%, then 0.1 for the last 15%."""
    if args.p2_path_dropout_schedule == "const":
        return args.p2_path_dropout
    f = step / max(args.max_steps, 1)
    if f < 0.33:
        return 0.5
    if f >= 0.85:
        return 0.1
    u = (f - 0.33) / (0.85 - 0.33)
    return 0.1 + 0.4 * 0.5 * (1 + math.cos(math.pi * u))


class PlateauTracker:
    """--stop-on-plateau. Per signal, the mean of the last 4 logged values (200
    steps at --log-every 50, to damp batch noise) is compared with the same window
    `window` steps earlier; plateau when |now - then| / |then| < tol for EVERY
    signal. Rank 0 decides; the decision is broadcast."""

    def __init__(self, window: int, tol: float, keys=("p1/per_dmu_blocks", "gate/bias")):
        self.window, self.tol, self.keys = window, tol, keys
        self.hist: dict[str, list[tuple[int, float]]] = {k: [] for k in keys}

    def update(self, step: int, logs: dict):
        for k in self.keys:
            if k in logs:
                self.hist[k].append((step, float(logs[k])))

    def _win(self, k, end):
        v = [x for s, x in self.hist[k] if end - 200 < s <= end]
        return sum(v) / len(v) if v else None

    def check(self, step: int):
        """-> (plateaued: bool, {key: relative change})"""
        rel = {}
        for k in self.keys:
            now, then = self._win(k, step), self._win(k, step - self.window)
            if now is None or then is None or abs(then) < 1e-12:
                return False, rel
            rel[k] = abs(now - then) / abs(then)
        return all(r < self.tol for r in rel.values()), rel


def rotate_checkpoints(out_dir: str, keep: int) -> None:
    """Delete all but the newest `keep` ckpt-<step> dirs. Main process only."""
    ck = sorted((d for d in os.listdir(out_dir)
                 if d.startswith("ckpt-") and d[5:].isdigit()),
                key=lambda d: int(d[5:]))
    for d in ck[:-keep] if keep > 0 else []:
        shutil.rmtree(os.path.join(out_dir, d), ignore_errors=True)
        print(f"rotated out {d}", flush=True)


def save_best(acc, out_dir: str, step: int, info: dict) -> None:
    """Replace best/ without a window where no best exists: save to best.tmp
    (collective), then rank 0 swaps it in and records which step it is."""
    tmp, dst = os.path.join(out_dir, "best.tmp"), os.path.join(out_dir, "best")
    if acc.is_main_process and os.path.exists(tmp):
        shutil.rmtree(tmp)
    acc.wait_for_everyone()
    save_ckpt(acc, tmp)
    acc.wait_for_everyone()
    if acc.is_main_process:
        old = os.path.join(out_dir, "best.old")
        if os.path.exists(dst):
            os.rename(dst, old)
        os.rename(tmp, dst)
        shutil.rmtree(old, ignore_errors=True)
        with open(os.path.join(dst, "BEST_STEP.json"), "w") as f:
            json.dump({"step": step, **info}, f)
        print(f"best/ <- step {step} {info}", flush=True)
    acc.wait_for_everyone()


def save_ckpt(acc, path: str) -> None:
    """save_state with timing: a ZeRO-2 checkpoint is ~54 GB and silent otherwise."""
    phase(f"saving checkpoint -> {path} ...")
    t = time.time()
    acc.save_state(path)
    phase(f"checkpoint saved in {time.time() - t:.0f}s")


def save_snapshot(acc, model, path: str) -> None:
    """Weights-only bf16 state dict (no optimizer / scheduler / RNG), written
    atomically by rank 0. ZeRO-2 does not shard parameters, so every rank holds
    the full model and no gather is needed."""
    acc.wait_for_everyone()
    if acc.is_main_process:
        phase(f"saving weights snapshot -> {path} ...")
        t = time.time()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sd = {k: v.detach().to("cpu", torch.bfloat16)
              for k, v in acc.unwrap_model(model).state_dict().items()}
        torch.save(sd, path + ".tmp")
        os.replace(path + ".tmp", path)
        phase(f"snapshot saved in {time.time() - t:.0f}s ({len(sd)} tensors)")
    acc.wait_for_everyone()


def load_weights_only(model, path: str) -> None:
    """Model weights only -- no optimizer, scheduler or RNG state -- so a stage
    with different parameter groups can start from the previous stage.

    Accepts a checkpoint dir written by acc.save_state (DeepSpeed ZeRO: the
    fp32 master weights are consolidated from the optimizer partitions), or a
    .safetensors / .pt / .bin state-dict file. Counts are printed because a
    silent partial load here would be brutal to diagnose later."""
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "latest")):
            from deepspeed.utils.zero_to_fp32 import \
                get_fp32_state_dict_from_zero_checkpoint
            sd = get_fp32_state_dict_from_zero_checkpoint(path)
        else:
            raise SystemExit(f"--init-from {path}: no DeepSpeed 'latest' file")
    elif path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu")
    sd = {(k[len("module."):] if k.startswith("module.") else k): v
          for k, v in sd.items()}
    own = model.state_dict()
    shape_bad = [k for k, v in sd.items() if k in own and own[k].shape != v.shape]
    for k in shape_bad:
        sd.pop(k)
    res = model.load_state_dict(sd, strict=False)
    matched = len(own) - len(res.missing_keys)
    print(f"init-from {path}: matched={matched}/{len(own)} "
          f"missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)} "
          f"shape_mismatch={len(shape_bad)}", flush=True)
    for name, ks in (("missing", res.missing_keys),
                     ("unexpected", res.unexpected_keys),
                     ("shape_mismatch", shape_bad)):
        if ks:
            print(f"  {name} (first 10): {ks[:10]}", flush=True)
    return dict(matched=matched, missing=res.missing_keys,
                unexpected=res.unexpected_keys, shape_mismatch=shape_bad)


def collinearity_penalty(deltas: dict) -> torch.Tensor:
    """
    Operationalises "collinear features interfere" as a training objective:
    penalise correlation between the velocity changes attributable to different
    conditioning sources. Redundancy handled in PREDICTION space rather than only
    at the data level.
    """
    keys = list(deltas)
    if len(keys) < 2:
        return torch.zeros((), device=next(iter(deltas.values())).device)
    v = torch.stack([deltas[k].flatten(1).mean(0) for k in keys])
    v = v - v.mean(-1, keepdim=True)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
    c = v @ v.t()
    off = ~torch.eye(len(keys), dtype=torch.bool, device=c.device)
    return c[off].abs().mean()


def verify_precision(args, acc, model, dl, lat_enc, txt_enc):
    """§4.3. One batch through the prepared (DeepSpeed bf16) engine with the P1
    heads set to small random values. Asserts, at every one of the 31 additions,
    mod_t output in the backbone dtype, delta fp32, the sum fp32 and autocast off;
    and at the block inputs, the modulation in the backbone dtype. Run twice: as
    training runs it, and again inside an explicit bf16 autocast region."""
    um = acc.unwrap_model(model)
    heads = list(um.mod_md_img) + list(um.mod_md_txt) + list(um.mod_md_sgl) + [um.mod_md_out]
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for hd in heads:
            for p in (hd.U.weight, hd.U.bias):
                p.copy_((torch.randn(p.shape, generator=g) * 1e-3).to(p.device, p.dtype))
    blk_in = []

    def pre(name):
        def fn(mod, a, kw):
            for k in ("temb_mod_img", "temb_mod_txt", "temb_mod"):
                v = kw.get(k)
                if v is not None:
                    blk_in.append((name, k, str(v.dtype)))
            for i, v in enumerate(a):      # gradient checkpointing passes positionally
                if torch.is_tensor(v) and v.ndim == 2 and v.shape[-1] in (18432, 9216):
                    blk_in.append((name, f"arg{i}", str(v.dtype)))
        return fn
    hs = [um.tf.transformer_blocks[0].register_forward_pre_hook(pre("double_0"), with_kwargs=True),
          um.tf.single_transformer_blocks[0].register_forward_pre_hook(pre("single_0"),
                                                                       with_kwargs=True)]
    batch = prepare_batch(next(iter(dl)), lat_enc, txt_enc)
    lat = batch["latents"]
    t = sample_timesteps(lat.shape[0], lat.device, mean=args.t_mean, std=args.t_std,
                         shift=args.t_shift)
    p2_drop = torch.rand(lat.shape[0], device=lat.device) < args.p2_path_dropout
    print(f"[precision] engine param dtypes: backbone {next(um.tf.parameters()).dtype}, "
          f"heads {um.mod_md_img[0].U.weight.dtype}, encoder "
          f"{next(um.encoder.parameters()).dtype}; DeepSpeed bf16 cfg "
          f"{acc.state.deepspeed_plugin.deepspeed_config.get('bf16') if acc.state.deepspeed_plugin else None}",
          flush=True)
    ok = True
    for label, ac in (("as trained", False), ("inside torch.autocast(bf16)", True)):
        um._precision_probe, blk_in[:] = [], []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ac):
            loss, *_ = rectified_flow_loss(
                lambda x, ts: model(hidden_states=x, encoder_hidden_states=batch["text_emb"],
                                    timestep=ts, img_ids=batch["img_ids"],
                                    txt_ids=batch["txt_ids"], metadata=batch["metadata"],
                                    training_dropout=True, p2_drop=p2_drop),
                lat, t)
        recs = um._precision_probe
        um._precision_probe = None
        combos = {(r["base"], r["delta"], r["sum"], r["autocast"]) for r in recs}
        print(f"[precision] {label}: {len(recs)} additions; (mod_t, delta, sum, autocast "
              f"at add) = {sorted(combos)}; block inputs {sorted(set(blk_in))}; "
              f"loss {loss.item():.4f}", flush=True)
        for r in recs[:2] + recs[-2:]:
            print(f"[precision]   {r}", flush=True)
        bb = str(next(um.tf.parameters()).dtype)
        ok &= (len(recs) == 2 * um.n_double + um.n_single + 1 and
               all(r["base"] == bb and r["delta"] == "torch.float32" and
                   r["sum"] == "torch.float32" and not r["autocast"] for r in recs) and
               len(blk_in) > 0 and all(d == bb for *_, d in blk_in))
    for h in hs:
        h.remove()
    print(f"[precision] {'PASS' if ok else 'FAIL'}", flush=True)
    assert ok, "P1 addition is not fp32 at the point of addition"


def main():
    args = parse_args()
    torch.set_num_threads(4)       # shared-machine limit: <=4 threads per process
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.run_name is None:
        args.run_name = f"stage{args.stage}-{time.strftime('%Y%m%d-%H%M%S')}"
    # before any tracker exists; offline run files land beside the checkpoints
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    os.environ.setdefault("WANDB_DIR", args.out_dir)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (1, 2, 4):
        raise SystemExit(f"use 1, 2 or 4 GPUs (got {world}); other counts make "
                         f"global-batch invariance impossible to preserve")
    denom = args.per_gpu_batch * world
    if args.global_batch % denom:
        raise SystemExit(f"--global-batch {args.global_batch} not divisible by "
                         f"per_gpu_batch*world={denom}")
    grad_accum = args.global_batch // denom
    print(f"world={world} per_gpu={args.per_gpu_batch} "
          f"grad_accum={grad_accum} global={args.global_batch}")

    acc = Accelerator(gradient_accumulation_steps=grad_accum,
                      mixed_precision="bf16",
                      # --wandb-mode disabled: no W&B at all (wandb-core from NFS
                      # missed its startup wait); train.log + diagnostics.jsonl remain
                      log_with=["tensorboard"] + (["wandb"] if args.wandb_mode != "disabled" else []),
                      project_dir=args.out_dir,
                      # FID runs on rank 0 only (~30 min at 512 samples) while
                      # the other ranks wait in a collective; the default 30 min
                      # NCCL timeout would kill the run
                      kwargs_handlers=[InitProcessGroupKwargs(
                          timeout=timedelta(hours=3))])
    ds_plugin = acc.state.deepspeed_plugin
    if ds_plugin is not None:
        # accelerate silently prefers the DeepSpeed value on a mismatch, which
        # would change the global batch
        ds_ga = ds_plugin.deepspeed_config.get("gradient_accumulation_steps")
        if ds_ga not in (grad_accum, "auto"):
            raise SystemExit(f"DeepSpeed gradient_accumulation_steps={ds_ga} != "
                             f"derived grad_accum={grad_accum}; fix the config")

    phase("loading FLUX.2 transformer (4B) ...")
    model = build_flowsat_c_p1pb(
        args.model_id, resolution=args.resolution,
        enc_cfg=EncoderConfig(sh_degree=args.sh_degree), spec=MetadataSpec(),
        glob_norm=args.p1_glob_norm, p1_rank=args.p1_rank)
    n_head = sum(p.numel() for m in (model.mod_md_img, model.mod_md_txt, model.mod_md_sgl)
                 for p in m.parameters())
    print(f"P1-perblock: glob_norm={args.p1_glob_norm} glob_dim={model.d_glob} "
          f"rank={args.p1_rank} | new params: trunk="
          f"{sum(p.numel() for p in model.p1_trunk.parameters())} block heads={n_head} "
          f"norm_out head={sum(p.numel() for p in model.mod_md_out.parameters())} "
          f"total P1={sum(p.numel() for p in model.p1_parameters())}", flush=True)

    # ---- stage 0: the identity gate -------------------------------------
    if args.stage == 0:
        model.set_stage(0)
        c = model.tf.config
        B, N = 2, (args.resolution // 16) ** 2
        batch = dict(
            hidden_states=torch.randn(B, N, c.in_channels),
            encoder_hidden_states=torch.randn(B, 32, c.joint_attention_dim),
            timestep=torch.rand(B),
            img_ids=torch.zeros(N, len(c.axes_dims_rope)),
            txt_ids=torch.zeros(32, len(c.axes_dims_rope)), metadata=None)
        print(f"stage 0 PASS: max|delta| = {model.assert_identity(batch)}")
        return

    if int(os.environ.get("RANK", "0")) == 0:
        # read by inference/sampler_p1pb.py to rebuild the same P1 heads
        json.dump(dict(p1_glob_norm=args.p1_glob_norm, p1_rank=args.p1_rank,
                       glob_dim=model.d_glob, sh_degree=args.sh_degree),
                  open(os.path.join(args.out_dir, "p1pb_config.json"), "w"), indent=1)
    phase("transformer loaded")
    # full fine-tune: activation memory at 4B trainable is significant.
    # FlowSatC.forward runs the blocks itself and honours this flag.
    model.tf.enable_gradient_checkpointing()
    model.set_stage(args.stage)
    if args.init_from:
        load_weights_only(model, args.init_from)
    # Stage 1.5: the inverse of stage 1 -- metadata pathways train, backbone is
    # inert. set_stage has no mode for it (stage 1 freezes metadata and trains
    # the backbone; stage 2 trains both), so it is applied here, after set_stage
    # and before the optimizer groups are built (an empty backbone group is
    # dropped below, which DeepSpeed ZeRO requires).
    if args.freeze_backbone:
        for prm in model.tf.parameters():
            prm.requires_grad = False
        print("--freeze-backbone: FLUX.2 backbone frozen "
              f"({sum(p.numel() for p in model.tf.parameters())} params, "
              f"0 trainable); metadata pathways only", flush=True)

    # The dataloader emits pixels and caption strings; the transformer consumes
    # packed latents, prompt embeddings and 4-D ids. flux2_io bridges them.
    # Both are frozen and never receive gradients.
    phase("loading frozen VAE + Qwen3-4B text encoder ...")
    lat_enc, txt_enc = load_flux2_io(args.model_id, acc.device,
                                     max_sequence_length=args.max_text_tokens)
    phase("VAE + text encoder loaded; scanning train manifest ...")

    train_policy = CaptionPolicy()
    if args.caption_null_frac is not None:
        f = float(args.caption_null_frac)
        assert 0.0 <= f <= 1.0, "--caption-null-frac must be in [0, 1]"
        rest = 1.0 - f
        train_policy = CaptionPolicy(full=rest * 0.65 / 0.90,
                                     terse=rest * 0.25 / 0.90, null=f)
    if args.caption_mix is not None:
        fr = [float(x) for x in args.caption_mix.split(",")]
        assert len(fr) == 3 and min(fr) >= 0 and abs(sum(fr) - 1) < 1e-6, \
            "--caption-mix needs three non-negative fractions summing to 1"
        train_policy = CaptionPolicy(full=fr[0], terse=fr[1], null=fr[2])
    # printed, not implied: the caption budget is the experiment here
    print(f"caption policy: full={train_policy.full:.3f} "
          f"terse={train_policy.terse:.3f} null={train_policy.null:.3f}"
          + ("  (metadata-only phase: the null bucket is the empty string "
             "through the Qwen3 chat template -- the model's learned CFG "
             "unconditional, not a zero tensor)" if train_policy.null >= 1.0
             else ""), flush=True)
    ds = FMoWCaptionDataset(args.manifest, resolution=args.resolution,
                            caption_policy=train_policy, seed=args.seed)
    phase(f"train manifest: {len(ds)} records; computing dataset stats ...")
    if acc.is_main_process:
        print(json.dumps(ds.pair_stats(), indent=2, default=str), flush=True)

    # Pairs must land in the SAME optimizer step. Under DDP that holds even when
    # the two members sit on different ranks, because gradients are all-reduced
    # before the step -- but only if grad_accum boundaries do not split a batch,
    # which the derived grad_accum above guarantees.
    sampler = (CounterfactualPairSampler(ds, args.per_gpu_batch,
                                         pair_frac=args.pair_frac,
                                         seed=args.seed + acc.process_index)
               if args.stage >= 2 and args.pair_frac > 0 else None)
    # 4 workers (train.py:379 has 8): shared-machine limit: 2-4 per job
    dl = DataLoader(ds, batch_size=args.per_gpu_batch, sampler=sampler,
                    shuffle=(sampler is None), num_workers=4, pin_memory=True,
                    drop_last=True, persistent_workers=True,
                    collate_fn=make_collate())

    val_dl, bank, fidclip, best = None, None, None, None
    if args.val_manifest:
        phase("scanning val manifest ...")
        val_ds = FMoWCaptionDataset(args.val_manifest, resolution=args.resolution,
                                    caption_policy=CaptionPolicy(1.0, 0.0, 0.0),
                                    seed=0)
        # category-interleaved fixed order: the manifest is sorted by category,
        # so an unshuffled prefix covered only 7 of 61 classes
        from torch.utils.data import Subset
        from flowsat_C.training.validation import category_balanced_order
        val_ds = Subset(val_ds, category_balanced_order(val_ds.records))
        val_dl = DataLoader(val_ds, batch_size=args.per_gpu_batch, shuffle=False,
                            num_workers=2, drop_last=True,
                            collate_fn=make_collate())
        best = BestTracker(args.out_dir)
    # val_dl is not passed through acc.prepare, so its tensors (metadata
    # included) must be moved to the device explicitly
    val_prep = lambda b: prepare_batch(send_to_device(b, acc.device),
                                       lat_enc, txt_enc)
    if val_dl is not None and acc.is_main_process:
        from flowsat_C.training.validation import FIDCLIP
        fidclip = FIDCLIP(acc.device,
                          cache_path=os.path.join(args.out_dir, "fid_ref_balanced.pt"))

    grad_stats, grad_probe = {}, args.stage >= 2

    if args.stage >= 2 and args.gate_init is not None:
        with torch.no_grad():
            model.gate_bias.fill_(args.gate_init)
        print(f"gate_bias reset to {args.gate_init}", flush=True)
    new_p, new_nd, bb_p, bb_nd, gate_p, head_p = [], [], [], [], [], []
    head_ids = {id(p) for p in model.p1_parameters()}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n == "gate_bias":
            gate_p.append(p)
            continue
        if id(p) in head_ids:          # P1 trunk + every head: own LR, no decay
            head_p.append(p)
            continue
        # No weight decay on any 1-D tensor: biases, LayerNorm weights,
        # alpha_double/alpha_single, the SH per-degree log_a. Decay on these
        # erodes small conditioning signals toward zero; introduced in stage 1.5
        # v2 and kept, so v2 and v3 stay comparable at matched step.
        tgt = (bb_nd if p.ndim <= 1 else bb_p) if n.startswith("tf.") else \
              (new_nd if p.ndim <= 1 else new_p)
        tgt.append(p)
    bb_lr = args.lr * (args.backbone_lr_mult if args.stage >= 2 else 1.0)
    if acc.is_main_process:
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_fr = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"trainable: P1 heads+trunk={sum(p.numel() for p in head_p)} "
              f"(lr {args.head_lr:g}, wd 0) | encoder/P2 tokens="
              f"{sum(p.numel() for p in new_p + new_nd)} (lr {args.lr:g}, wd 0.01 / 0 on "
              f"{len(new_nd)} 1-D) | gate={sum(p.numel() for p in gate_p)} "
              f"(effective gate-lr {args.gate_lr:g}, wd 0) | backbone="
              f"{sum(p.numel() for p in bb_p + bb_nd)} (lr {bb_lr:g})", flush=True)
        print(f"parameters: trainable {n_tr} | frozen {n_fr}", flush=True)
    groups = [{"params": head_p, "lr": args.head_lr, "weight_decay": 0.0},
              {"params": new_p, "lr": args.lr},
              {"params": new_nd, "lr": args.lr, "weight_decay": 0.0},
              {"params": gate_p, "lr": args.gate_lr, "weight_decay": 0.0},
              # backbone at reduced LR from stage 2 onward, so pretrained weights
              # are not disrupted while the metadata pathways learn from scratch
              {"params": bb_p, "lr": bb_lr},
              {"params": bb_nd, "lr": bb_lr, "weight_decay": 0.0}]
    # stage 1 has no new-module params; DeepSpeed ZeRO rejects empty groups
    opt = torch.optim.AdamW(
        [g for g in groups if g["params"]],
        betas=(0.9, 0.95), weight_decay=0.01, eps=1e-8)
    # accelerate's AcceleratedScheduler calls scheduler.step() num_processes
    # times per optimizer step (split_batches=False), so LambdaLR's counter runs
    # at world x the optimizer step. Divide it back out: the first run (before
    # this fix) had an effective 500-step warmup and a cosine ending at 50k.
    def lr_lambda(s):
        s = s // world
        return (min(1.0, (s + 1) / args.warmup) *
                0.5 * (1 + math.cos(math.pi * min(s / args.max_steps, 1.0))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    phase("DeepSpeed / accelerate prepare (moves model to GPUs, shards optimizer) ...")
    model, opt, dl, sched = acc.prepare(model, opt, dl, sched)
    phase("prepare done")
    if args.resume:
        phase(f"resuming full state from {args.resume} ...")
        acc.load_state(args.resume)

    if args.verify_precision:
        verify_precision(args, acc, model, dl, lat_enc, txt_enc)
        acc.end_training()
        return

    # EMA starts AFTER the stage-2 gates begin opening; from step 0 the average
    # would be dominated by the pre-metadata model.
    ema = None
    if args.ema and args.stage >= 2:
        from torch.optim.swa_utils import AveragedModel
        ema = AveragedModel(model, avg_fn=lambda a, b, _:
                            args.ema_decay * a + (1 - args.ema_decay) * b)

    diag = DiagnosticLogger(acc, args.out_dir)
    # tensorboard hparams accepts only scalars/str: stringify lists and None.
    # init_trackers only initialises on the main process.
    acc.init_trackers(
        args.wandb_project,
        config={k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                for k, v in vars(args).items()},
        init_kwargs={"wandb": {
            "name": args.run_name,
            # group by stage so stage1 and stage2 sit in one comparable view
            "group": f"stage{args.stage}",
            "tags": [f"stage{args.stage}", "flux2-klein-base-4b", "full-ft",
                     f"res{args.resolution}", f"gb{args.global_batch}"],
            "notes": "FlowSAT-C metadata controllability",
        }})
    if acc.is_main_process:
        try:
            # acc.log(..., step=step) makes W&B's x-axis the optimizer step
            run = acc.get_tracker("wandb", unwrap=True)
            print(f"wandb: mode={os.environ['WANDB_MODE']} "
                  f"url={getattr(run, 'url', None)} dir={run.dir}", flush=True)
        except Exception as e:
            print(f"wandb setup note: {e}", flush=True)
    # eval_s: time spent in val/FID, excluded from img_s so throughput stays
    # a training number
    step, t0, last_val, eval_s = 0, time.time(), None, 0.0
    best_val = float("inf")
    bj = os.path.join(args.out_dir, "best", "BEST_STEP.json")
    if args.resume and os.path.exists(bj):     # a resume must not overwrite a better best/
        best_val = json.load(open(bj)).get("val/loss", best_val)
    if args.resume:
        # the scheduler is restored by load_state; the loop counter must match it
        # or logging, eval cadence and max_steps are all offset
        tail = os.path.basename(os.path.normpath(args.resume))
        if tail.startswith("ckpt-") and tail[5:].isdigit():
            step = int(tail[5:])
        else:
            step = int(getattr(sched, "scheduler", sched).last_epoch) // world
        phase(f"resumed at step {step} (lr {sched.get_last_lr()[0]:.3e})")
    step0 = step
    phase(f"starting training: {args.max_steps} steps, {args.global_batch} img/step, "
          f"{len(ds)} train images (~{len(ds) / args.global_batch:.0f} steps/epoch). "
          f"First optimizer step = {grad_accum} micro-batches per GPU.")
    last_pred, last_grad_norm = [None], [None]
    plateau = PlateauTracker(args.plateau_window, args.plateau_tol)
    p2_drop_n, p2_drop_tot, stop_now = 0, 0, False
    p1_drop_n = 0
    phase(f"P1 path dropout: {args.p1_path_dropout:.3f}; "
          f"P2 path dropout: {args.p2_path_dropout_schedule} "
          f"(p at step {step} = {p2_drop_prob(args, step):.3f}); "
          f"stop-on-plateau={'on' if args.stop_on_plateau else 'off'} "
          f"(window {args.plateau_window}, tol {args.plateau_tol:g})")
    bar = tqdm(total=args.max_steps, initial=step, desc="train", unit="step",
               dynamic_ncols=True, disable=not acc.is_main_process)

    while step < args.max_steps:
        for batch in dl:
            batch = prepare_batch(batch, lat_enc, txt_enc)
            with acc.accumulate(model):
                latents = batch["latents"]
                t = sample_timesteps(latents.shape[0], latents.device,
                                     mean=args.t_mean, std=args.t_std,
                                     shift=args.t_shift,
                                     high_noise_frac=args.high_noise_frac)
                # per SAMPLE, not per batch: each record independently loses P2
                p_drop = p2_drop_prob(args, step) if args.stage >= 2 else 0.0
                p2_drop = torch.rand(latents.shape[0], device=latents.device) < p_drop
                p2_drop_n += int(p2_drop.sum())
                p2_drop_tot += p2_drop.numel()
                p1_p = args.p1_path_dropout if args.stage >= 2 else 0.0
                p1_drop = (torch.rand(latents.shape[0], device=latents.device) < p1_p
                           if p1_p > 0 else None)
                p1_drop_n += int(p1_drop.sum()) if p1_drop is not None else 0

                def model_fn(x_t, ts):
                    out = model(hidden_states=x_t,
                                 encoder_hidden_states=batch["text_emb"],
                                 timestep=ts, img_ids=batch["img_ids"],
                                 txt_ids=batch["txt_ids"],
                                 metadata=batch.get("metadata"),
                                 training_dropout=(args.stage >= 2),
                                 p2_drop=p2_drop, p1_drop=p1_drop,
                                 collect_diagnostics=((step + 1) % args.log_every == 0))
                    last_pred[0] = out.detach()
                    return out

                loss, per_sample, x_t, target = rectified_flow_loss(model_fn, latents, t)
                extra = {}
                if args.lambda_collin > 0 and args.stage >= 2:
                    from flowsat_C.models.encoder import GROUPS
                    full = model_fn(x_t, t)
                    d = {g: full - model(hidden_states=x_t,
                                         encoder_hidden_states=batch["text_emb"],
                                         timestep=t, img_ids=batch["img_ids"],
                                         txt_ids=batch["txt_ids"],
                                         metadata=batch["metadata"], force_null=[g])
                         for g in GROUPS}
                    pen = collinearity_penalty(d)
                    loss = loss + args.lambda_collin * pen
                    extra["collin"] = pen.item()

                eng = getattr(acc, "deepspeed_engine_wrapped", None)
                report = step < args.grad_report_steps
                if (grad_probe and acc.sync_gradients and eng is not None
                        and ((step + 1) % args.log_every == 0 or report)):
                    # identical to acc.backward under DeepSpeed, split so the
                    # accumulated grads can be read before engine.step() zeroes them
                    eng.engine.set_gradient_accumulation_boundary(is_boundary=True)
                    eng.engine.backward(loss)
                    grad_stats["grads"] = pathway_grad_norms(acc.unwrap_model(model))
                    if report:
                        tab = head_grad_table(acc.unwrap_model(model))
                        if acc.is_main_process:
                            H = tab["heads"]
                            nz = lambda j, ks: sum(H[k][j] > 0 for k in ks)
                            blk = [k for k in H if k != "normout"]
                            tqdm.write(
                                f"GRAD step {step + 1}: " + " ".join(
                                    f"{k}={v:.3e}" for k, v in grad_stats["grads"].items())
                                + f" | U_k weight grads non-zero {nz(0, blk)}/{len(blk)}, "
                                f"D_k {nz(1, blk)}/{len(blk)}, norm_out U "
                                f"{H['normout'][0]:.3e} D {H['normout'][1]:.3e} | backbone "
                                f"tensors with grad {tab['backbone_tensors_with_grad']}")
                            tqdm.write(f"GRAD step {step + 1} per head (U, D): " + " ".join(
                                f"{k}=({u:.2e},{d:.2e})" for k, (u, d) in H.items()))
                    eng.engine.step()
                else:
                    acc.backward(loss)
                if acc.sync_gradients:
                    gn = acc.clip_grad_norm_(model.parameters(), 1.0)
                    if gn is not None:          # None under DeepSpeed (engine clips)
                        last_grad_norm[0] = float(gn)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

            if not acc.sync_gradients:
                continue
            step += 1
            bar.update(1)
            if step == 1:
                phase(f"first optimizer step done ({time.time() - t0:.0f}s)")
            if ema is not None:
                ema.update_parameters(model)

            if step % args.log_every == 0:
                logs = {"loss": loss.item(), "lr": sched.get_last_lr()[0],
                        "img_s": (step - step0) * args.global_batch
                        / max(time.time() - t0 - eval_s, 1e-6), **extra}
                # v1's health signals: direction agreement of predicted vs target
                # velocity (1 = perfect), and the pre-clip global gradient norm
                with torch.no_grad():
                    logs["vel_cos"] = torch.nn.functional.cosine_similarity(
                        last_pred[0].float().flatten(1), target.float().flatten(1),
                        dim=1).mean().item()
                ds_gn = getattr(model, "get_global_grad_norm", None)
                gn = ds_gn() if callable(ds_gn) else last_grad_norm[0]
                if gn is not None:
                    logs["grad_norm"] = float(gn)
                logs.update(loss_by_timestep_bucket(per_sample, t))
                logs.update(diag.gather(acc.unwrap_model(model)))
                um_d = acc.unwrap_model(model)._diag
                logs.update({k: v for k, v in um_d.items() if k.startswith("p1/")})
                # fraction of samples (this rank, since the last log) with P2 dropped
                logs["p2/drop_frac"] = p2_drop_n / max(p2_drop_tot, 1)
                logs["p2/drop_p"] = p2_drop_prob(args, step)
                logs["p1/drop_frac"] = p1_drop_n / max(p2_drop_tot, 1)
                p2_drop_n, p2_drop_tot, p1_drop_n = 0, 0, 0
                if grad_probe:
                    logs.update(grad_stats.pop("grads", {}))
                acc.log(logs, step=step)
                plateau.update(step, logs)
                if acc.is_main_process:
                    bar.set_postfix(loss=f"{logs['loss']:.4f}",
                                    img_s=f"{logs['img_s']:.2f}",
                                    lr=f"{logs['lr']:.2e}", refresh=False)
                    # one short human-readable line; every metric is in W&B
                    gates = (f"gate {logs.get('gate/bias', 0):.3f} | P1 dmu/mu per-rec "
                             f"{logs.get('p1/per_dmu_blocks', 0):.2e} const "
                             f"{logs.get('p1/const_dmu_blocks', 0):.2e} (d0 "
                             f"{logs.get('p1/per_dmu_double0', 0):.1e}, out "
                             f"{logs.get('p1/per_dmu_normout', 0):.1e}) | P2 drop "
                             f"{logs['p2/drop_frac']:.2f} | P1 drop "
                             f"{logs['p1/drop_frac']:.2f}")
                    train_s = max(time.time() - t0 - eval_s, 1e-6)
                    eta_h = (args.max_steps - step) * train_s / max(step - step0, 1) / 3600
                    tqdm.write(f"step {step:>6}/{args.max_steps} | loss {logs['loss']:.4f}"
                               f" | vel_cos {logs['vel_cos']:.3f}"
                               + (f" | grad {logs['grad_norm']:.3g}" if "grad_norm" in logs else "")
                               + f" | lr {logs['lr']:.2e} | {logs['img_s']:.2f} img/s"
                               f" | ETA {eta_h / 24:.1f} d | {gates}")
                if args.stop_on_plateau:
                    verdict = [False, {}]
                    if acc.is_main_process:
                        verdict = list(plateau.check(step))
                    broadcast_object_list(verdict)
                    if verdict[1] and acc.is_main_process and step % 500 == 0:
                        tqdm.write(f"PLATEAU step {step} | rel change over "
                                   f"{args.plateau_window}: " + " ".join(
                                       f"{k}={v:.3f}" for k, v in verdict[1].items()))
                    if verdict[0]:
                        phase(f"STOP ON PLATEAU at step {step}: " + " ".join(
                            f"{k} {v:.3f}" for k, v in verdict[1].items())
                            + f" all < {args.plateau_tol:g}")
                        stop_now = True

            # FID falling is NOT evidence controllability is alive
            if args.stage >= 2 and step % args.probe_every == 0:
                try:
                    um = acc.unwrap_model(model)
                    p = diag.controllability_probe(um, batch)
                    p.update(diag.attention_mass(um, batch))
                    acc.log(p, step=step); diag.write(step, p)
                    if acc.is_main_process:
                        tqdm.write(f"PROBE step {step:>6} | " + " ".join(
                            f"{k[5:]}={v:.3g}" for k, v in p.items()
                            if k.startswith("ctrl/")) + " | " + " ".join(
                            f"{k[5:]}={v:.2e}" for k, v in p.items()
                            if k in ("attn/txt->md", "attn/img->md")))
                except Exception as e:          # a probe must never kill a run
                    phase(f"probe FAILED at step {step}: {e!r}")

            t_eval = time.time()
            # checkpoint BEFORE val/FID at the same step: a failure inside
            # evaluation must never cost training progress
            if step % args.save_every == 0 or (stop_now and step % args.save_every):
                save_ckpt(acc, os.path.join(args.out_dir, f"ckpt-{step}"))
                if acc.is_main_process:
                    rotate_checkpoints(args.out_dir, args.keep_last)
            if args.snapshot_every and step % args.snapshot_every == 0:
                save_snapshot(acc, model, os.path.join(
                    args.out_dir, "snapshots", f"step-{step:06d}.pt"))
            if val_dl is not None and step % args.val_every == 0:
                m = acc.unwrap_model(model)
                phase(f"validation loss at step {step} ({args.val_batches} batches) ...")
                if bank is None:
                    bank = FixedNoiseBank(args.val_batches * args.per_gpu_batch,
                                          batch["latents"].shape[1:],
                                          batch["latents"].device)
                v = validation_loss(m, val_dl, bank, args.val_batches, prep=val_prep)
                acc.log(v, step=step); diag.write(step, v)
                last_val = v
                if args.best_on == "val_loss":
                    vl = float(v["val/loss"])
                    is_best = [math.isfinite(vl) and vl < best_val]
                    broadcast_object_list(is_best)
                    if is_best[0]:
                        best_val = vl
                        save_best(acc, args.out_dir, step, {"val/loss": vl})
                if acc.is_main_process:
                    hi = [x for k, x in v.items() if k.startswith("val/loss_t")
                          and float(k[len("val/loss_t"):]) >= 0.75]
                    tqdm.write(f"VAL  step {step:>6} | val loss {v['val/loss']:.4f}"
                               + (f" | high-noise (t>=0.75) {sum(hi) / len(hi):.4f}"
                                  if hi else "") + f" | n={v['val/n']}")

            if val_dl is not None and step % args.fid_every == 0:
                from flowsat_C.training.validation import validation_full
                is_best = [False]
                if acc.is_main_process:
                    grid = os.path.join(args.out_dir, f"samples_{step:06d}.png")
                    phase(f"FID/CLIP at step {step}: sampling {args.fid_samples} images "
                          f"(other GPU waits) ...")
                    t_fid = time.time()
                    try:
                        # w_md=1.0: metadata pathways are shut in stage 1, so
                        # metadata guidance is meaningless there
                        f = validation_full(acc.unwrap_model(model), val_dl, lat_enc,
                                            txt_enc, fidclip, val_prep, args.out_dir,
                                            step, n_samples=args.fid_samples,
                                            w_md=1.0 if args.stage < 2 else 2.0,
                                            grid_path=grid)
                        f["val/fid_seconds"] = time.time() - t_fid
                        if args.stage >= 2:
                            from flowsat_C.training.validation import controllability_eval
                            phase(f"controllability sweeps at step {step} "
                                  f"(gsd/cloud/season/geo x {args.ctrl_sites} sites) ...")
                            t_c = time.time()
                            c, cgrids = controllability_eval(
                                acc.unwrap_model(model), val_dl, lat_enc, val_prep,
                                args.out_dir, step, n_sites=args.ctrl_sites, w_md=2.0)
                            f.update(c)
                            for key, path, cap in cgrids:
                                log_image_wandb(acc, key, path, step, caption=cap)
                            tqdm.write(f"CTRL step {step:>6} | " + " ".join(
                                f"{k[5:]}={v:.3f}" for k, v in c.items())
                                + f" | took {(time.time() - t_c) / 60:.0f} min")
                        acc.log(f, step=step); diag.write(step, f)
                        tqdm.write(f"FID  step {step:>6} | FID {f['val/fid']:.2f}"
                                   f" | CLIP {f['val/clip']:.2f} | n={f['val/n_fid']}"
                                   f" | took {f['val/fid_seconds'] / 60:.0f} min"
                                   f" | grid {os.path.basename(grid)}")
                        # selection is on FID *and* controllability: picking on FID
                        # alone reproduces the v1 failure by construction, since the
                        # best-FID checkpoint can be the one where metadata is inert
                        is_best = [best is not None and args.best_on == "score" and
                                   best.update(step, {**(last_val or {}), **f})]
                    except Exception as e:      # an eval bug must never kill a run
                        import traceback
                        traceback.print_exc()
                        torch.cuda.empty_cache()
                        phase(f"FID/CLIP FAILED at step {step}: {e!r} (training continues)")
                    model.train()
                    log_image_wandb(acc, "samples", grid, step)
                # Other ranks wait here while rank 0 generates. save_state is
                # collective under DeepSpeed, so every rank must call it: the
                # main-process decision is broadcast.
                broadcast_object_list(is_best)
                if is_best[0]:
                    save_best(acc, args.out_dir, step, {"criterion": "score"})

            if (val_dl is not None and step % args.grid_every == 0
                    and step % args.fid_every != 0
                    and (args.grid_until <= 0 or step <= args.grid_until)):
                if acc.is_main_process:
                    from flowsat_C.training.validation import sample_grid_val
                    grid = os.path.join(args.out_dir, f"samples_{step:06d}.png")
                    phase(f"sample grid at step {step} (16 images, other GPU waits) ...")
                    try:
                        t_g = time.time()
                        sample_grid_val(acc.unwrap_model(model), val_dl, lat_enc,
                                        val_prep, grid, n=16,
                                        w_md=1.0 if args.stage < 2 else 2.0)
                        log_image_wandb(acc, "samples", grid, step)
                        phase(f"sample grid saved: {os.path.basename(grid)} "
                              f"({time.time() - t_g:.0f}s)")
                        if args.grid_null_text:
                            # the collapse check: with the caption nulled, any
                            # structure in the image came from metadata (or from
                            # the backbone's prior), and a collapse to the
                            # conditional mean shows up here, not in the
                            # text-conditioned grid above
                            ngrid = os.path.join(
                                args.out_dir, f"samples_nulltext_{step:06d}.png")
                            sample_grid_val(acc.unwrap_model(model), val_dl,
                                            lat_enc, val_prep, ngrid, n=16,
                                            w_md=1.0 if args.stage < 2 else 2.0,
                                            null_text=True)
                            log_image_wandb(acc, "samples_nulltext", ngrid, step)
                            phase(f"metadata-only grid saved: "
                                  f"{os.path.basename(ngrid)}")
                    except Exception as e:      # a grid must never kill a run
                        torch.cuda.empty_cache()
                        phase(f"sample grid FAILED at step {step}: {e!r}")
                # keep ranks in step while rank 0 generates
                broadcast_object_list([step])
            eval_s += time.time() - t_eval

            if args.halt_at_step and step >= args.halt_at_step:   # 13 Nov: stop early, LR schedule still set by --max-steps
                stop_now = True
            if step >= args.max_steps or stop_now:
                break
        if stop_now:
            break

    bar.close()
    phase("training loop finished")
    if args.keep_stage_checkpoint:
        final = os.path.join(args.out_dir, f"stage{args.stage}-final")
        save_ckpt(acc, final)
    if acc.is_main_process and args.keep_stage_checkpoint:
        shutil.copy(os.path.join(args.out_dir, "diagnostics.jsonl"),
                    os.path.join(final, "diagnostics.jsonl")) if os.path.exists(
                        os.path.join(args.out_dir, "diagnostics.jsonl")) else None
        print(f"stage {args.stage} artifact preserved -> {final}")

    acc.end_training()


if __name__ == "__main__":
    main()