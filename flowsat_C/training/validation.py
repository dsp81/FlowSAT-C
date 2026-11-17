"""
validation.py — held-out evaluation for FlowSAT-C.

WHY THIS EXISTS
v1 failed invisibly: FID fell, CLIP rose, controllability was ~0, and nothing
being logged would have shown it. Loss curves cannot detect the failure this
project is about. Every metric here is chosen so that a run going wrong is
visible from the log.

SPLIT: captions_v3_agnostic is TRAIN-ONLY, so the fMoW val/ split is unusable
(no captions). We hold out a slice of train instead, and we hold out by
LOCATION, not by image. fMoW has 4.36 views per location; an image-level split
puts the same site in both train and val and every metric becomes optimistic.
Deterministic md5 of location_id, so the split is identical across runs,
machines and GPU counts with no state to carry.

VAL LOSS: timesteps are FIXED and stratified, never resampled. Random t makes
val loss noisy enough to hide a real regression. Same t, same noise, same
samples on every evaluation -> differences are the model, not the draw.
"""

from __future__ import annotations

import hashlib
import json
import math
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from flowsat_C.training.flow_matching import rectified_flow_loss
from flowsat_C.data.metadata import log_uniform_sweep
from flowsat_C.models.encoder import GROUPS


# ------------------------------------------------------------------ split


def _bar(**kw):
    """Progress bar on rank 0 only."""
    return tqdm(dynamic_ncols=True, leave=False,
                disable=int(os.environ.get("RANK", "0")) != 0, **kw)


def val_location_ids(records, frac: float = 0.02, salt: str = "flowsatc") -> set:
    """Deterministic location-level holdout. No RNG, no state, no drift."""
    keep = set()
    thresh = int(frac * (1 << 32))
    for r in records:
        loc = r["location_id"]
        h = int(hashlib.md5(f"{salt}:{loc}".encode()).hexdigest()[:8], 16)
        if h < thresh:
            keep.add(loc)
    return keep


def split_manifest(manifest_path: str, out_train: str, out_val: str,
                   frac: float = 0.02) -> dict:
    recs = [json.loads(l) for l in open(manifest_path) if l.strip()]
    val_locs = val_location_ids(recs, frac)
    n_t = n_v = 0
    with open(out_train, "w") as ft, open(out_val, "w") as fv:
        for r in recs:
            if r["location_id"] in val_locs:
                fv.write(json.dumps(r) + "\n"); n_v += 1
            else:
                ft.write(json.dumps(r) + "\n"); n_t += 1
    st = dict(train=n_t, val=n_v, val_locations=len(val_locs),
              val_frac=n_v / max(n_t + n_v, 1))
    print(json.dumps(st, indent=2))
    return st


def category_balanced_order(records, seed: int = 0) -> list[int]:
    """Deterministic val order that interleaves categories. The val manifest is
    sorted by category, so the first N records of an unshuffled loader were
    7 of 61 classes (first 512: airport..barn; first 16: 4 airports) -- grids,
    val loss and FID all saw only that slice. Round-robin over categories,
    shuffled (seeded) within each: any prefix is as category-balanced as
    possible, and the order is identical on every evaluation."""
    import random
    by_cat = {}
    for i, r in enumerate(records):
        by_cat.setdefault(r["category"], []).append(i)
    rng = random.Random(seed)
    queues = []
    for c in sorted(by_cat):
        idx = by_cat[c][:]
        rng.shuffle(idx)
        queues.append(idx)
    order = []
    for k in range(max(len(q) for q in queues)):
        order.extend(q[k] for q in queues if k < len(q))
    return order


# ------------------------------------------------------------------ val loss


class FixedNoiseBank:
    """
    Frozen (timestep, noise) pairs so val loss is comparable across evaluations.
    Timesteps are stratified over `n_buckets` so every noise regime is measured
    -- structure lives at high t, texture at low t, and a regression in one can
    be invisible in the average.
    """

    def __init__(self, n_samples: int, shape, device, n_buckets: int = 8, seed=1234):
        g = torch.Generator(device="cpu").manual_seed(seed)
        edges = torch.linspace(0.02, 0.98, n_buckets + 1)
        t = torch.cat([torch.full((math.ceil(n_samples / n_buckets),),
                                  float((edges[i] + edges[i + 1]) / 2))
                       for i in range(n_buckets)])[:n_samples]
        self.t = t.to(device)
        self.noise = torch.randn(n_samples, *shape, generator=g).to(device)
        self.n_buckets = n_buckets

    def slice(self, lo, hi):
        return self.t[lo:hi], self.noise[lo:hi]


@torch.no_grad()
def validation_loss(model, val_loader, bank: FixedNoiseBank,
                    max_batches: int = 32, prep=None) -> dict:
    """prep: flux2_io.prepare_batch bound to the frozen VAE/text encoder."""
    model.eval()
    tot, n, per_bucket = 0.0, 0, {}
    bar = _bar(total=max_batches, desc="val loss", unit="batch")
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        bar.update(1)
        if prep is not None:
            batch = prep(batch)
        lat = batch["latents"]
        b = lat.shape[0]
        t, noise = bank.slice(n, n + b)
        if t.shape[0] < b:
            break
        # bank is fp32; an fp32 x_t would hit the bf16 backbone (no autocast under DeepSpeed)
        noise = noise.to(lat.dtype)

        def fn(x_t, ts):
            return model(hidden_states=x_t,
                         encoder_hidden_states=batch["text_emb"], timestep=ts,
                         img_ids=batch["img_ids"], txt_ids=batch["txt_ids"],
                         metadata=batch.get("metadata"), training_dropout=False)

        _, ps, _, _ = rectified_flow_loss(fn, lat, t, noise=noise)
        tot += ps.sum().item(); n += b
        for k in range(bank.n_buckets):
            m = ((t * bank.n_buckets).long() == k)
            if m.any():
                per_bucket.setdefault(k, []).append(ps[m].mean().item())

    bar.close()
    model.train()
    out = {"val/loss": tot / max(n, 1), "val/n": n}
    for k, v in per_bucket.items():
        out[f"val/loss_t{k / bank.n_buckets:.2f}"] = sum(v) / len(v)
    return out


# --------------------------------------------------- controllability on val


@torch.no_grad()
def controllability_sweep(model, vae, text_encode_fn, sites: list[dict],
                          oracles: dict | None = None, num_steps: int = 20,
                          w_txt: float = 3.5, w_md: float = 2.0) -> dict:
    """
    The metric that matters. For each fixed site, hold seed / caption / all other
    fields constant and sweep ONE field. Multiplicative fields are swept
    LOG-UNIFORMLY: v1 used {3.0,2.0,1.0,0.6,0.3}, whose log2 gaps are
    {0.58,1.00,0.74,1.00}, so any monotonicity statistic partly measured the
    sampling grid rather than the model.

    `oracles` maps field -> callable(images) -> predicted value. If supplied, we
    report Spearman correlation between conditioned and predicted values, which
    is the honest measure. Without oracles we report response magnitude and
    monotonicity of the latent trajectory, which detects "field does nothing"
    but not "field does the wrong thing".

    NOTE: any oracle used here must be a DIFFERENT model, trained on different
    data, from any oracle used in a training loss. Otherwise you are measuring
    what you optimised.
    """
    from flowsat_C.inference.sampler import sample
    model.eval()
    out = {}

    sweeps = {
        "gsd_m": log_uniform_sweep(0.3, 4.0, 5),
        "doy": [15.0, 105.0, 196.0, 288.0],          # linear + cyclic: uniform
        "cloud_frac": [0.0, 0.25, 0.5, 0.75],
        "lat_deg": None,                              # handled per-site below
    }

    for field, values in sweeps.items():
        deltas, preds, conds = [], [], []
        for site in sites:
            vals = values
            if field == "lat_deg":
                base = float(site["metadata"]["lat_deg"][0])
                vals = [base, -base]                  # hemisphere inversion test
            prev = None
            for v in vals:
                md = {k: (x.clone() if torch.is_tensor(x) else x)
                      for k, x in site["metadata"].items()}
                md[field] = torch.full_like(md[field], float(v))
                lat = sample(model, site["latents"], site["text_emb"],
                             site["null_text_emb"], site["img_ids"],
                             site["txt_ids"], md, num_steps=num_steps,
                             w_txt=w_txt, w_md=w_md)
                if prev is not None:
                    deltas.append((lat - prev).float().flatten().norm().item())
                prev = lat
                if oracles and field in oracles:
                    img = vae.decode(lat).sample
                    preds.append(float(oracles[field](img)))
                    conds.append(float(v))

        out[f"ctrl/{field}_response"] = sum(deltas) / max(len(deltas), 1)
        if preds:
            out[f"ctrl/{field}_spearman"] = _spearman(conds, preds)

    model.train()
    return out


def _spearman(a, b) -> float:
    def rank(x):
        o = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for j, i in enumerate(o):
            r[i] = float(j)
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) *
                    sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


# ------------------------------------------------------------------ FID / CLIP


class FIDCLIP:
    """
    Reference statistics are computed ONCE from val reals and cached. Identical
    implementation, identical preprocessing, identical sample count on every
    evaluation -- FID is not comparable otherwise, and a drifting reference set
    silently produces a falling FID that means nothing.

    Runs on ONE process. sync_on_compute=False: torchmetrics otherwise
    all-gathers states inside compute(), which deadlocks when the other ranks
    are not also computing. The real-feature states are saved/loaded
    explicitly because torchmetrics excludes metric states from state_dict.
    """

    REAL_STATES = ("real_features_sum", "real_features_cov_sum",
                   "real_features_num_samples")

    def __init__(self, device, cache_path: str | None = None, chunk: int = 32,
                 clip_name: str = "openai/clip-vit-base-patch32"):
        from torchmetrics.image.fid import FrechetInceptionDistance
        from transformers import CLIPModel, CLIPProcessor
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True,
                                            reset_real_features=False,
                                            sync_on_compute=False).to(device)
        # CLIPScore computed here, not via torchmetrics.CLIPScore: under
        # transformers 5.x get_image_features returns a ModelOutput and
        # torchmetrics 1.9 crashes on .norm (killed the run at step 4000).
        # It also did not truncate captions to CLIP's 77 tokens.
        self.clip_model = CLIPModel.from_pretrained(clip_name).to(device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(clip_name)
        self.device, self.cache_path, self.chunk = device, cache_path, chunk
        self.ref_done = False
        if cache_path and os.path.exists(cache_path):
            ref = torch.load(cache_path, map_location=device)
            for k in self.REAL_STATES:
                setattr(self.fid, k, ref[k].to(device))
            self.ref_done = int(ref["real_features_num_samples"]) > 0

    @torch.no_grad()
    def add_real(self, images):
        for i in range(0, images.shape[0], self.chunk):
            self.fid.update(images[i:i + self.chunk].clamp(0, 1).to(self.device),
                            real=True)

    def freeze_reference(self):
        if self.cache_path and not self.ref_done:
            torch.save({k: getattr(self.fid, k).detach().cpu()
                        for k in self.REAL_STATES}, self.cache_path)
        self.ref_done = True

    @torch.no_grad()
    def clip_scores(self, images, captions) -> torch.Tensor:
        """Per-image CLIPScore = 100 * max(cos(image, text), 0), as torchmetrics.
        images in [0,1]; captions truncated to 77 CLIP tokens."""
        m = self.clip_model
        inp = self.clip_proc(text=list(captions),
                             images=[(x * 255).round().to(torch.uint8).cpu() for x in images],
                             return_tensors="pt", padding=True, truncation=True,
                             max_length=77)
        # sub-model pooler_output + projection: stable across transformers versions
        img = m.visual_projection(m.vision_model(
            pixel_values=inp["pixel_values"].to(self.device)).pooler_output)
        txt = m.text_projection(m.text_model(
            input_ids=inp["input_ids"].to(self.device),
            attention_mask=inp["attention_mask"].to(self.device)).pooler_output)
        img = F.normalize(img.float(), dim=-1)
        txt = F.normalize(txt.float(), dim=-1)
        return (100 * (img * txt).sum(-1)).clamp(min=0)

    @torch.no_grad()
    def score(self, fake_images, captions) -> dict:
        # chunked: 512 images at once through Inception/CLIP on a rank that
        # already holds ~69 GB of training state would risk OOM
        clip = []
        for i in range(0, fake_images.shape[0], self.chunk):
            x = fake_images[i:i + self.chunk].clamp(0, 1).to(self.device)
            self.fid.update(x, real=False)
            clip.append(self.clip_scores(x, captions[i:i + self.chunk]).cpu())
        f, c = float(self.fid.compute()), float(torch.cat(clip).mean())
        # fakes must not accumulate across evaluations; reals are kept
        # (reset_real_features=False)
        self.fid.reset()
        return {"val/fid": f, "val/clip": c}


@torch.no_grad()
def generate_val(model, val_loader, lat_enc, prep, n_samples, num_steps=20,
                 w_txt=3.5, w_md=1.0, seed=4321, desc="sampling",
                 null_text=False):
    """First n_samples val prompts + metadata -> images in [0,1] (CPU) and their
    captions. The val loader is unshuffled and the noise is seeded, so repeated
    calls differ only by the model: grids from different steps line up."""
    from flowsat_C.inference.sampler import sample
    g = torch.Generator(device="cpu").manual_seed(seed)
    fakes, captions, n = [], [], 0
    bar = _bar(total=n_samples, desc=desc, unit="img")
    for batch in val_loader:
        batch = prep(batch)
        B = batch["latents"].shape[0]
        lat0 = batch["latents"]
        noise = torch.randn(lat0.shape, generator=g).to(lat0.device, lat0.dtype)
        if null_text:
            # metadata-only sampling: the conditional branch gets the SAME
            # learned unconditional the model trains on when the caption is
            # nulled, so the only thing left steering the image is metadata.
            batch["text_emb"] = batch["null_text_emb"]
        lat = sample(model, noise, batch["text_emb"], batch["null_text_emb"],
                     batch["img_ids"], batch["txt_ids"], batch["metadata"],
                     num_steps=num_steps, w_txt=w_txt, w_md=w_md)
        img = lat_enc.decode(lat, model.latent_hw).clamp(-1, 1).add(1).div(2)
        fakes.append(img.float().cpu())
        captions.extend(batch["caption"])
        n += B
        bar.update(B)
        if n >= n_samples:
            break
    bar.close()
    return torch.cat(fakes)[:n_samples], captions[:n_samples]


@torch.no_grad()
def sample_grid_val(model, val_loader, lat_enc, prep, grid_path, n=16,
                    num_steps=20, w_txt=3.5, w_md=1.0, seed=4321,
                    null_text=False):
    """Cheap visual check between FIDs: n images from the SAME first val prompts
    and seed as the FID grid, saved as a sqrt(n)-wide grid. ~1-2 min."""
    from torchvision.utils import save_image
    model.eval()
    fakes, captions = generate_val(model, val_loader, lat_enc, prep, n, num_steps,
                                   w_txt, w_md, seed, desc="sample grid",
                                   null_text=null_text)
    save_image(fakes, grid_path, nrow=max(int(n ** 0.5), 1))
    model.train()
    return captions


@torch.no_grad()
def validation_full(model, val_loader, lat_enc, txt_enc, fidclip,
                    prep, out_dir, step, n_samples=512, num_steps=20,
                    w_txt=3.5, w_md=1.0, grid_path=None, seed=4321):
    """
    Generate from val prompts + metadata, score FID/CLIP against val reals.

    N=512 is a deliberate compromise. FID at this N is biased HIGH and is NOT
    comparable to any published number -- including our own Table 2 at 31.10.
    It is for TRENDS and checkpoint selection only. Run a full-N FID once, on
    the selected checkpoint, for anything reportable.

    Reference statistics are built ONCE from val reals and cached: a drifting
    reference set produces a falling FID that means nothing. Val loader is
    unshuffled, so reals and prompts are the same first n_samples val images,
    and the sampling noise is seeded so evaluations differ only by the model.
    """
    model.eval()
    n = 0

    if not fidclip.ref_done:
        bar = _bar(total=n_samples, desc="FID reference (val reals, once)", unit="img")
        for batch in val_loader:
            imgs = batch["pixel_values"].add(1).div(2)      # [-1,1] -> [0,1]
            fidclip.add_real(imgs)
            n += imgs.shape[0]
            bar.update(imgs.shape[0])
            if n >= n_samples:
                break
        bar.close()
        fidclip.freeze_reference()
        n = 0

    fakes, captions = generate_val(model, val_loader, lat_enc, prep, n_samples,
                                   num_steps, w_txt, w_md, seed,
                                   desc=f"FID sampling ({num_steps} steps x 3 CFG passes)")
    # grid BEFORE scoring: at step 4000 scoring crashed and 30 min of samples
    # were lost with nothing written
    if grid_path:
        from torchvision.utils import save_image
        save_image(fakes[:16], grid_path, nrow=4)

    out = fidclip.score(fakes, captions)
    out["val/n_fid"] = len(fakes)

    model.train()
    return out


# ------------------------------------------------- controllability sweeps (live)


CTRL_SWEEPS = {
    # multiplicative -> log-uniform (see log_uniform_sweep)
    "gsd": ("gsd_m", log_uniform_sweep(0.3, 4.0, 5)),
    "cloud": ("cloud_frac", [0.0, 0.25, 0.5, 0.75]),
    "season": ("doy", [15.0, 105.0, 196.0, 288.0]),
    # (lat, lon): France, Egypt, India, Brazil, Norway -- the v2 sweep places
    "geo": (("lat_deg", "lon_deg"), [(49.0, 1.8), (30.0, 31.2), (19.0, 72.9),
                                     (-3.1, -60.0), (60.0, 10.7)]),
}


def _clone_tree(x):
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, dict):
        return {k: _clone_tree(v) for k, v in x.items()}
    return x


@torch.no_grad()
def controllability_eval(model, val_loader, lat_enc, prep, out_dir, step,
                         n_sites: int = 4, num_steps: int = 20, w_txt: float = 3.5,
                         w_md: float = 2.0, seed: int = 4321) -> tuple[dict, list]:
    """
    One field swept at a time on fixed val sites; seed, caption and every other
    field held constant. Reports, per field, the mean relative latent change
    between consecutive sweep values (ctrl/<field>_response): ~0 means the field
    is ignored -- the v1 failure. Saves a sites x values image grid per field.
    No oracle, so this detects "does nothing", not "does the wrong thing".
    """
    from flowsat_C.inference.sampler import sample
    from torchvision.utils import save_image
    model.eval()
    batch = prep(next(iter(val_loader)))
    n = min(n_sites, batch["latents"].shape[0])
    sl = lambda v: v[:n]
    g = torch.Generator(device="cpu").manual_seed(seed)
    lat0 = batch["latents"][:n]
    noise = torch.randn(lat0.shape, generator=g).to(lat0.device, lat0.dtype)
    from flowsat_C.training.diagnostics import slice_tree
    md0 = slice_tree(batch["metadata"], n)
    out, grids = {}, []
    for name, (keys, values) in CTRL_SWEEPS.items():
        lats, imgs = [], []
        for v in values:
            md = _clone_tree(md0)
            if isinstance(keys, tuple):
                for k, x in zip(keys, v):
                    md[k] = torch.full_like(md[k], float(x))
            else:
                md[keys] = torch.full_like(md[keys], float(v))
            lat = sample(model, noise, sl(batch["text_emb"]), sl(batch["null_text_emb"]),
                         batch["img_ids"], batch["txt_ids"], md, num_steps=num_steps,
                         w_txt=w_txt, w_md=w_md)
            lats.append(lat.float())
            img = lat_enc.decode(lat, model.latent_hw).clamp(-1, 1).add(1).div(2)
            imgs.append(F.interpolate(img.float(), size=256, mode="area").cpu())
        rel = [((b - a).flatten(1).norm(dim=1) / a.flatten(1).norm(dim=1).clamp(min=1e-6)).mean()
               for a, b in zip(lats[:-1], lats[1:])]
        out[f"ctrl/{name}_response"] = float(torch.stack(rel).mean())
        # rows = sites, columns = sweep values
        grid = torch.stack(imgs, 1).reshape(-1, *imgs[0].shape[1:])
        path = os.path.join(out_dir, f"ctrl_{name}_{step:06d}.png")
        save_image(grid, path, nrow=len(values))
        grids.append((f"ctrl_{name}", path, f"{name}: {values}"))
    model.train()
    return out, grids


# ------------------------------------------------------------ sample grid


@torch.no_grad()
def sample_grid(model, vae, sites, out_path, num_steps: int = 20,
                w_txt: float = 3.5, w_md: float = 2.0):
    """
    Fixed sites, fixed seed, one row per GSD sweep value. Eyeballing this every
    few thousand steps catches failures no scalar will: mode collapse, colour
    drift, a metadata pathway that has started dominating text.
    """
    from torchvision.utils import save_image
    from flowsat_C.inference.sampler import sample
    model.eval()
    rows = []
    for site in sites[:4]:
        for g in log_uniform_sweep(0.3, 4.0, 5):
            md = {k: (x.clone() if torch.is_tensor(x) else x)
                  for k, x in site["metadata"].items()}
            md["gsd_m"] = torch.full_like(md["gsd_m"], g)
            lat = sample(model, site["latents"], site["text_emb"],
                         site["null_text_emb"], site["img_ids"], site["txt_ids"],
                         md, num_steps=num_steps, w_txt=w_txt, w_md=w_md)
            rows.append(vae.decode(lat).sample.clamp(-1, 1).add(1).div(2)[0].cpu())
    save_image(torch.stack(rows), out_path, nrow=5)
    model.train()
    return out_path


# --------------------------------------------------------- checkpoint select


class BestTracker:
    """
    Selects on a COMBINED criterion. Selecting on FID alone reproduces the v1
    failure by construction: the best-FID checkpoint can be the one where
    metadata does nothing.

        score = fid_norm - lambda * mean(ctrl spearman)

    Lower is better. If no oracles are available, ctrl response magnitude is
    used, which is weaker but still refuses a checkpoint with a dead pathway.
    """

    def __init__(self, out_dir: str, lam: float = 20.0):
        self.out_dir, self.lam, self.best = out_dir, lam, float("inf")
        self.history = []

    def update(self, step: int, metrics: dict) -> bool:
        fid = metrics.get("val/fid")
        # no FID this evaluation: scoring inf - lambda*c would be meaningless
        if fid is None or not math.isfinite(fid):
            return False
        ctrl = [v for k, v in metrics.items() if k.startswith("ctrl/")
                and k.endswith("_spearman")]
        if not ctrl:
            ctrl = [v for k, v in metrics.items() if k.endswith("_response")]
            lam = self.lam * 0.01
        else:
            lam = self.lam
        c = sum(ctrl) / len(ctrl) if ctrl else 0.0
        score = fid - lam * c
        self.history.append(dict(step=step, score=score, fid=fid, ctrl=c))
        with open(os.path.join(self.out_dir, "best.jsonl"), "a") as f:
            f.write(json.dumps(self.history[-1]) + "\n")
        if score < self.best:
            self.best = score
            return True
        return False