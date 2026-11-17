"""
Stages: build-pairs -> select-spaces -> fit-readers -> plan -> generate -> score -> meta -> card.

  build-pairs     manifests -> dataset.jsonl; TRAIN pairs (VIF, joint groups decided here, on TRAIN) and
                  TEST pairs with the same design -> design.json, pairs_{train,test}.json        (CPU)
  select-spaces   features of the TRAIN pair images in every space (stats normaliser fit on TRAIN),
                  split-half reliability per field and space -> selection.json             (GPU for dino/clip)
  fit-readers     C2 real readers, C3 full/base readers, E4 order model, all on TRAIN reals -> readers.pkl
  plan            requests of every enabled measure, deduplicated -> plan.json, plan_summary.json
                  (images per measure + GPU-hour estimate from plan.seconds_per_image)     (CPU)
  generate        the adapter over the plan into the image store; --worker i --nworkers n shards the work
                  by request key so a second GPU can join                                  (GPU)
  score           features of everything needed, every measure, BH, gates, summary -> results.json, card
  meta            the meta-evaluation harness (meta/harness.py)
  card            results.json -> control_card.md / .html
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from metacf.config import get
from metacf.requests import Plan
from metacf.v2.core.pairs import build_pairs as _build_pairs
from metacf.v2.core.regression import SiteStats, split_half
from metacf.v2.core.select import save as save_selection
from metacf.v2.core.select import select_spaces as _select
from metacf.v2.run import log

PHASE_IDS = ("C0", "C1", "C2", "C3", "C4", "C5", "E1", "E2", "E3", "E4")


def measures(run, ids=None):
    from metacf.v2.measures import ALL as M
    from metacf.v2.physics import ALL as P
    out = []
    for cls in M + P:
        m = cls(run)
        if (ids is None or m.id in ids) and m.enabled():
            out.append(m)
    return out


# ---------------------------------------------------------------- build-pairs
def build_pairs(run):
    if run.ds is None:
        run.build_dataset()
    d = run.make_design()
    pc = get(run.cfg, "pairs", {}) or {}
    tr = _build_pairs(run.train, d, "train", int(pc.get("cap", 10)), run.seed("pairs_train"),
                      pc.get("max_sites_train"), float(pc.get("vif_max", 5.0)), True, log)
    te = _build_pairs(run.test, d, "test", int(pc.get("cap", 10)), run.seed("pairs_test"),
                      pc.get("max_sites_test"), float(pc.get("vif_max", 5.0)), False, log)
    run.pairs = dict(train=tr, test=te)
    run.save_json("design.json", d.to_json())
    run.save_json("pairs_train.json", tr.to_json())
    run.save_json("pairs_test.json", te.to_json())
    log(f"pairs: train {tr.n} over {tr.report.get('n_sites')} sites, test {te.n} over {te.report.get('n_sites')} "
        f"sites; joint groups {d.joint}")
    return run.pairs


def load_all(run, need_selection=True):
    if run.ds is None:
        run.load_dataset()
    if run.design is None:
        run.load_design()
    if not run.pairs:
        run.load_pairs()
    if need_selection and run.selection is None:
        run.load_selection()
    return run


# ---------------------------------------------------------------- select-spaces
def select_spaces(run):
    load_all(run, need_selection=False)
    ps = run.pairs["train"]
    sites, sidx = ps.site_index()
    sc = get(run.cfg, "selection", {}) or {}
    rel = {}
    for sp in run.space_names():
        log(f"select-spaces: {sp} on {ps.n} TRAIN pairs / {len(sites)} sites")
        Y = run.real_feats(sp, ps.b) - run.real_feats(sp, ps.a)
        ss = SiteStats(ps.dM, Y, sidx, len(sites))
        rel[sp] = split_half(ss, run.design.blocks, int(sc.get("n_splits", 50)), run.seed("sel", sp), run.B,
                             int(sc.get("ci_splits", 10)))
        rel[sp]["__fit_r2__"] = ss.r2(ss.fit())
        run.space(sp).release()
    fit = {sp: rel[sp].pop("__fit_r2__") for sp in rel}
    sel = _select(rel, float(sc.get("min_reliability", 0.5)))
    run.selection = sel
    if run.persist:
        save_selection(os.path.join(run.pdir, "selection.json"), sel,
                       dict(fit_r2_train=fit, n_pairs=ps.n, n_sites=len(sites), spaces=run.space_names(),
                            space_models={sp: run.space(sp).model_id for sp in run.space_names()}))
    for f, v in sel.items():
        log(f"  {f}: {v.get('space')} rel {v.get('reliability', float('nan')):.3f} "
            f"[{v.get('lo')}, {v.get('hi')}] {'VALID' if v['valid'] else 'INVALID ' + v['reason']}")
    return sel


# ---------------------------------------------------------------- fit-readers
def fit_readers(run):
    load_all(run)
    from metacf.v2.measures.c2_transfer import C2Transfer
    from metacf.v2.measures.c3_information import C3Information
    from metacf.v2.physics.e4_year import E4Year
    out = {}
    m2, m3, e4 = C2Transfer(run), C3Information(run), E4Year(run)
    if m2.enabled():
        out["C2"] = m2.fit_real()
        log(f"fit-readers: C2 real readers for {list(out['C2'])}")
    if m3.enabled():
        out["C3"] = m3.fit_readers()
        log(f"fit-readers: C3 readers for {list(out['C3'])}")
    if e4.enabled() and e4.field() is not None:
        out["E4"] = e4.fit_model()
        log(f"fit-readers: E4 order model (lambda {out['E4'].lam}, CV acc {out['E4'].cv_acc:.3f})")
    run.save_readers(out)
    return out


# ---------------------------------------------------------------- plan
def make_plan(run, ids=None):
    load_all(run)
    plan = Plan()
    st = run.store()
    for m in measures(run, ids):
        plan.add(m.id, m.plan(), st.model_id, st.checkpoint)
    return plan


def write_plan(run, plan):
    spi = float(get(run.cfg, "plan.seconds_per_image", 3.65))
    summ = plan.summary(spi)
    summ["by_measure_hours"] = {k: v["new_unique"] * spi / 3600 for k, v in plan.per_test.items()}
    if run.persist:
        plan.to_json(os.path.join(run.mdir, "plan.json"))
        json.dump(summ, open(os.path.join(run.mdir, "plan_summary.json"), "w"), indent=1)
    return summ


# ---------------------------------------------------------------- generate
def _shard(key, nworkers):
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % max(1, nworkers)


def _put(st, q, img):
    """ImageStore.put, tolerant of a second worker writing the same image at the same moment (both use the
    same temp name; the loser's os.replace fails but the image is there)."""
    try:
        st.put(q, img)
    except (FileNotFoundError, OSError):
        if not st.has(q):
            raise


def generate(run, plan=None, worker=0, nworkers=1, log_every=64):
    plan = plan or make_plan(run)
    st = run.store()
    ad = run.adapter
    # own shard first, then the other shards in rotation (skipping images already made), so a lone worker
    # still finishes the whole plan and a second worker joining later just takes work off it
    order = [(worker + i) % max(1, nworkers) for i in range(max(1, nworkers))]
    todo = [(k, r) for s_ in order for k, r in plan.unique.items() if _shard(k, nworkers) == s_ and not st.has(r)]
    log(f"generate: worker {worker}/{nworkers}: {len(todo)} images to make "
        f"({len(plan.unique)} in the plan)")
    if hasattr(ad, "generate_batch"):
        groups = {}
        for k, r in todo:
            groups.setdefault((_shard(k, nworkers), r.null_mask, r.guidance), []).append(r)
        bs = int(getattr(ad, "batch", 8))
        n, t0 = 0, time.time()
        for g, reqs in groups.items():
            for i in range(0, len(reqs), bs):
                chunk = reqs[i:i + bs]
                chunk = [q for q in chunk if not st.has(q)]          # another worker may have made them
                if not chunk:
                    continue
                for q, img in zip(chunk, ad.generate_batch(chunk)):
                    _put(st, q, img)
                n += len(chunk)
                if n % log_every < len(chunk) or n == len(todo):
                    rate = (time.time() - t0) / max(n, 1)
                    pm = getattr(ad, "peak_mem_gb", None)
                    log(f"  {n}/{len(todo)} images, {rate:.2f} s/img, ETA {(len(todo) - n) * rate / 3600:.1f} h"
                        + (f", peak GPU mem {pm:.1f} GiB" if pm else ""))
    else:
        for i, (k, r) in enumerate(todo):
            if not st.has(r):
                _put(st, r, np.asarray(ad.generate_request(r)))
    return len(todo)


def wait_complete(run, plan, poll=120):
    st = run.store()
    while True:
        miss = sum(1 for r in plan.unique.values() if not st.has(r))
        if miss == 0:
            return
        log(f"waiting for {miss} images from other workers")
        time.sleep(poll)


# ---------------------------------------------------------------- score
def prefetch(run, plan, ms):
    """Features of every generated image in every space a measure may read, in big batches."""
    reqs = list(plan.unique.values())
    for sp in run.space_names():
        run.gen_feats(sp, reqs, raw=True)
        run.space(sp).release()


def score(run, ids=None, plan=None, do_prefetch=True):
    load_all(run)
    ms = measures(run, ids)
    plan = plan or make_plan(run, ids)
    if do_prefetch:
        prefetch(run, plan, ms)
    rows = []
    for m in ms:
        t0 = time.time()
        try:
            rs = m.compute()
        except Exception as e:                    # a measure that cannot run is reported, not hidden
            import traceback
            from metacf.v2.core.norm import invalid
            traceback.print_exc()
            rs = [invalid(m.id, "*", "error", f"ERROR: {e!r}", status="ERROR")]
        for x in rs:
            x.setdefault("modified", False)
            x["modified"] = bool(x["modified"] or getattr(m, "modified", False) and x["metric"] == "biome")
        rows += rs
        log(f"{m.id} {m.name}: {len(rs)} rows in {time.time() - t0:.1f}s")
    from metacf.v2 import summary
    summary.apply_bh(rows, run.q)
    for m in ms:
        rows = m.finalize(rows)
    seeds_note = dict(base_seed=run.base_seed)
    extra = dict(selection=run.selection, pairs_report={k: v.report for k, v in run.pairs.items()},
                 design=run.design.to_json(), plan=plan.summary(float(get(run.cfg, "plan.seconds_per_image", 3.65))),
                 seeds=seeds_note, spaces={sp: run.space(sp).model_id for sp in run.space_names()})
    res = summary.write(run, rows, extra)
    if run.cfg.get("v2_3"):
        from metacf.v2 import summary_v23
        summary_v23.write(run, res)
    return res


def rescore(run, src_dir, ids, c1_fields=None, link_images=True):
    """v2_1: recompute only `ids` (C1 restricted to `c1_fields` when given) on the generations of an
    existing scored model dir `src_dir`, and merge with its other rows. Writes results.json + card into
    THIS run's model dir (a new config hash), never into src_dir. Images are reused through a symlink
    (request keys depend only on model id / checkpoint / request, not on the config hash). Every merged
    row goes through BH again; only the recomputed measures re-apply their gates (the other rows keep the
    status they were given in src_dir). Never generates: missing images surface as PENDING / errors."""
    load_all(run)
    src = json.load(open(os.path.join(src_dir, "results.json")))
    img = os.path.join(run.mdir, "images")
    if link_images and not os.path.exists(img):
        os.symlink(os.path.abspath(os.path.join(src_dir, "images")), img)
    ms = measures(run, ids)
    rows = []
    for m in ms:
        if m.id == "C1" and c1_fields:
            m.only_fields = set(c1_fields)
        t0 = time.time()
        try:
            rs = m.compute()
        except Exception as e:
            import traceback
            from metacf.v2.core.norm import invalid
            traceback.print_exc()
            rs = [invalid(m.id, "*", "error", f"ERROR: {e!r}", status="ERROR")]
        for x in rs:
            x.setdefault("modified", False)
        rows += rs
        log(f"rescore {m.id} {m.name}: {len(rs)} rows in {time.time() - t0:.1f}s")
    redo = {m.id for m in ms}

    def replaced(x):
        if x["measure"] not in redo:
            return False
        if x["measure"] == "C1" and c1_fields:
            return x["field"] in c1_fields
        return True
    kept = [x for x in src["rows"] if not replaced(x)]
    for x in kept:
        x["extra"] = dict(x.get("extra") or {}, from_dir=os.path.abspath(src_dir))
    allrows = kept + rows
    from metacf.v2 import summary
    summary.apply_bh(allrows, run.q)
    for m in ms:
        allrows = m.finalize(allrows)
    extra = {k: src[k] for k in ("selection", "pairs_report", "design", "plan", "seeds", "spaces") if k in src}
    extra["rescored_from"] = dict(dir=os.path.abspath(src_dir), config_hash=src.get("config_hash"),
                                  measures=sorted(redo), c1_fields=sorted(c1_fields or []))
    res = summary.write(run, allrows, extra)
    from metacf.v2 import summary_v21
    summary_v21.write(run, res)
    if run.cfg.get("v2_3"):
        from metacf.v2 import summary_v23
        summary_v23.write(run, res)
    return res


def zoom_check(run, spaces=None):
    """Real-data sanity table for the zoom reference (metacf.v2.core.zoom.check) -> <protocol>/zoom_check.json."""
    load_all(run)
    from metacf.v2.core import zoom
    res = zoom.check(run, spaces)
    if run.persist:
        p = os.path.join(run.pdir, "zoom_check.json")
        with open(p, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    return res


def card(run):
    from metacf.v2 import summary
    res = json.load(open(os.path.join(run.mdir, "results.json")))
    return summary.card(res, run.mdir)
