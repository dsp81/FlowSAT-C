"""Stage-0 gate for the P1-perblock model (copy of scripts/stage0_identity_check.py).

With every U_k zero-init the new model must equal stock FLUX.2. Checked on the
SAME random inputs as the original gate, in fp32 and in bf16, for:

  orig        the ORIGINAL code's path: FlowSatC.assert_identity (tokens off, no metadata)
  p1pb_nomd   FlowSatCP1PB, no metadata (pure stage-0 path)
  p1pb_md     FlowSatCP1PB, metadata PRESENT, P2 tokens OFF  -> P1 heads active, delta = 0
  p1pb_md_p2  FlowSatCP1PB, metadata PRESENT, P2 tokens ON at the init gate (-20)
  orig_md_p2  the ORIGINAL FlowSatC in the same state (to_temb zero, tokens on, gate -20)

max|delta| vs the stock transformer is printed for each. The first three must be
exactly 0.0. The last is reported: at gate -20 the 16 metadata keys hold ~2e-9 of
each softmax row, which is numerically nil but not structurally zero.
"""
import argparse
import json

import torch

from flowsat_C.models.encoder import MetadataSpec
from flowsat_C.models.transformer import FlowSatC
from flowsat_C.models.transformer_p1pb import FlowSatCP1PB


def synthetic_metadata(B, device):
    """Physical-unit metadata, the encoder's own self-test values (encoder.py:465-485)."""
    g = torch.Generator().manual_seed(0)
    r = lambda lo, hi: (lo + (hi - lo) * torch.rand(B, generator=g)).to(device)
    return dict(
        lat_deg=r(-60, 60), lon_deg=r(-180, 180), gsd_m=r(0.3, 4.0), cloud_frac=r(0, 0.8),
        doy=r(1, 365), year=torch.randint(2002, 2018, (B,), generator=g).to(device),
        sun_elev_deg=r(10, 80), sun_azimuth_deg=r(90, 220),
        azimuth_valid=torch.ones(B, dtype=torch.bool, device=device),
        platform=torch.randint(0, 11, (B,), generator=g).to(device),
        cov=dict(elevation_m=r(-50, 2000), relief_m=r(0, 500), coast_km=r(0, 500),
                 pop_density=r(0, 20000),
                 koppen=torch.randint(0, 31, (B,), generator=g).to(device),
                 landcover_mode=torch.randint(0, 12, (B,), generator=g).to(device),
                 landcover_hist=torch.rand(B, MetadataSpec().n_landcover_classes,
                                           generator=g).to(device)))


def run(model_id, dtype, a):
    from diffusers import Flux2Transformer2DModel
    tf = Flux2Transformer2DModel.from_pretrained(model_id, subfolder="transformer",
                                                 torch_dtype=dtype).to(a.device).eval()
    c = tf.config
    B, N = a.batch, (a.resolution // 16) ** 2
    g = torch.Generator().manual_seed(0)
    batch = dict(
        hidden_states=torch.randn(B, N, c.in_channels, generator=g).to(a.device, dtype),
        encoder_hidden_states=torch.randn(B, a.n_txt, c.joint_attention_dim,
                                          generator=g).to(a.device, dtype),
        timestep=torch.rand(B, generator=g).to(a.device, dtype),
        img_ids=torch.zeros(N, len(c.axes_dims_rope)).to(a.device),
        txt_ids=torch.zeros(a.n_txt, len(c.axes_dims_rope)).to(a.device),
        metadata=None)
    md = synthetic_metadata(B, a.device)
    res = {}
    with torch.no_grad():
        base = tf(hidden_states=batch["hidden_states"],
                  encoder_hidden_states=batch["encoder_hidden_states"],
                  timestep=batch["timestep"], img_ids=batch["img_ids"],
                  txt_ids=batch["txt_ids"], return_dict=False)[0]

        orig = FlowSatC(tf).to(a.device)
        orig.set_stage(0)
        res["orig"] = orig.assert_identity(batch)
        # attribution for p1pb_md_p2: the ORIGINAL model with P2 tokens on at the
        # same init gate (to_temb zero) on the same inputs
        orig.encoder.to(dtype)
        orig.metadata_tokens_enabled = True
        res["orig_md_p2"] = float((orig(**{**batch, "metadata": md}) - base).abs().max())
        del orig

        new = FlowSatCP1PB(tf, glob_norm=a.glob_norm, p1_rank=a.p1_rank).to(a.device)
        # DeepSpeed bf16 casts every module, the new ones included
        new.encoder.to(dtype); [m.to(dtype) for m in new.p1_modules()]
        new.set_stage(2)
        assert all(float(h.U.weight.abs().max()) == 0 and float(h.U.bias.abs().max()) == 0
                   for h in list(new.mod_md_img) + list(new.mod_md_txt) + list(new.mod_md_sgl)
                   + [new.mod_md_out]), "U_k not zero at init"
        gate = float(new.gate_bias)
        diff = lambda o: float((o - base).abs().max())
        new.metadata_tokens_enabled = False
        res["p1pb_nomd"] = diff(new(**batch))
        res["p1pb_md"] = diff(new(**{**batch, "metadata": md}))
        new.metadata_tokens_enabled = True
        res["p1pb_md_p2"] = diff(new(**{**batch, "metadata": md}))
        # the heads really ran: delta is an exact zero tensor of the right shape
        new._precision_probe = []
        new(**{**batch, "metadata": md})
        res["p1_additions"] = len(new._precision_probe)
        res["gate_init"] = gate
    del tf, new
    torch.cuda.empty_cache()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="black-forest-labs/FLUX.2-klein-base-4B")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--n-txt", type=int, default=32)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--glob-norm", default="ln_mag")
    p.add_argument("--p1-rank", type=int, default=128)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    torch.set_num_threads(4)

    out = {}
    for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        out[name] = run(a.model_id, dt, a)
        print(f"[{name}] " + " ".join(f"{k}={v}" for k, v in out[name].items()), flush=True)
    ok = all(out[d][k] == 0.0 for d in out for k in ("orig", "p1pb_nomd", "p1pb_md"))
    print(f"stage 0 P1-perblock {'PASS' if ok else 'FAIL'}: "
          f"max|delta| fp32 {out['fp32']['p1pb_md']} / bf16 {out['bf16']['p1pb_md']} "
          f"(metadata present, heads zero); with P2 tokens at gate "
          f"{out['bf16']['gate_init']}: fp32 {out['fp32']['p1pb_md_p2']} / "
          f"bf16 {out['bf16']['p1pb_md_p2']}", flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    assert ok


if __name__ == "__main__":
    main()
