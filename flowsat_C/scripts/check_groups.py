"""Gate 2: every metadata group must be wired into the forward pass.

A disconnected field trains without error and surfaces months later as
"geolocation has no controllability" -- indistinguishable from a genuine
negative result, which is the conclusion this project exists to draw. Eight
forward passes buy that distinction.

WHY THE GATES ARE OPENED FIRST. At init the model is deliberately an exact
no-op for metadata: to_temb is zero-init and gate_bias is -20
(~2e-11 of softmax mass). Comparing forward(force_null=[g]) against
forward(force_null=None) on the as-initialised model therefore returns ~0 for
EVERY group -- it would report all eight disconnected, which is the same signal
as the failure this gate is meant to detect. So the test perturbs the zero-inits
and sets gate_bias=0 first. This is a WIRING test on a throwaway model object,
not a statement about the trained model; nothing is saved.

Two levels are reported because they fail differently:
  encoder  -- does nulling the group change (tokens, global)?  If not, the group
              is not wired into the metadata representation at all.
  model    -- does that change reach the transformer output?  If the encoder
              moves and the model does not, the graft is the broken link.
"""
import argparse
import json
import sys

import torch

from flowsat_C.data.dataset import FMoWCaptionDataset
from flowsat_C.data.metadata import collate_metadata
from flowsat_C.models.encoder import GROUPS
from flowsat_C.models.transformer import build_flowsat_c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--n-txt", type=int, default=32)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    model = build_flowsat_c(a.model_id, resolution=a.resolution)
    model.set_stage(2)
    model = model.to(a.device).eval()

    # open the deliberately-shut gates so a real difference is observable
    with torch.no_grad():
        model.to_temb.weight.normal_(0.0, 0.02)
        model.gate_bias.fill_(0.0)
    print("gates opened for the wiring test "
          "(to_temb randomised, gate_bias=0); nothing is saved\n")

    # real metadata, not synthetic -- bug 6 hid behind a self-built tensor
    ds = FMoWCaptionDataset(a.manifest, return_pixels=False)
    n = min(a.batch, len(ds))
    items = [ds[i] for i in range(n)]
    md = collate_metadata([it["_md"] for it in items], [it["_cov"] for it in items])
    md = {k: (v.to(a.device) if torch.is_tensor(v) else
              {kk: vv.to(a.device) for kk, vv in v.items()}) for k, v in md.items()}

    c = model.tf.config
    N = (a.resolution // 16) ** 2
    g = torch.Generator(device="cpu").manual_seed(0)
    dt = next(model.tf.parameters()).dtype
    batch = dict(
        hidden_states=torch.randn(n, N, c.in_channels, generator=g).to(a.device, dt),
        encoder_hidden_states=torch.randn(n, a.n_txt, c.joint_attention_dim,
                                          generator=g).to(a.device, dt),
        timestep=torch.full((n,), 0.7).to(a.device, dt),
        img_ids=torch.zeros(N, len(c.axes_dims_rope)).to(a.device),
        txt_ids=torch.zeros(a.n_txt, len(c.axes_dims_rope)).to(a.device),
        metadata=md)

    # same bf16 autocast as training (Accelerator mixed_precision="bf16"): the
    # new modules are fp32 and the backbone is bf16
    with torch.no_grad(), torch.autocast(a.device.split(":")[0], dtype=torch.bfloat16):
        enc_tok, enc_glob = model.encoder(md)
        base = model(**batch).float()

        print(f"{'group':8s} {'encoder d(tokens)':>18s} {'encoder d(global)':>18s} "
              f"{'model max|delta|':>18s}   verdict")
        bad = []
        for grp in GROUPS:
            t2, g2 = model.encoder(md, force_null=[grp])
            de_t = (enc_tok - t2).abs().max().float().item()
            de_g = (enc_glob - g2).abs().max().float().item()
            out = model(**{**batch, "force_null": [grp]}).float()
            dm = (base - out).abs().max().item()
            if dm == 0.0:
                verdict = "DISCONNECTED (encoder moved)" if de_t > 0 else "DISCONNECTED"
                bad.append(grp)
            else:
                verdict = "ok"
            print(f"{grp:8s} {de_t:18.6e} {de_g:18.6e} {dm:18.6e}   {verdict}")

    if bad:
        print(f"\nFAIL: {bad} produce an identical output when nulled -- "
              f"not wired into the forward pass.")
        sys.exit(1)
    print("\nPASS: all 8 groups change the output when nulled.")


if __name__ == "__main__":
    main()
