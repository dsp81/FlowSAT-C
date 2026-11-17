"""d_lpips = LPIPS(AlexNet) between paired images (spec section 2). Lazy; GPU in practice. Uses the
`lpips` package already in the cap env (eval_metrics.Paired uses the same model)."""
from __future__ import annotations

import numpy as np


class LpipsFn:
    def __init__(self, device="cuda", batch=32):
        self.device, self.batch, self._m = device, batch, None

    def __call__(self, xs, ys):
        import torch
        if self._m is None:
            import lpips
            self._m = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(xs), self.batch):
                a = torch.as_tensor(np.stack(xs[i:i + self.batch])).permute(0, 3, 1, 2).float()
                b = torch.as_tensor(np.stack(ys[i:i + self.batch])).permute(0, 3, 1, 2).float()
                a = (a / 127.5 - 1).to(self.device)
                b = (b / 127.5 - 1).to(self.device)
                out.append(self._m(a, b).flatten().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


def make_lpips(cfg):
    lc = cfg.get("lpips", {}) or {}
    return LpipsFn(lc.get("device", "cuda")) if lc.get("enabled", False) else None
