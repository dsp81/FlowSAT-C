"""
flux2_io.py — VAE and text-encoder plumbing for FlowSAT-C.

This module is the layer the trainer was missing: the dataloader emits
pixel_values and caption strings, the transformer consumes packed latents,
prompt embeddings and 4-D RoPE ids. Nothing bridged them.

Every transform here DELEGATES to the diffusers pipeline's own @staticmethods
rather than reimplementing them. FLUX.2's latent path is unusual -- 2x2
patchify, then BatchNorm normalisation using the VAE's running statistics rather
than the familiar (x - shift) * scale -- and a hand-rolled copy would drift from
the checkpoint the moment diffusers changes. Verified against diffusers 0.40:

    latents = retrieve_latents(vae.encode(x), sample_mode="argmax")   # (B,32,H/8,W/8)
    latents = _patchify_latents(latents)                              # (B,128,H/16,W/16)
    latents = (latents - vae.bn.running_mean) / sqrt(vae.bn.running_var + eps)
    ids     = _prepare_latent_ids(latents)                            # (B, H*W, 4)
    latents = _pack_latents(latents)                                  # (B, H*W, 128)

Text (Klein variants use Qwen3, not Mistral): a chat template is applied, then
hidden states from layers (9, 18, 27) are STACKED, giving 3 x 2560 = 7680 dims.
Encoding a bare prompt string without the chat template produces embeddings the
backbone was never trained on.

VERSION SKEW: written against diffusers 0.40. Ramanujan has 0.38. Run
`check_api()` before training -- if a symbol is missing the error names it
instead of surfacing as a shape mismatch 200 steps in.
"""

from __future__ import annotations

import torch

_PIPE = None


def _pipe_cls():
    """Klein pipeline class, whose staticmethods we borrow. Name has changed
    across diffusers versions, so try candidates in order."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    import diffusers
    for name in ("Flux2KleinPipeline", "Flux2Pipeline"):
        if hasattr(diffusers, name):
            _PIPE = getattr(diffusers, name)
            return _PIPE
    raise ImportError("no Flux2 pipeline class found in diffusers")


def check_api() -> dict:
    """Fail loudly at startup rather than 200 steps into a run."""
    P = _pipe_cls()
    need = ["_patchify_latents", "_pack_latents", "_prepare_latent_ids",
            "_prepare_text_ids", "_get_qwen3_prompt_embeds"]
    missing = [n for n in need if not hasattr(P, n)]
    if missing:
        raise RuntimeError(
            f"{P.__name__} lacks {missing}. diffusers version skew -- inspect "
            f"the installed pipeline_flux2_klein.py and adapt flux2_io.py.")
    return {"pipeline": P.__name__, "ok": True}


# ------------------------------------------------------------------------ VAE


class Flux2LatentEncoder:
    """Frozen VAE -> packed latents + 4-D ids."""

    def __init__(self, vae, dtype=torch.bfloat16):
        self.vae = vae.eval().requires_grad_(False)
        self.dtype = dtype
        self.P = _pipe_cls()

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor):
        """pixel_values in [-1, 1], (B,3,H,W) -> (latents (B,N,128), ids (B,N,4))"""
        from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_latents
        x = pixel_values.to(self.vae.device, self.vae.dtype)
        lat = retrieve_latents(self.vae.encode(x), sample_mode="argmax")
        lat = self.P._patchify_latents(lat)

        # BatchNorm statistics from the checkpoint, NOT shift/scale factors.
        bn_m = self.vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
        bn_s = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1)
                          + self.vae.config.batch_norm_eps).to(lat.device, lat.dtype)
        lat = (lat - bn_m) / bn_s

        ids = self.P._prepare_latent_ids(lat).to(lat.device)
        return self.P._pack_latents(lat).to(self.dtype), ids

    @torch.no_grad()
    def decode(self, packed: torch.Tensor, hw: tuple[int, int]):
        h, w = hw
        B, N, C = packed.shape
        lat = packed.permute(0, 2, 1).reshape(B, C, h, w).to(self.vae.dtype)
        bn_m = self.vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
        bn_s = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1)
                          + self.vae.config.batch_norm_eps).to(lat.device, lat.dtype)
        lat = self.P._unpatchify_latents(lat * bn_s + bn_m)
        return self.vae.decode(lat).sample


# ------------------------------------------------------------------ text


class Flux2TextEncoder:
    """
    Frozen Qwen3 -> (prompt_embeds (B,L,7680), text_ids (B,L,4)).

    Encoded on the fly: caption buckets (full / terse / null) are drawn per
    sample per epoch, so precomputing every variant would need ~8 TB.
    max_sequence_length defaults to 256 rather than the pipeline's 512 because
    the metadata-agnostic captions are ~200 tokens and the Qwen3 forward is the
    second-largest memory consumer in the step.
    """

    def __init__(self, text_encoder, tokenizer, dtype=torch.bfloat16,
                 max_sequence_length: int = 256, layers=(9, 18, 27)):
        self.te = text_encoder.eval().requires_grad_(False)
        self.tok = tokenizer
        self.dtype = dtype
        self.max_len = max_sequence_length
        self.layers = tuple(layers)
        self.P = _pipe_cls()
        self._null = None

    @torch.no_grad()
    def encode(self, prompts: list[str]):
        emb = self.P._get_qwen3_prompt_embeds(
            text_encoder=self.te, tokenizer=self.tok, prompt=list(prompts),
            dtype=self.dtype, device=self.te.device,
            max_sequence_length=self.max_len, hidden_states_layers=self.layers)
        ids = self.P._prepare_text_ids(emb).to(emb.device)
        return emb, ids

    @torch.no_grad()
    def null(self, batch_size: int):
        """Cached unconditional branch for CFG. The empty string still goes
        through the chat template, so this is the model's true null, not zeros."""
        if self._null is None:
            self._null = self.encode([""])
        e, i = self._null
        return e.expand(batch_size, -1, -1), i


# --------------------------------------------------------------------- loader


def load_flux2_io(model_id: str, device, dtype=torch.bfloat16,
                  max_sequence_length: int = 256):
    from diffusers import AutoencoderKLFlux2
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    check_api()
    vae = AutoencoderKLFlux2.from_pretrained(model_id, subfolder="vae",
                                             torch_dtype=dtype).to(device)
    te = Qwen3ForCausalLM.from_pretrained(model_id, subfolder="text_encoder",
                                          torch_dtype=dtype).to(device)
    tok = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    return (Flux2LatentEncoder(vae, dtype),
            Flux2TextEncoder(te, tok, dtype, max_sequence_length))


def prepare_batch(batch: dict, lat_enc: Flux2LatentEncoder,
                  txt_enc: Flux2TextEncoder) -> dict:
    """dataloader batch -> the keys the transformer actually consumes."""
    latents, img_ids = lat_enc.encode(batch["pixel_values"])
    text_emb, txt_ids = txt_enc.encode(batch["caption"])
    null_emb, _ = txt_enc.null(latents.shape[0])
    batch.update(latents=latents, img_ids=img_ids, text_emb=text_emb,
                 txt_ids=txt_ids, null_text_emb=null_emb)
    return batch