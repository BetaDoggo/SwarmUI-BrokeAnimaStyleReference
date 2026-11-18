"""Broke Anima Style Reference: a training-free art style from a picture, for Anima (Cosmos Predict2 MiniTrainDIT)
and Krea 2 (SingleStreamDiT).

Each sampling step, the picture is noised to that step's sigma (fixed noise) and run through the unmodified model
with a neutral caption. After every block, each channel of the generated image's tokens is moved a share
`stats_match` of the way to the picture's per-channel mean and standard deviation at the same block (AdaIN inside
the DiT, on the residual stream). Only 2 x D numbers per block cross over, with no positions, so the picture's
palette, tonality and texture carry over but its subject doesn't.

Krea 2 is single-stream: text and image tokens share one sequence, so the statistics are taken over, and applied to,
the image tokens only (transformer_options["img_slice"]).

reference_megapixels (default 1) is the size the picture is shrunk to first, aspect kept; it is never enlarged.
About the generated image's own size works best: smaller loses coarse texture such as pixel-art blocks, larger
drifts off the style, and at 4 with a 1024x1024 image the result falls apart. Bigger is also slower.

restyle (on by default) leaves the first quarter of the steps alone (steps 0 and 1 of 8), so the image keeps the
composition it would have had without the reference.

Only the positive prompt's evaluations are matched (cond_or_uncond 0); the negative branch runs as usual. Costs one
extra model pass per matched evaluation.
"""
import logging

import torch

import comfy.patcher_extension
import comfy.utils

WRAPPER_KEY = "broke_anima_style_reference"
# the reference pass's prompt per model, the picture's default size and the noise it's noised with
CAPTION = "no humans, scenery, detailed background"
KREA2_CAPTION = "scenery with a detailed background, no people"
MEGAPIXELS = 1.0
NOISE_SEED = 0x5EED
# restyle: the first quarter of the steps run without the reference (2 of 8)
RESTYLE_SKIP = 0.25
EPS = 1e-5


def _family(model) -> str:
    dm = model.model.diffusion_model
    if hasattr(dm, "blocks") and hasattr(dm, "preprocess_text_embeds"):
        return "anima"
    if hasattr(dm, "blocks") and hasattr(dm, "txtfusion"):
        return "krea2"
    raise RuntimeError("Broke Anima Style Reference only works with Anima and Krea 2 models")


def _encode_image(model, vae, image, megapixels: float = MEGAPIXELS):
    """[B, H, W, C] -> model-space latent [1, 16, 1, h, w] of the first image, shrunk (never enlarged) to
    `megapixels`, aspect kept, sides multiples of 16."""
    pixels = image[:1].movedim(-1, 1)
    h, w = pixels.shape[-2:]
    s = min(1.0, (megapixels * 1024 * 1024 / (w * h)) ** 0.5)
    width, height = max(16, int(w * s) // 16 * 16), max(16, int(h * s) // 16 * 16)
    pixels = comfy.utils.common_upscale(pixels, width, height, "lanczos", "disabled").movedim(1, -1)
    latent = vae.encode(pixels[..., :3])
    if latent.ndim == 4:
        latent = latent.unsqueeze(2)
    return model.get_model_object("process_latent_in")(latent), (width, height)


def _step_index(sigma: float, sample_sigmas) -> int | None:
    """The step this sigma belongs to (a solver's intermediate sigma takes the step it is in)."""
    s = [float(v) for v in sample_sigmas]
    for i in range(len(s) - 1):
        if sigma <= s[i] * (1 + 1e-5) and sigma > s[i + 1]:
            return i
    return None


def _stats(x: torch.Tensor):
    """Per-channel mean and (population) variance over every token: x [B, ..., D] -> ([B, D], [B, D])."""
    xf = x.float().flatten(1, -2)
    mean = xf.mean(1)
    var = (xf * xf).mean(1) - mean * mean
    return mean, var.clamp_min(0)


def _make_wrapper(family: str, ref_latent: torch.Tensor, ref_cond, strength: float, restyle: bool):
    noise_cache = {}
    ctx_cache = {}

    def run(executor, x, timesteps, context, call_ref, call_main, options):
        """The shared part. call_ref(x_ref, t, ctx, ref_options) and call_main() run the model."""
        flags = options.get("cond_or_uncond") or [0]
        sigmas = options.get("sigmas")
        if strength <= 0 or sigmas is None or 0 not in flags:
            return call_main()
        sigma = float(sigmas.flatten()[0])
        sample_sigmas = options.get("sample_sigmas")
        if restyle and sample_sigmas is not None:
            step = _step_index(sigma, sample_sigmas)
            if step is not None and step < round(RESTYLE_SKIP * (len(sample_sigmas) - 1)):
                return call_main()
        dm = executor.class_obj
        blocks = dm.blocks

        # the picture noised to this sigma with fixed noise (flow: x = (1 - sigma) x0 + sigma e)
        ref = ref_latent.to(x.device, x.dtype)
        if x.ndim == 4 and ref.ndim == 5:
            ref = ref.squeeze(2)
        key = (tuple(ref.shape), x.device, x.dtype)
        if key not in noise_cache:
            noise_cache.clear()
            g = torch.Generator("cpu").manual_seed(NOISE_SEED)
            noise_cache[key] = torch.randn(ref.shape, generator=g, dtype=torch.float32).to(x.device, x.dtype)
        x_ref = (1.0 - sigma) * ref + sigma * noise_cache[key]
        ckey = (x.device, x.dtype)
        if ckey not in ctx_cache:
            ctx_cache.clear()
            cross, meta = ref_cond
            if family == "anima":
                ids, weights = meta.get("t5xxl_ids"), meta.get("t5xxl_weights")
                ctx_cache[ckey] = dm.preprocess_text_embeds(
                    cross.to(x.device, x.dtype), None if ids is None else ids.unsqueeze(0).to(x.device),
                    None if weights is None else weights.unsqueeze(0).unsqueeze(-1).to(x.device, x.dtype))
            else:
                ctx_cache[ckey] = cross.to(x.device, context.dtype)
        ctx = ctx_cache[ckey]

        n = x.shape[0] // len(flags)
        rows = [r for r in range(x.shape[0]) if flags[r // n] == 0]
        ref_stats = [None] * len(blocks)
        state = {"mode": "capture"}

        def hook_for(b):
            def hook(module, args, kw, out):
                # Anima's blocks carry only the image's tokens; Krea 2's carry the text's first
                a, e = kw["transformer_options"]["img_slice"] if family == "krea2" else (0, None)
                img = out[:, a:e] if family == "krea2" else out
                if state["mode"] == "capture":
                    ref_stats[b] = _stats(img[:1])
                    return None
                mr, vr = ref_stats[b]
                y = out.clone()
                part = img[rows].float()
                mt, vt = _stats(part)
                shape = (part.shape[0],) + (1,) * (part.ndim - 2) + (part.shape[-1],)
                scale = ((vr + EPS) / (vt + EPS)).sqrt().reshape(shape)
                target = (part - mt.reshape(shape)) * scale + mr.reshape(shape)
                moved = (part + strength * (target - part)).to(out.dtype)
                if family == "krea2":
                    y[rows, a:e] = moved
                else:
                    y[rows] = moved
                return y
            return hook

        handles = [blk.register_forward_hook(hook_for(b), with_kwargs=True) for b, blk in enumerate(blocks)]
        try:
            ref_options = dict(options)
            ref_options["cond_or_uncond"] = [0]
            call_ref(x_ref, timesteps[:1], ctx.expand(1, *ctx.shape[1:]), ref_options)
            state["mode"] = "apply"
            return call_main()
        finally:
            for h in handles:
                h.remove()

    def anima(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        def call_ref(xr, t, ctx, ref_options):
            ref_kwargs = dict(kwargs)
            ref_kwargs["transformer_options"] = ref_options
            executor(xr, t, ctx, fps, padding_mask, **ref_kwargs)
        return run(executor, x, timesteps, context, call_ref,
                   lambda: executor(x, timesteps, context, fps, padding_mask, **kwargs), kwargs.get("transformer_options", {}))

    def krea2(executor, x, timesteps, context, attention_mask=None, ref_latents=None, transformer_options={}, **kwargs):
        def call_ref(xr, t, ctx, ref_options):
            executor(xr, t, ctx, None, None, ref_options, **kwargs)
        return run(executor, x, timesteps, context, call_ref,
                   lambda: executor(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs),
                   transformer_options)

    return anima if family == "anima" else krea2


class BrokeAnimaStyleReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "an Anima or Krea 2 model"}),
                "clip": ("CLIP", {"tooltip": "the model's text encoder: encodes the reference pass's neutral caption"}),
                "vae": ("VAE",),
                "image": ("IMAGE", {"tooltip": "the style picture (first image of the batch)"}),
                "stats_match": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "how far each channel moves towards the picture's statistics after every block. "
                                                     "0 = off; 0.6 is a good default; near 1 the image falls apart on Anima"}),
                "restyle": ("BOOLEAN", {"default": True,
                                        "tooltip": "leave the first quarter of the steps alone, keeping the composition the "
                                                   "prompt would have had without the picture"}),
                "reference_megapixels": ("FLOAT", {"default": MEGAPIXELS, "min": 0.5, "max": 4.0, "step": 0.05,
                                                   "tooltip": "the picture is shrunk (never enlarged) to this area, aspect kept. "
                                                              "About the generated image's size works best (1 for 1024x1024): smaller "
                                                              "loses coarse texture such as pixel-art blocks, larger drifts off the "
                                                              "style and falls apart near 4. Bigger is slower"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "advanced/model"
    DESCRIPTION = ("Training-free style from a picture for Anima and Krea 2: matches each block's per-channel statistics "
                   "to the picture's at the same noise level. Takes palette, tonality and texture, not the subject.")

    def apply(self, model, clip, vae, image, stats_match, restyle, reference_megapixels=MEGAPIXELS):
        if stats_match <= 0:
            return (model,)
        family = _family(model)
        latent, (width, height) = _encode_image(model, vae, image, float(reference_megapixels))
        caption = KREA2_CAPTION if family == "krea2" else CAPTION
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(caption))[0]
        m = model.clone()
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY,
                               _make_wrapper(family, latent, cond, float(stats_match), bool(restyle)))
        logging.info(f"[Broke Anima Style Reference] {family}, picture {width}x{height}, stats_match {stats_match}, restyle {restyle}")
        return (m,)


NODE_CLASS_MAPPINGS = {"BrokeAnimaStyleReference": BrokeAnimaStyleReference}
NODE_DISPLAY_NAME_MAPPINGS = {"BrokeAnimaStyleReference": "Broke Anima Style Reference"}
