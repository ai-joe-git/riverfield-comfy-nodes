"""Force ComfyUI's split attention on the Gemma 4 text encoder, on ROCm only.

Why: on ROCm 10.1 / gfx1151 the SDPA kernel traps on the Gemma 4 encoder of LTX-2.5
(`hipErrorProfilerNotInitialized`, a device-side assert). The workaround used to be
`--use-split-cross-attention` for the whole process, but ComfyUI's split attention builds
the full attention scores — O(tokens²) memory per head. At 8 s of H3 video that is hundreds
of GB, and the machine dies, while the DiT is perfectly happy on SDPA's flash path (AOTriton).

So: patch `comfy.text_encoders.gemma4.optimized_attention_for_device` to return the split
implementation, and run ComfyUI without the global flag. The encoder sees short prompts, the
memory is trivial there; every diffusion model keeps the fast, linear-memory path.

The patch is gated on `torch.version.hip`, so CUDA and CPU builds keep their default
attention. Loaded as a custom node so it is applied before the first prompt and survives a
ComfyUI update: nothing in ComfyUI's own files is touched.
"""
import logging

log = logging.getLogger("RiverField")

try:
    import torch

    import comfy.ldm.modules.attention as attention
    import comfy.text_encoders.gemma4 as gemma4

    if torch.version.hip:
        _split = attention.attention_split

        def _split_for_gemma4(device, **kwargs):
            # gemma4 calls optimized_attention_for_device(device, mask=..., small_input=True)
            return _split

        gemma4.optimized_attention_for_device = _split_for_gemma4
        log.info("[RiverField] gemma4 text encoder pinned to split attention (SDPA traps on ROCm)")
except Exception:  # noqa: BLE001 - never block ComfyUI startup
    log.exception("[RiverField] could not patch gemma4 attention")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
