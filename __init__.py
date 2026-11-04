"""RiverField ComfyUI nodes.

Three small, self-contained nodes used by the RiverField studio:

- ``RiverFieldH3Planner`` builds the ``timeline_data`` document of the MiniMax H3
  Material Planner (what its own node expects as a JSON widget).
- ``RiverFieldLTXPlanner`` builds the ``timeline_data``, ``local_prompts`` and
  ``segment_lengths`` of the LTX 2.5 Director, and turns a reference sheet into the
  still guide clip the IC-LoRA motion track reads.
- ``RiverFieldGemma4Attn`` is a startup fix, not a node: on ROCm 10.1 / gfx1151 the
  SDPA kernel traps on the Gemma 4 text encoder of LTX-2.5, so this pins that encoder
  to split attention while every diffusion model keeps the fast SDPA path.

None of them modify another pack's files.
"""
from .nodes import gemma4_attn, h3_planner, ltx_planner

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _module in (gemma4_attn, h3_planner, ltx_planner):
    NODE_CLASS_MAPPINGS.update(getattr(_module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_module, "NODE_DISPLAY_NAME_MAPPINGS", {}))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
