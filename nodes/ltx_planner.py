"""RiverField LTX planner.

Turns the director panel's windows into the three strings the ``LTXDirectorCS25``
node expects: ``timeline_data``, ``local_prompts`` and ``segment_lengths``.

The panel (``web/js/director.js``) stores windows in *panel frames* at 24 fps — that
is the H3 timeline grid, snapped to 5+17n frames. LTX renders in 8n+1 pixel frames,
so the planner converts by seconds and snaps the total length itself. Cuts can move
by a frame or two; the prompt order never changes.

Everything the Director needs to know about the timeline (global prompt, reference
mode, tracks) lives in the ``timeline_data`` JSON document, exactly like the H3
planner node: the app shows normal inputs, this node writes the JSON.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

import folder_paths

PANEL_FPS = 24.0
MIN_FRAMES = 9  # 8*1+1
REFERENCE_SIZE = (512, 288)
DEFAULT_REFERENCE_LORA = "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors"

TIMELINE_DEFAULTS = {
    "mainTrackEnabled": True,
    "audioTrackEnabled": True,
    "motionTrackEnabled": True,
    "overrideAudio": False,
    "inpaint_audio": True,
    "disable_prompt_relay": False,
    "reference_mode": "OFF",
    "retakeMode": False,
    "characters": [],
    "segments": [],
    "motionSegments": [],
    "audioSegments": [],
}


def snap_frames(frames: float) -> int:
    """LTX renders 8n+1 pixel frames; round to the nearest legal length."""
    n = max(1, int(round(frames)))
    return max(MIN_FRAMES, ((n - 1 + 7) // 8) * 8 + 1)


def reference_video(image_name: str, frames: int, fps: float) -> str | None:
    """Hold the reference sheet on every frame, as the IC-LoRA guide video expects.

    The Director's motion track only reads video files, so the sheet becomes a short
    clip at the stage-1 resolution. Cached by source name, mtime, length and fps.
    """
    name = (image_name or "").strip()
    if not name:
        return None
    try:
        source = folder_paths.get_annotated_filepath(name)
    except Exception:
        return None
    if not os.path.isfile(source):
        return None

    key = hashlib.sha1(
        f"{os.path.abspath(source)}|{os.path.getmtime(source)}|{frames}|{fps}".encode()
    ).hexdigest()[:12]
    out = os.path.join(folder_paths.get_input_directory(), f"riverfield_refsheet_{key}.mp4")
    if os.path.isfile(out):
        return os.path.basename(out)

    width, height = REFERENCE_SIZE
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", source,
        "-frames:v", str(max(1, int(frames))), "-r", str(fps),
        "-vf", (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"),
        "-pix_fmt", "yuv420p", "-c:v", "libx264", out,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return os.path.basename(out) if os.path.isfile(out) else None


def parse_windows(segment_windows: str, duration: float, fps: float) -> list[dict]:
    """Panel windows in 24 fps frames -> windows in seconds, sorted by start."""
    spec = None
    if segment_windows and segment_windows.strip():
        try:
            spec = json.loads(segment_windows)
        except Exception:
            spec = None

    windows: list[dict] = []
    if isinstance(spec, dict):
        for raw in spec.get("windows") or []:
            try:
                start = max(0.0, float(raw.get("start", 0)) / PANEL_FPS)
                end = max(0.0, float(raw.get("end", 0)) / PANEL_FPS)
            except (TypeError, ValueError):
                continue
            if end - start < 1.0 / max(fps, 1.0):
                continue
            windows.append({"start": start, "end": end,
                            "prompt": str(raw.get("prompt") or "").strip()})
    if not windows:
        seconds = max(1.0 / max(fps, 1.0), float(duration or 5.0))
        windows = [{"start": 0.0, "end": seconds, "prompt": ""}]
    windows.sort(key=lambda w: w["start"])
    return windows


def build_timeline(
    *,
    segment_windows: str = "",
    global_prompt: str = "",
    duration: float = 5.0,
    fps: float = 25.0,
    reference_sheet: str = "",
    reference_lora: str = DEFAULT_REFERENCE_LORA,
) -> tuple[str, str, str, int, float, float, int, int, str]:
    fps = float(fps) or 25.0
    windows = parse_windows(segment_windows, duration, fps)

    total_seconds = max(w["end"] for w in windows)
    duration_frames = snap_frames(total_seconds * fps)
    local_prompts = " | ".join(w["prompt"] for w in windows)
    segment_lengths = ",".join(
        str(max(1, int(round((w["end"] - w["start"]) * fps)))) for w in windows
    )

    # A reference sheet rides the motion track as a still clip and pulls the IC-LoRA in.
    ic_lora_name = "None"
    motion_segments = []
    guide = reference_video(reference_sheet, duration_frames, fps)
    if guide:
        ic_lora_name = (reference_lora or DEFAULT_REFERENCE_LORA).strip()
        motion_segments.append({
            "id": "riverfield-ref",
            "start": 0,
            "length": duration_frames,
            "type": "motion_video",
            "videoFile": guide,
            "trimStart": 0,
            "videoStrength": 1.0,
            "videoAttentionStrength": 0.65,
            "resampleMode": "nearest",
        })

    document = dict(TIMELINE_DEFAULTS)
    document.update({
        "global_prompt": (global_prompt or "").strip(),
        "normalStartFrame": 0,
        "normalDurationFrames": duration_frames,
        "motionSegments": motion_segments,
    })

    return (
        json.dumps(document, ensure_ascii=False),
        local_prompts,
        segment_lengths,
        duration_frames,
        round(duration_frames / fps, 6),
        fps,
        0,
        duration_frames,
        ic_lora_name,
    )


class RiverFieldLTXPlanner:
    """Build the LTX Director timeline document from the panel's segment windows."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment_windows": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Windows written by the director panel (start/end in 24 fps frames, per-window prompt).",
                }),
                "global_prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Describes the scene, characters and style; anchors every shot.",
                }),
                "duration": ("FLOAT", {
                    "default": 5.0, "min": 0.5, "max": 120.0, "step": 0.1,
                    "tooltip": "Total length in seconds when the panel has no windows (otherwise the windows win).",
                }),
                "fps": ("FLOAT", {
                    "default": 25.0, "min": 1.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Frame rate of the render. LTX 2.5 templates ship at 25 fps.",
                }),
                "reference_sheet": ("STRING", {
                    "default": "",
                    "tooltip": "Optional reference sheet (image in ComfyUI's input folder): characters, props and sets laid out side by side. Held on every frame as the IC-LoRA guide.",
                }),
                "reference_lora": ("STRING", {
                    "default": DEFAULT_REFERENCE_LORA,
                    "tooltip": "IC-LoRA applied when a reference sheet is given.",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "FLOAT", "FLOAT", "INT", "INT", "STRING")
    RETURN_NAMES = ("timeline_data", "local_prompts", "segment_lengths",
                    "duration_frames", "duration_seconds", "frame_rate",
                    "start_frame", "end_frame", "ic_lora_name")
    FUNCTION = "build"
    CATEGORY = "RiverField"

    def build(self, segment_windows: str = "", global_prompt: str = "",
              duration: float = 5.0, fps: float = 25.0,
              reference_sheet: str = "", reference_lora: str = DEFAULT_REFERENCE_LORA):
        return build_timeline(segment_windows=segment_windows,
                              global_prompt=global_prompt,
                              duration=duration, fps=fps,
                              reference_sheet=reference_sheet,
                              reference_lora=reference_lora)


NODE_CLASS_MAPPINGS = {"RiverFieldLTXPlanner": RiverFieldLTXPlanner}
NODE_DISPLAY_NAME_MAPPINGS = {"RiverFieldLTXPlanner": "RiverField LTX Planner"}
