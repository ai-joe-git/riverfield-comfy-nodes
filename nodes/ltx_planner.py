"""RiverField LTX planner.

Turns the director panel's windows and media into the three strings the
``LTXDirectorCS25`` node expects: ``timeline_data``, ``local_prompts`` and
``segment_lengths``.

The panel (``web/js/director.js``) stores windows in *panel frames* at 24 fps — that
is the H3 timeline grid, snapped to 5+17n frames. LTX renders in 8n+1 pixel frames,
so the planner converts by seconds and snaps the total length itself. Cuts can move
by a frame or two; the prompt order never changes.

Media follows the panel's three tracks:

- pictures assigned to a window become keyframes (``segments``), placed at the start
  of the shot (or spread over it when a window has several);
- video clips on the timeline become IC-LoRA motion guides (``motionSegments``);
- audio clips become the custom audio track (``audioSegments``), with the window
  chips as a fallback when nothing was dropped on the audio track.

A reference sheet is held on every frame as a still guide clip, because the Director's
motion track only reads video files. One IC-LoRA is applied for the whole graph: the
sheet and the motion clips share it.
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
MAX_IMAGES = 4
MAX_VIDEOS = 3
MAX_AUDIOS = 3

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


def parse_spec(segment_windows: str) -> dict | None:
    if not segment_windows or not segment_windows.strip():
        return None
    try:
        spec = json.loads(segment_windows)
    except Exception:
        return None
    return spec if isinstance(spec, dict) else None


def parse_windows(spec: dict | None, duration: float, fps: float) -> list[dict]:
    """Panel windows in 24 fps frames -> windows in seconds with their media indices."""
    windows: list[dict] = []
    if spec:
        for raw in spec.get("windows") or []:
            try:
                start = max(0.0, float(raw.get("start", 0)) / PANEL_FPS)
                end = max(0.0, float(raw.get("end", 0)) / PANEL_FPS)
            except (TypeError, ValueError):
                continue
            if end - start < 1.0 / max(fps, 1.0):
                continue
            windows.append({
                "start": start,
                "end": end,
                "prompt": str(raw.get("prompt") or "").strip(),
                "images": [i for i in (raw.get("images") or []) if isinstance(i, int)],
                "audios": [i for i in (raw.get("audios") or []) if isinstance(i, int)],
            })
    if not windows:
        seconds = max(1.0 / max(fps, 1.0), float(duration or 5.0))
        windows = [{"start": 0.0, "end": seconds, "prompt": "", "images": [], "audios": []}]
    windows.sort(key=lambda w: w["start"])
    return windows


def _filename(files: tuple[str, ...], index) -> str:
    """The panel's media index is the position among the numbered slots, empty or not."""
    if not isinstance(index, int) or index < 0 or index >= len(files):
        return ""
    return (files[index] or "").strip()


def _keyframes(windows: list[dict], images: tuple[str, ...], fps: float) -> list[dict]:
    segments = []
    for window in windows:
        assigned = [f for f in (_filename(images, i) for i in window["images"]) if f]
        if not assigned:
            continue
        span = max(1, int(round((window["end"] - window["start"]) * fps)))
        base = int(round(window["start"] * fps))
        for k, name in enumerate(assigned):
            offset = int(round(k * span / len(assigned))) if len(assigned) > 1 else 0
            segments.append({
                "id": f"rf-image-{len(segments) + 1}",
                "start": base + offset,
                "length": 1,
                "type": "image",
                "isAnchor": False,
                "isEndFrame": False,
                "imageFile": name,
            })
    return segments


def _motion_clips(spec: dict | None, videos: tuple[str, ...], fps: float,
                  total_seconds: float) -> list[dict]:
    segments = []
    for item in (spec or {}).get("videos") or []:
        if not isinstance(item, dict):
            continue
        name = _filename(videos, item.get("index"))
        if not name:
            continue
        start = max(0.0, float(item.get("start", 0) or 0))
        length = max(1.0 / fps, float(item.get("duration", total_seconds) or total_seconds))
        trim = max(0.0, float(item.get("trimStart", 0) or 0))
        segments.append({
            "id": f"rf-motion-{len(segments) + 1}",
            "start": int(round(start * fps)),
            "length": max(1, int(round(length * fps))),
            "type": "motion_video",
            "videoFile": name,
            "trimStart": int(round(trim * fps)),
            "videoStrength": 1.0,
            "videoAttentionStrength": 0.65,
            "resampleMode": "nearest",
        })
    return segments


def _audio_clips(spec: dict | None, windows: list[dict], audios: tuple[str, ...],
                 fps: float, total_seconds: float) -> list[dict]:
    segments = []
    for item in (spec or {}).get("audios") or []:
        if not isinstance(item, dict):
            continue
        name = _filename(audios, item.get("index"))
        if not name:
            continue
        start = max(0.0, float(item.get("start", 0) or 0))
        length = max(1.0 / fps, float(item.get("duration", total_seconds) or total_seconds))
        trim = max(0.0, float(item.get("trimStart", 0) or 0))
        segments.append({
            "id": f"rf-audio-{len(segments) + 1}",
            "start": int(round(start * fps)),
            "length": max(1, int(round(length * fps))),
            "audioFile": name,
            "trimStart": int(round(trim * fps)),
        })
    if segments:
        return segments

    # no clip on the audio track: use the window chips, one audio per shot
    for window in windows:
        name = next((f for f in (_filename(audios, i) for i in window["audios"]) if f), "")
        if not name:
            continue
        segments.append({
            "id": f"rf-audio-{len(segments) + 1}",
            "start": int(round(window["start"] * fps)),
            "length": max(1, int(round((window["end"] - window["start"]) * fps))),
            "audioFile": name,
            "trimStart": 0,
        })
    return segments


def build_timeline(
    *,
    segment_windows: str = "",
    global_prompt: str = "",
    duration: float = 5.0,
    fps: float = 25.0,
    reference_sheet: str = "",
    reference_lora: str = DEFAULT_REFERENCE_LORA,
    images: tuple[str, ...] = (),
    videos: tuple[str, ...] = (),
    audios: tuple[str, ...] = (),
    window: int = -1,
) -> tuple[str, str, str, int, float, float, int, int, str]:
    fps = float(fps) or 25.0
    spec = parse_spec(segment_windows)
    windows = parse_windows(spec, duration, fps)

    total_seconds = max(w["end"] for w in windows)
    duration_frames = snap_frames(total_seconds * fps)
    local_prompts = " | ".join(w["prompt"] for w in windows)
    segment_lengths = ",".join(
        str(max(1, int(round((w["end"] - w["start"]) * fps)))) for w in windows
    )

    keyframes = _keyframes(windows, images, fps)
    motion_segments = _motion_clips(spec, videos, fps, total_seconds)
    audio_segments = _audio_clips(spec, windows, audios, fps, total_seconds)

    # A reference sheet rides the motion track as a still clip and pulls the IC-LoRA in.
    ic_lora_name = "None"
    guide = reference_video(reference_sheet, duration_frames, fps)
    if guide:
        motion_segments.insert(0, {
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
    if guide or motion_segments:
        ic_lora_name = (reference_lora or DEFAULT_REFERENCE_LORA).strip()

    # A window index restricts the outputs to one shot while keeping the full timeline
    # document: the four-chain graph gives each chain its own window and prompt.
    start_out, length_out = 0, duration_frames
    if 0 <= int(window) < len(windows):
        shot = windows[int(window)]
        start_out = int(round(shot["start"] * fps))
        length_out = max(1, int(round((shot["end"] - shot["start"]) * fps)))
        local_prompts = shot["prompt"]
        segment_lengths = str(length_out)

    document = dict(TIMELINE_DEFAULTS)
    document.update({
        "global_prompt": (global_prompt or "").strip(),
        "normalStartFrame": 0,
        "normalDurationFrames": duration_frames,
        "segments": keyframes,
        "motionSegments": motion_segments,
        "audioSegments": audio_segments,
    })

    return (
        json.dumps(document, ensure_ascii=False),
        local_prompts,
        segment_lengths,
        length_out,
        round(length_out / fps, 6),
        fps,
        start_out,
        start_out + length_out,
        ic_lora_name,
    )


class RiverFieldLTXPlanner:
    """Build the LTX Director timeline document from the panel's windows and media."""

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
                    "tooltip": "Optional reference sheet: characters, props and sets laid out side by side. Held on every frame as the IC-LoRA guide.",
                }),
                "reference_lora": ("STRING", {
                    "default": DEFAULT_REFERENCE_LORA,
                    "tooltip": "IC-LoRA applied when a sheet or a motion clip is used.",
                }),
            },
            "optional": {
                "window": ("INT", {
                    "default": -1, "min": -1, "max": 15, "step": 1,
                    "tooltip": "Render only this window (-1: the whole timeline). The four-chain graph uses 0..3.",
                }),
                **{
                    f"image{i + 1}": ("STRING", {
                        "default": "",
                        "tooltip": "Keyframe picture, placed at the start of the shot it is assigned to.",
                    }) for i in range(MAX_IMAGES)
                },
                **{
                    f"video{i + 1}": ("STRING", {
                        "default": "",
                        "tooltip": "Motion guide clip (IC-LoRA video guidance) on the timeline.",
                    }) for i in range(MAX_VIDEOS)
                },
                **{
                    f"audio{i + 1}": ("STRING", {
                        "default": "",
                        "tooltip": "Custom audio clip on the timeline.",
                    }) for i in range(MAX_AUDIOS)
                },
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "FLOAT", "FLOAT", "INT", "INT", "STRING")
    RETURN_NAMES = ("timeline_data", "local_prompts", "segment_lengths",
                    "duration_frames", "duration_seconds", "frame_rate",
                    "start_frame", "end_frame", "ic_lora_name")
    FUNCTION = "build"
    CATEGORY = "RiverField"

    def build(self, segment_windows: str = "", global_prompt: str = "",
              duration: float = 5.0, fps: float = 25.0,
              reference_sheet: str = "", reference_lora: str = DEFAULT_REFERENCE_LORA,
              window: int = -1, **media):
        images = tuple(media.get(f"image{i + 1}", "") for i in range(MAX_IMAGES))
        videos = tuple(media.get(f"video{i + 1}", "") for i in range(MAX_VIDEOS))
        audios = tuple(media.get(f"audio{i + 1}", "") for i in range(MAX_AUDIOS))
        return build_timeline(segment_windows=segment_windows,
                              global_prompt=global_prompt,
                              duration=duration, fps=fps,
                              reference_sheet=reference_sheet,
                              reference_lora=reference_lora, window=window,
                              images=images, videos=videos, audios=audios)


NODE_CLASS_MAPPINGS = {"RiverFieldLTXPlanner": RiverFieldLTXPlanner}
NODE_DISPLAY_NAME_MAPPINGS = {"RiverFieldLTXPlanner": "RiverField LTX Planner"}
