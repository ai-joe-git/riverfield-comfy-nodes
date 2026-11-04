"""RiverField H3 Material Planner — a structured front-end to the MiniMax H3 planner.

The MiniMax H3 Timeline Director plugin keeps its whole editing state in the
``timeline_data`` widget of ``MiniMaxH3TimelinePlanner``: a JSON document (version 10) holding
the media bins, the global prompt, the per-segment windows/prompts and the two-stage sampling
options. That widget is normally edited by a browser canvas.

This node builds the same document from plain, app-friendly fields, so a RiverField app can
expose every Material Planner option as a normal form input while the planner node still
receives a real ``timeline_data`` string. ``json.dumps`` does the escaping (a prompt with
quotes or newlines cannot corrupt the document), and the frame windows use the plugin's own
geometry (H3 lengths are ``5 + 17*n`` frames, seams default to the nearest valid overlap).

Set ``advanced_timeline_json`` to a non-empty string to bypass the builder and pass a raw
Material Planner document through unchanged — that is the exact escape hatch for presets or
for states the structured fields do not cover.

Frames, ids and ordering follow ``js/segment_windows.mjs`` and the plugin's ``normalizeState``
so the output is indistinguishable from a canvas-authored state.
"""

from __future__ import annotations

import json

try:  # ComfyUI is always present at runtime; keep the module importable for tests.
    import folder_paths
except Exception:  # pragma: no cover
    folder_paths = None

FPS = 24
MAX_LENGTH = 3592
MAX_IMAGES = 9
MAX_AUDIOS = 3
MAX_VIDEOS = 3
MIN_LENGTH = 5


def aligned_length(frames: float, minimum: int = MIN_LENGTH) -> int:
    """Nearest H3-valid length: 5 + 17*n frames, clamped to [minimum-aligned, 3592]."""
    frames = float(frames)
    lower = 5 + 17 * max(0, -(-(minimum - 5) // 17))  # ceil((minimum-5)/17)
    return min(MAX_LENGTH, max(lower, 5 + 17 * max(0, round((frames - 5) / 17))))


def nearest_overlap(frames: int, maximum: int) -> int:
    choices = [0, 1] + [5 + 17 * i for i in range(0, max(0, (maximum - 5) // 17) + 1)]
    choices = [n for n in choices if n <= maximum]
    return min(choices, key=lambda n: abs(n - frames)) if choices else 0


def create_windows(count: int, default_frames: int) -> list[dict]:
    """Same geometry as segment_windows.mjs createWindows([], count, frames)."""
    result: list[dict] = []
    for _ in range(max(1, count)):
        length = aligned_length(default_frames)
        overlap = nearest_overlap(39, min(length, result[-1]["endFrame"] - result[-1]["startFrame"]) - 1) if result else 0
        start = result[-1]["endFrame"] - overlap if result else 0
        result.append({"images": [], "audios": [], "prompt": "", "startFrame": start, "endFrame": start + length})
    return result


def _lines(text: str, limit: int) -> list[str]:
    items = [ln.strip() for ln in (text or "").splitlines()]
    return [x for x in items if x][:limit]


def _asset_id(prefix: str, index: int) -> str:
    return f"{prefix}-{index + 1}"


def build_timeline(
    *,
    global_prompt: str = "",
    generation_seconds: float = 5.0,
    start_seconds: float = 0.0,
    fps: int = FPS,
    segment_count: int = 1,
    segment_prompts: str = "",
    images: list[str] | None = None,
    audios: list[str] | None = None,
    audio_mode: str = "reference",
    videos: list[str] | None = None,
    video_audio: bool = True,
    two_stage: bool = False,
    upscaler: str = "",
    high_steps: int = 2,
    segment_windows: str = "",
) -> str:
    fps = int(fps) or FPS
    seconds = max(5.0 / fps, float(generation_seconds))
    start = max(0.0, float(start_seconds))
    count = max(1, int(segment_count))
    frames = aligned_length(round(seconds * fps))

    spec = None
    if segment_windows and segment_windows.strip():
        try:
            spec = json.loads(segment_windows)
        except Exception:
            spec = None

    image_files = [f for f in (images or []) if f][:MAX_IMAGES]
    audio_files = [f for f in (audios or []) if f][:MAX_AUDIOS]
    video_files = [f for f in (videos or []) if f][:MAX_VIDEOS]

    image_assets = [
        {"id": _asset_id("rf-image", i), "file": f, "name": f.replace("\\", "/").split("/")[-1]}
        for i, f in enumerate(image_files)
    ]
    audio_assets = [
        {
            "id": _asset_id("rf-audio", i),
            "file": f,
            "name": f.replace("\\", "/").split("/")[-1],
            "duration": round(seconds, 6),
            "trimStart": 0,
            "audioMode": "locked" if str(audio_mode).lower() == "locked" else "reference",
        }
        for i, f in enumerate(audio_files)
    ]
    video_assets = [
        {
            "id": _asset_id("rf-video", i),
            "file": f,
            "name": f.replace("\\", "/").split("/")[-1],
            "start": 0,
            "duration": round(seconds, 6),
            "trimStart": 0,
            "sourceDuration": round(seconds, 6),
            "hasAudio": bool(video_audio),
            "referenceMode": "guide",
            "referenceWidth": 0,
            "referenceHeight": 0,
            "proxy": "",
            "peaks": [],
        }
        for i, f in enumerate(video_files)
    ]

    image_ids = [a["id"] for a in image_assets]
    audio_ids = [a["id"] for a in audio_assets]

    spec_windows = None
    if isinstance(spec, dict) and isinstance(spec.get("windows"), list) and spec["windows"]:
        spec_windows = []
        for index, raw in enumerate(spec["windows"]):
            if not isinstance(raw, dict):
                continue
            start = max(0, int(round(float(raw.get("start", 0) or 0))))
            end = max(start + MIN_LENGTH, int(round(float(raw.get("end", start + frames) or 0))))
            spec_windows.append({
                "startFrame": start,
                "endFrame": end,
                "prompt": str(raw.get("prompt") or ""),
                "images": [image_ids[i] for i in (raw.get("images") or []) if isinstance(i, int) and 0 <= i < len(image_ids)],
                "audios": [audio_ids[i] for i in (raw.get("audios") or []) if isinstance(i, int) and 0 <= i < len(audio_ids)],
            })
    if spec_windows:
        windows = spec_windows
        count = len(windows)
        seconds = (windows[0]["endFrame"] - windows[0]["startFrame"]) / fps
    else:
        prompts = _lines(segment_prompts, 4096)
        windows = create_windows(count, frames)
        for i, window in enumerate(windows):
            window["prompt"] = prompts[i] if i < len(prompts) else ""
            window["images"] = list(image_ids)
            window["audios"] = list(audio_ids)

    # timing des clips vidéo/audio venant du panneau
    if isinstance(spec, dict):
        for item in (spec.get("videos") or []):
            if not isinstance(item, dict): continue
            i = item.get("index")
            if isinstance(i, int) and 0 <= i < len(video_assets):
                clip = video_assets[i]
                clip["start"] = max(0.0, float(item.get("start", 0) or 0))
                clip["duration"] = max(5.0 / fps, float(item.get("duration", seconds) or seconds))
                clip["trimStart"] = max(0.0, float(item.get("trimStart", 0) or 0))
                if item.get("sourceDuration"):
                    clip["sourceDuration"] = max(0.0, float(item["sourceDuration"]))
                if item.get("width"):
                    clip["referenceWidth"] = max(0, int(item["width"]))
                if item.get("height"):
                    clip["referenceHeight"] = max(0, int(item["height"]))
                if "hasAudio" in item:
                    clip["hasAudio"] = bool(item["hasAudio"])
                mode = str(item.get("referenceMode") or "guide")
                clip["referenceMode"] = mode if mode in {"guide", "edit", "boundary"} else "guide"
        for item in (spec.get("audios") or []):
            if not isinstance(item, dict): continue
            i = item.get("index")
            if isinstance(i, int) and 0 <= i < len(audio_assets):
                audio_assets[i]["trimStart"] = max(0.0, float(item.get("trimStart", 0) or 0))
                audio_assets[i]["duration"] = max(5.0 / fps, float(item.get("duration", seconds) or seconds))

    document = {
        "version": 10,
        "fps": fps,
        "globalPrompt": global_prompt or "",
        "secondPass": bool(two_stage),
        "secondPassModel": upscaler or "",
        "secondPassHighSteps": max(0, int(high_steps)),
        "selection": {"start": start, "duration": seconds},
        "videoAudioEnabled": bool(video_audio),
        "videoClips": video_assets,
        "images": image_assets,
        "audios": audio_assets,
        "segmentConfig": {
            "count": count,
            "activeIndex": 0,
            "mode": "timeline",
            "segments": windows,
        },
    }
    return json.dumps(document, ensure_ascii=False)


class RiverFieldH3Planner:
    """Build the MiniMax H3 Material Planner ``timeline_data`` document from form fields."""

    CATEGORY = "RiverField"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("timeline_data",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        upscalers = [""]
        if folder_paths is not None:
            try:
                upscalers += list(folder_paths.get_filename_list("latent_upscale_models") or [])
            except Exception:
                pass
        return {
            "required": {
                "advanced_timeline_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Non-empty: passed through unchanged instead of the fields below.",
                }),
                "global_prompt": ("STRING", {"default": "", "multiline": True}),
                "generation_seconds": ("FLOAT", {"default": 5.0, "min": 0.21, "max": 150.0, "step": 0.1}),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "fps": ("INT", {"default": FPS, "min": 1, "max": 60}),
                "segment_count": ("INT", {"default": 1, "min": 1, "max": 64}),
                "segment_prompts": ("STRING", {"default": "", "multiline": True,
                                               "tooltip": "One prompt per segment, one per line. Empty lines fall back to the global prompt."}),
                "audio_mode": (["reference", "locked"], {"default": "reference"}),
                "video_audio": ("BOOLEAN", {"default": True}),
                "two_stage": ("BOOLEAN", {"default": False}),
                "upscaler": (upscalers, {"default": ""}),
                "high_steps": ("INT", {"default": 2, "min": 0, "max": 100}),
                "segment_windows": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "JSON from the Studio Director panel: windows, per-segment prompts and media assignment."}),
            },
            "optional": {
                **{f"image{i}": ("STRING", {"default": "", "tooltip": f"Input-relative file for reference picture {i}."})
                   for i in range(1, MAX_IMAGES + 1)},
                **{f"audio{i}": ("STRING", {"default": "", "tooltip": f"Input-relative file for reference audio {i}."})
                   for i in range(1, MAX_AUDIOS + 1)},
                **{f"video{i}": ("STRING", {"default": "", "tooltip": f"Input-relative file for reference video {i}."})
                   for i in range(1, MAX_VIDEOS + 1)},
            },
        }

    def build(self, advanced_timeline_json="", global_prompt="", generation_seconds=5.0,
              start_seconds=0.0, fps=FPS, segment_count=1, segment_prompts="",
              audio_mode="reference", video_audio=True, two_stage=False, upscaler="",
              high_steps=2, segment_windows="", **media):
        if advanced_timeline_json and advanced_timeline_json.strip():
            return (advanced_timeline_json.strip(),)
        pick = lambda prefix, n: [str(media.get(f"{prefix}{i}") or "").strip() for i in range(1, n + 1)]
        return (build_timeline(
            global_prompt=global_prompt,
            generation_seconds=generation_seconds,
            start_seconds=start_seconds,
            fps=fps,
            segment_count=segment_count,
            segment_prompts=segment_prompts,
            images=pick("image", MAX_IMAGES),
            audios=pick("audio", MAX_AUDIOS),
            audio_mode=audio_mode,
            videos=pick("video", MAX_VIDEOS),
            video_audio=video_audio,
            two_stage=two_stage,
            upscaler=upscaler,
            high_steps=high_steps,
            segment_windows=segment_windows,
        ),)


NODE_CLASS_MAPPINGS = {"RiverFieldH3Planner": RiverFieldH3Planner}
NODE_DISPLAY_NAME_MAPPINGS = {"RiverFieldH3Planner": "RiverField H3 Material Planner"}
