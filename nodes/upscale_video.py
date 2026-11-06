"""RiverField video upscale.

One ESRGAN (or any spandrel upscale model) pass over a video, in chunks, with the
original audio copied through untouched.

Why a node: a two-hour video cannot be decoded into one IMAGE batch, and ComfyUI has
no native loop. This walks the file with PyAV one frame at a time, writes a chunk
every `chunk_seconds`, concatenates the chunks with ffmpeg (stream copy, no re-encode)
and then muxes the source audio with `-c:a copy`. Chunk files are kept on disk when
`keep_chunks` is on, so a crash can be resumed by hand.
"""
from __future__ import annotations

import hashlib
import os
import shutil

from fractions import Fraction

import av
import numpy as np
import torch

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import InputImpl


def _output_dir() -> str:
    path = os.path.join(folder_paths.get_output_directory(), "riverfield", "upscale-video")
    os.makedirs(path, exist_ok=True)
    return path


class RiverFieldUpscaleVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "tooltip": "Video in ComfyUI's input folder."}),
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "spandrel model from models/upscale_models."}),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05,
                                       "tooltip": "1.0 keeps the model factor; 0.5 halves the upscaled size."}),
                "chunk_seconds": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 600.0, "step": 1.0,
                                            "tooltip": "Encode a chunk every N seconds; bounds memory and temp size."}),
                "tile": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64,
                                 "tooltip": "Tiling inside the upscale model."}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51, "step": 1}),
                "preset": (["veryfast", "fast", "medium", "slow"], {"default": "veryfast"}),
                "keep_audio": ("BOOLEAN", {"default": True}),
                "keep_chunks": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "upscale"
    CATEGORY = "RiverField"

    def upscale(self, video_path, upscale_model, scale_by, chunk_seconds, tile, crf, preset,
                keep_audio, keep_chunks):
        source = folder_paths.get_annotated_filepath((video_path or "").strip())
        container = av.open(source)
        try:
            stream = container.streams.video[0]
            rate = stream.average_rate or Fraction(24, 1)
            fps = float(rate)
            src_w, src_h = stream.width, stream.height
        finally:
            container.close()

        factor = float(getattr(upscale_model, "scale", 4.0))
        out_w = max(2, int(round(src_w * factor * scale_by)))
        out_h = max(2, int(round(src_h * factor * scale_by)))
        out_w -= out_w % 2
        out_h -= out_h % 2

        batch_frames = 4
        memory_required = (512 * 512 * 3) * 4 * max(factor, 1.0) * 384.0
        memory_required += out_w * out_h * 3 * 4 * batch_frames
        comfy.model_management.load_models_gpu([upscale_model.patcher],
                                               memory_required=memory_required,
                                               force_full_load=True)
        device = upscale_model.patcher.load_device
        out_device = comfy.model_management.intermediate_device()

        key = hashlib.sha1(f"{source}|{os.path.getmtime(source)}|{out_w}x{out_h}|{factor}".encode()).hexdigest()[:12]
        work = _output_dir()
        chunk_dir = os.path.join(work, f"chunks_{key}")
        os.makedirs(chunk_dir, exist_ok=True)

        def upscale_batch(arrays):
            images = torch.from_numpy(np.stack(arrays).copy()).float().div_(255.0)
            images = images.movedim(-1, -3).to(device)
            scaled = comfy.utils.tiled_scale(
                images, lambda a: upscale_model(a.float()),
                tile_x=tile, tile_y=tile, overlap=32,
                upscale_amount=factor, output_device=out_device,
            )
            scaled = torch.clamp(scaled.movedim(-3, -1), 0.0, 1.0)
            if scaled.shape[1] != out_h or scaled.shape[2] != out_w:
                scaled = comfy.utils.common_upscale(scaled.movedim(-1, -3), out_w, out_h, "lanczos", "disabled").movedim(-3, -1)
            return [(item.cpu().numpy() * 255.0).round().astype(np.uint8) for item in scaled]

        chunk_paths = []
        container = av.open(source)
        try:
            stream = container.streams.video[0]
            total = int(stream.frames or 0)
            bar = comfy.utils.ProgressBar(total) if total > 0 else None
            frames_per_chunk = max(1, int(round(chunk_seconds * fps)))
            encoder = None
            encoder_path = None
            written = 0
            index = 0

            def open_chunk():
                nonlocal encoder, encoder_path, written
                encoder_path = os.path.join(chunk_dir, f"chunk_{index:04d}.mp4")
                encoder = av.open(encoder_path, "w")
                out_stream = encoder.add_stream("h264", rate=rate)
                out_stream.width = out_w
                out_stream.height = out_h
                out_stream.pix_fmt = "yuv420p"
                out_stream.options = {"crf": str(crf), "preset": preset}
                written = 0
                return out_stream

            def close_chunk():
                nonlocal encoder
                if encoder is None:
                    return
                for packet in encoder.streams.video[0].encode():
                    encoder.mux(packet)
                encoder.close()
                chunk_paths.append(encoder_path)
                encoder = None

            out_stream = open_chunk()
            pending = []

            def encode_frame(array):
                nonlocal out_stream, written, index
                video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                for packet in out_stream.encode(video_frame):
                    encoder.mux(packet)
                written += 1
                if written >= frames_per_chunk:
                    close_chunk()
                    index += 1
                    out_stream = open_chunk()

            for frame in container.decode(stream):
                pending.append(frame.to_ndarray(format="rgb24"))
                if len(pending) >= batch_frames:
                    for array in upscale_batch(pending):
                        encode_frame(array)
                    if bar is not None:
                        bar.update(len(pending))
                    pending = []
            if pending:
                for array in upscale_batch(pending):
                    encode_frame(array)
                if bar is not None:
                    bar.update(len(pending))
            close_chunk()
        finally:
            container.close()

        list_path = os.path.join(chunk_dir, "chunks.txt")
        with open(list_path, "w", encoding="utf-8") as handle:
            for path in chunk_paths:
                handle.write(f"file '{path}'\n")
        joined = os.path.join(work, f"joined_{key}.mp4")
        import ffmpeg  # ffmpeg-python; the ffmpeg binary is already a requirement

        (
            ffmpeg.input(list_path, format="concat", safe=0)
            .output(joined, c="copy")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        final = os.path.join(work, f"upscaled_{key}.mp4")
        has_audio = keep_audio and any(
            s.get("codec_type") == "audio" for s in ffmpeg.probe(source).get("streams", [])
        )
        if has_audio:
            (
                ffmpeg.output(
                    ffmpeg.input(joined)["v:0"],
                    ffmpeg.input(source)["a:0"],
                    final, c="copy", shortest=None,
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        else:
            shutil.move(joined, final)

        if not keep_chunks:
            shutil.rmtree(chunk_dir, ignore_errors=True)
            if os.path.exists(joined) and joined != final:
                os.remove(joined)

        return (InputImpl.VideoFromFile(final),)


NODE_CLASS_MAPPINGS = {"RiverFieldUpscaleVideo": RiverFieldUpscaleVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"RiverFieldUpscaleVideo": "RiverField Upscale Video"}
