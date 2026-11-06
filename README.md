# RiverField ComfyUI nodes

Small, self-contained nodes used by the [RiverField](https://github.com/ai-joe-git/riverfield)
studio. None of them modify another pack's files; the two planners only build the JSON
documents that existing packs already expect as widgets.

## Nodes

### RiverField H3 Planner

Builds the `timeline_data` document of the MiniMax H3 **Material Planner**
(`MiniMaxH3TimelinePlanner`). Wire its `timeline_data` output into the planner node.

| Input | What it is |
| --- | --- |
| `segment_windows` | JSON written by a timeline panel: windows with `start`/`end` in 24 fps panel frames, a prompt per window, optional media assignment |
| `global_prompt` | scene, characters, style |
| `generation_seconds` | segment length when there are no windows |
| `start_seconds`, `fps` | timeline origin and frame rate |
| `segment_count`, `segment_prompts` | form-driven alternative to the panel |
| `image1..image9`, `video1..video3`, `audio1..audio3` | media slots, empty slots are ignored |
| `audio_mode`, `video_audio`, `two_stage`, `upscaler`, `high_steps`, `advanced_timeline_json` | planner options |

Output: `timeline_data` (string). Windows are snapped to H3-legal `5 + 17n` frame lengths.

### RiverField LTX Planner

Builds the three strings the LTX 2.5 Director (`LTXDirectorCS25`) expects, and turns an
optional reference sheet into the still guide clip its IC-LoRA motion track reads.

| Input | What it is |
| --- | --- |
| `segment_windows` | same panel JSON as above (24 fps panel frames) |
| `global_prompt` | scene, characters, style |
| `duration` | total length when there are no windows |
| `fps` | render frame rate (25 fps matches the LTX 2.5 templates) |
| `reference_sheet` | optional image in ComfyUI's input folder: characters, props and sets side by side |
| `reference_lora` | IC-LoRA applied when a sheet is given (Ingredients by default) |

Outputs: `timeline_data`, `local_prompts`, `segment_lengths`, `duration_frames`,
`duration_seconds`, `frame_rate`, `start_frame`, `end_frame`, `ic_lora_name`.

Lengths are converted by seconds and snapped to the LTX `8n + 1` grid. A reference sheet is
held on every frame with `ffmpeg -loop 1` (the Director's motion track only reads video
files) and the chosen LoRA name is emitted for the guide node.

### Gemma 4 attention fix (startup)

Not a node: on import, **ROCm builds only**, it pins `comfy.text_encoders.gemma4` to split
attention. On ROCm 10.1 / gfx1151 the SDPA kernel traps on the Gemma 4 encoder of LTX-2.5
(`hipErrorProfilerNotInitialized`); the global `--use-split-cross-attention` workaround makes
H3 video quadratic in memory, so the fix is scoped to the encoder. CUDA and CPU builds are
left untouched.


### RiverField Upscale Video

Chunked ESRGAN (or any spandrel) upscale for long clips. Four frames per batch, one
chunk file every `chunk_seconds`, concatenated with ffmpeg stream copy, then the
source audio is muxed with `-c:a copy`. No full-clip decode in memory: a 2 h video
stays at a flat RAM footprint. `keep_chunks` keeps the per-chunk mp4s for a manual
resume after a crash.

## Install

ComfyUI Manager: search for **RiverField**, or install `riverfield-comfy-nodes`.

Manual:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ai-joe-git/riverfield-comfy-nodes
```

Restart ComfyUI. The nodes appear under the **RiverField** category.

## Requirements

- ComfyUI (tested on 0.43.0).
- For the planners: the pack that owns the target widget — `MiniMaxH3TimelinePlanner`
  (Comfy-Org MiniMax H3) or `LTXDirectorCS25` (CGlide's LTX-2.5-Director). The planners do not
  import them; they only produce strings.
- `ffmpeg` on the PATH for the LTX reference sheet clip.

## License

MIT. See [LICENSE](LICENSE).
