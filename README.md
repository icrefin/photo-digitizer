# Photo Digitizer

A Tauri desktop app that digitizes folders of scanned photo prints:

1. **Split** — automatically detects and extracts every individual photo from a
   scanner sheet (any count, any tilt, perspective-corrected).
2. **Straighten** — detects the correct upright orientation of each photo using
   on-device AI: YuNet face detection first, MobileNet-SSD object detection as a
   fallback. A tiny manual ⟲/⟳ control is available per photo.
3. **Enhance (optional, per photo)** —
   - *AI upscale*: **Real-ESRGAN ×4** (RRDBNet) via ONNX Runtime, tiled with
     overlap blending.
   - *AI color*: **Zhang et al. colorization** (SIGGRAPH-era Caffe model via
     OpenCV DNN) — faded prints are re-colored; already-colorful prints keep
     most of their original chroma.
   - *AI faces*: **GFPGAN v1.4** (ONNX) — faces are detected with YuNet,
     5-point aligned to a 512×512 crop, restored, warped back and
     feather-blended (~1 s per face on CPU).
4. **Confirm & save** — side-by-side slider comparison (original ↔ enhanced)
   before anything touches disk. Saving goes to a folder you pick.

Everything runs **locally** — no cloud, no uploads.

## Project layout

```
photo-restorer/
├── ui/                     frontend (plain HTML/JS/CSS, no bundler)
│   ├── index.html
│   ├── app.js
│   └── style.css
├── src-tauri/
│   ├── src/
│   │   ├── detect.rs       sheet segmentation → photo quads → perspective crop
│   │   ├── orient.rs       rotation detection (YuNet faces, MobileNet-SSD objects)
│   │   ├── enhance.rs      Real-ESRGAN tiling + colorization pipeline
│   │   ├── service.rs      single worker thread that owns all model handles
│   │   ├── cmds.rs         Tauri command surface
│   │   ├── pipeline.rs     shared orchestration
│   │   └── bin/pipeline.rs CLI test harness (no GUI)
│   ├── models/             downloaded AI models (git-ignored)
│   └── tauri.conf.json
└── scripts/download_models.sh
```

## Setup (macOS, Apple Silicon)

```bash
# system deps
brew install opencv@4 onnxruntime

# JS deps + Rust deps
npm install

# AI models (~260 MB total) -> src-tauri/models/
./scripts/download_models.sh
```

Models used:

| File | Purpose | Source |
|------|---------|--------|
| `yunet.onnx` | face detection for orientation | [opencv_zoo](https://github.com/opencv/opencv_zoo) |
| `ssd_mobilenet.caffemodel` + `.prototxt` | object detection fallback for orientation | [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD) |
| `esrgan_x4.onnx` | ×4 super-resolution | [anakhiu/realesrgan-onnx](https://huggingface.co/anakhiu/realesrgan-onnx) (Real-ESRGAN x4plus) |
| `colorizer.caffemodel` + `.prototxt` + `points_in_hull.npy` | colorization | [richzhang/colorization](https://github.com/richzhang/colorization) (release v2) |
| `gfpgan_1.4.onnx` | face restoration | [facefusion/models-3.0.0](https://huggingface.co/facefusion/models-3.0.0) |

> On a CN network, point `HOMEBREW_BOTTLE_DOMAIN` at
> `https://mirrors.tuna.tsinghua.edu.cn/homebrew-bottles` and swap
> `huggingface.co` → `hf-mirror.com` in the script (cargo already uses the
> USTC mirror via the user's global cargo config).

## Run

```bash
npm run dev      # Tauri dev window
```

Workflow in the app:

1. The folder picker defaults to `~/workspace/digitize_old_photos/Scanned` if
   present; otherwise Browse to your scan folder.
2. Click a sheet in the sidebar (auto-detects that sheet) or hit
   **Detect all sheets**. The center pane shows the original sheet with the
   detected quads outlined on the left and the extracted photos on the right.
3. Review the extracted photos; fix any rotation with ⟲/⟳.
4. Select photos, choose *AI upscale* and/or *AI color*, hit
   **Enhance selected** (progress per photo/tile; **✕ Cancel** aborts the
   remaining work at any time).
5. The compare window opens automatically: fixed left = original, right =
   enhanced. Scroll to zoom each pane, drag to pan, double-click to reset.
6. **Save selected…** writes `<stem>_p<index>.jpg/png` (JPEG q95 or PNG) to
   the folder you choose — a natural choice is `<scan folder>/processed`.

## CLI test harness

The detection/enhancement core doubles as a CLI so the pipeline can be
verified without the GUI:

```bash
cd src-tauri
cargo run --bin pipeline -- detect ../../Scanned --out /tmp/out --debug
cargo run --bin pipeline -- enhance /tmp/out/sheet_p0.png --upscale --colorize --out /tmp/out
```

`detect` writes `<stem>_p<i>.png` per extracted photo and `<stem>_debug.png`
with the detected quads drawn on the sheet.

## Notes & tuning

- Detection scales each sheet to ≤1600 px for segmentation (Canny + Otsu
  strategies merged, IoU-deduped); extraction warps the full-resolution sheet.
- Upscale inputs are capped at 640 px on the long side (256 px tiles, 32 px
  overlap, linear blending) → output is ×4, e.g. 640×480 → 2560×1920. Raise
  `MAX_UPSCALE_INPUT` in `enhance.rs` for higher quality at quadratic cost.
- Colorization runs at ≤512 px on the long side; predicted chroma is upsampled
  and blended with the original chroma using a mean-saturation weight
  (faded → mostly AI color, vivid → mostly original).
- Orientation: YuNet landmarks pose-check every detected face (eyes above
  nose above mouth ⇒ ×3 weight), so misdetections on rotated variants can't
  win. The object detector (SSD) fallback only decides when its total score
  is ≥0.9 *and* 1.3× the runner-up; otherwise the photo keeps 0°.
- Extraction: after the perspective warp, `trim_white_borders` crops
  near-uniform white scan margins (≥98% bright rows/cols, capped at 30% per
  side, min 120px kept).
- The `ml-service` worker thread owns all model handles (OpenCV `Net` is not
  `Send`); the Tauri app talks to it over a channel, so heavy inference never
  blocks the UI thread.
