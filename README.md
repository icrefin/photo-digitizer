# Photo Digitizer

A Tauri desktop app that digitizes folders of scanned photo prints — entirely
locally, no cloud, no uploads.

> 一款将扫描的老照片文件夹数字化的桌面应用 — 完全本地运行，不上传、不联网。

---

## English

### What it does

The app turns a folder of scanner sheets into individual, straightened,
enhanced photos:

1. **Split** — automatically detects and extracts every individual photo from a
   scanner sheet (any count, any tilt, perspective-corrected). If two photos
   were wrongly detected as one, you can separate them by hand.
2. **Straighten** — detects the correct upright orientation of each photo using
   on-device AI: YuNet face detection first, MobileNet-SSD object detection as
   a fallback. A tiny manual ⟲/⟳ control is available per photo.
3. **Enhance (optional, per photo)**:
   - *AI upscale*: **Real-ESRGAN ×4** (RRDBNet) via ONNX Runtime, tiled with
     overlap blending — seconds per photo, bundled with the app.
   - *AI upscale (best)*: **Phantom AI** — [dreamoving/Phantom](https://github.com/dreamoving/Phantom)'s
     PASD diffusion super-resolution via an optional Python sidecar
     ([sidecar/phantom](sidecar/phantom/README.md)). One-click setup, ~4 min
     per photo on Apple Silicon, best results on badly degraded old photos.
   - *AI color*: **Zhang et al. colorization** — faded prints are re-colored;
     already-colorful prints keep most of their original chroma.
   - *AI faces*: **GFPGAN v1.4** (ONNX) — faces are detected, aligned, restored
     and feather-blended (~1 s per face on CPU).
4. **Confirm & save** — side-by-side slider comparison (original ↔ enhanced)
   before anything touches disk. Saving goes to a folder you pick.

Everything runs **locally**: models are bundled in the app, so the release
build works offline after installation.

**Phantom AI is optional.** When the [sidecar](sidecar/phantom/README.md)
is set up (`uv sync` + `./download_models.sh`, ~6 GB), the Enhance panel
gains a "Phantom AI (diffusion, minutes)" engine next to the bundled
Real-ESRGAN; without it, the engine row stays hidden and the default
Real-ESRGAN ×4 is used.

### Features at a glance

| Feature | Where |
|---|---|
| English / 中文 interface (persisted) | Header, top-right selector |
| Detect one sheet / all sheets | Toolbar, or click a sheet in the sidebar |
| Rotate 90° ⟲ / ⟳ | Photo card buttons |
| Adjust crop (drag corners) | ✎ on the photo card |
| **Manual split** — right-click the sheet, drag a box around one photo | Sheet view |
| AI upscale / color / face restore | Enhance panel |
| Phantom AI diffusion upscale (optional sidecar) | Enhance panel engine radio |
| Compare original ↔ enhanced (scroll zoom, drag pan) | Click a photo |
| Reset to original | Compare window |
| About dialog — version shown as the build timestamp | Footer (🔗 About) |
| Save as JPEG/PNG to any folder | Save panel |

### Requirements

- macOS on Apple Silicon (the release build is `arm64`)
- For building from source: `brew install opencv@4 onnxruntime`, Node.js, a
  Rust toolchain

### Build from source

```bash
# JS deps + Rust deps
npm install

# AI models (~260 MB total) -> src-tauri/models/
./scripts/download_models.sh

npm run dev      # development window (hot rebuilds)
npm run build    # release: binary + .app + .dmg
```

Release artifacts:

| Artifact | Path |
|---|---|
| App binary | `src-tauri/target/release/photo-digitizer` |
| macOS app | `src-tauri/target/release/bundle/macos/Photo Digitizer.app` |
| DMG installer | `src-tauri/target/release/bundle/dmg/Photo Digitizer_0.1.0_aarch64.dmg` |

The DMG is ~500 MB because the AI models are bundled inside the app, so it
works right after installation with no downloads.

> **Signing note**: the bundle carries an ad-hoc signature only. It runs
> normally on the machine that built it. If you distribute the DMG, Gatekeeper
> will show "unidentified developer" on other Macs. On first launch either:
> - right-click the app → **Open** (once per app), or
> - remove the quarantine flag with `xattr`:
>
>   ```bash
>   xattr -dr com.apple.quarantine "/Applications/Photo Digitizer.app"
>   ```
>
>   (`xattr` works on a Tauri app exactly like on any other .app bundle — it
>   removes the quarantine attribute from the bundle and everything inside it;
>   nothing Tauri-specific interferes.)
>   中文：也可以运行上面的 xattr 命令清除隔离属性，绕过 Gatekeeper 的
>   “未受信任的开发者”提示。
>
> For distribution, sign + notarize the app with an Apple Developer ID
> (e.g. `tauri sign`) before releasing.

### How to use the app

**1. Choose the language.** English or 中文 — top-right of the header. Your
choice is remembered; the first launch follows the system language.

**2. Load your scan folder.** Browse… (or type the path and press Enter).
The sidebar lists every image in the folder (jpg/jpeg/png/tif/tiff/bmp).

**3. Detect photos.** Click a sheet in the sidebar to detect just that sheet,
or **Detect all sheets** for the whole folder with per-sheet progress. The
left pane shows the sheet with detected quads drawn on it; the right pane
shows the extracted photos, each straightened and white-trimmed.

**4. Review the photos.** Each card shows a badge with the orientation method
(`face` / `object`), a `✎ crop` badge for manually cropped photos, and an
`AI ✦` badge for enhanced ones. Use **Select all** or the corner checkboxes to
choose photos.

**5. Fix rotation.** ⟲ / ⟳ rotate the photo 90°; the card badge updates.

**6. Adjust a crop.** Click **✎** on the card: the sheet shows the photo's
quad with draggable corner handles. Drag them, then **Apply crop** (or Cancel).
The sheet now draws your box in solid green, matching the cropped photo.
Manually cropped photos keep their crop when sheets are re-detected.

**7. Split a wrongly-merged photo.** If auto-detection combined two photos
into one, **right-click the sheet** and drag a box around one of them, adjust
the corners, and **Apply crop** — it is extracted as an additional photo that
appears on the right, marked `✎ crop`.

**8. Enhance.** Check any combination of *AI upscale*, *AI color*, *AI faces*
and hit **Enhance selected**. The progress bar reports each stage (loading
models, upscaling, colorizing, restoring faces); **✕ Cancel** aborts the
remaining work. The compare window opens automatically when done.

**9. Compare & reset.** Click any photo to open original ↔ enhanced side by
side. Scroll to zoom, drag to pan, double-click to reset. Not satisfied?
**Reset to original** discards the enhancement — the photo goes back to its
original state and can be enhanced again with different options.

**10. Save.** Choose JPEG (small) or PNG (lossless), press **Save selected…**
and pick the destination folder (the dialog's confirm button reads **Save**).
Files are written as `<stem>_p<index>.jpg/png`, e.g.
`20260829095358_001_p0.jpg`. Saved photos are `-enh` enhanced versions when an
enhancement exists, otherwise the original extraction.

### Project layout

```
photo-digitizer/
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

### CLI test harness

The detection/enhancement core doubles as a CLI so the pipeline can be
verified without the GUI:

```bash
cd src-tauri
cargo run --bin pipeline -- detect ../../Scanned --out /tmp/out --debug
cargo run --bin pipeline -- enhance /tmp/out/sheet_p0.png --upscale --colorize --out /tmp/out
```

`detect` writes `<stem>_p<i>.png` per extracted photo and `<stem>_debug.png`
with the detected quads drawn on the sheet.

### Notes & tuning

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

### Models

| File | Purpose | Source |
|------|---------|--------|
| `yunet.onnx` | face detection for orientation | [opencv_zoo](https://github.com/opencv/opencv_zoo) |
| `ssd_mobilenet.caffemodel` + `.prototxt` | object detection fallback for orientation | [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD) |
| `esrgan_x4.onnx` | ×4 super-resolution | [anakhiu/realesrgan-onnx](https://huggingface.co/anakhiu/realesrgan-onnx) (Real-ESRGAN x4plus) |
| `colorizer.caffemodel` + `.prototxt` + `points_in_hull.npy` | colorization | [richzhang/colorization](https://github.com/richzhang/colorization) (release v2) |
| `gfpgan_1.4.onnx` | face restoration | [facefusion/models-3.0.0](https://huggingface.co/facefusion/models-3.0.0) |

---

## 中文

### 功能介绍

将扫描的老照片文件夹变成一张张独立、摆正、可增强的照片：

1. **拆分** — 自动检测并提取扫描页上的每一张照片（数量不限、倾斜任意、
   自动透视校正）。若两张照片被误检测为一张，可以手动分离。
2. **摆正** — 使用设备端 AI 判断每张照片的正确朝向：优先 YuNet 人脸检测，
   MobileNet-SSD 物体检测作为后备。每张照片另配有 ⟲/⟳ 手动旋转按钮。
3. **增强（可选，逐张）**：
   - *AI 放大*：**Real-ESRGAN ×4**（RRDBNet，ONNX Runtime），分块重叠融合。
   - *AI 上色*：**Zhang 等 上色模型** — 褪色照片重新上色；本就有色彩的照片
     保留大部分原色彩。
   - *AI 人脸*：**GFPGAN v1.4**（ONNX）— 自动检测、对齐、修复并羽化融合人脸
     （CPU 每张约 1 秒）。
4. **确认与保存** — 保存前可左右并排对比（原图 ↔ 增强后）。保存到任意文件夹。

全部功能**本地运行**：模型打包在应用内，安装后可离线使用。

### 功能一览

| 功能 | 位置 |
|---|---|
| 英文 / 中文界面（记住选择） | 标题栏右上角选择器 |
| 检测单页 / 检测全部页 | 工具栏，或点击左侧列表中的扫描页 |
| 旋转 90° ⟲ / ⟳ | 照片卡片上的按钮 |
| 调整裁剪（拖动角点） | 照片卡片上的 ✎ |
| **手动拆分** — 右键扫描页，拖框圈选其中一张 | 扫描页视图 |
| AI 放大 / 上色 / 人脸修复 | 右侧增强面板 |
| 原图 ↔ 增强后对比（滚轮缩放、拖动平移） | 点击照片 |
| 恢复原图 | 对比窗口 |
| 保存为 JPEG/PNG 到任意文件夹 | 保存面板 |

### 系统要求

- macOS（Apple Silicon，发布版本为 `arm64`）
- 从源码构建需要：`brew install opencv@4 onnxruntime`、Node.js、Rust 工具链

### 源码构建

```bash
# 安装 JS 依赖与 Rust 依赖
npm install

# 下载 AI 模型（共约 260 MB）→ src-tauri/models/
./scripts/download_models.sh

npm run dev      # 开发窗口（热重建)
npm run build    # 发布版：二进制 + .app + .dmg
```

发布产物：

| 产物 | 路径 |
|---|---|
| 主程序 | `src-tauri/target/release/photo-digitizer` |
| macOS 应用 | `src-tauri/target/release/bundle/macos/Photo Digitizer.app` |
| DMG 安装包 | `src-tauri/target/release/bundle/dmg/Photo Digitizer_0.1.0_aarch64.dmg` |

DMG 约 500 MB：AI 模型已打包进应用，安装后立即使用，无需再下载。

> **签名说明**：当前包仅带 ad-hoc 签名。在本机构建的机器上可直接运行；若
> 分发 DMG，macOS Gatekeeper 会提示"未标识的开发者" — 首次右键 → 打开即可，
> 或使用 Apple Developer ID 签名并公证（如 `tauri sign`）后再发布。

### 使用说明

**1. 选择语言。** 标题栏右上角切换 English / 中文；选择会被记住，首次启动
跟随系统语言。

**2. 加载扫描文件夹。** 点击"浏览…"（或直接输入路径）后按"加载"。左侧列表
显示文件夹内所有图片（jpg/jpeg/png/tif/tiff/bmp）。

**3. 检测照片。** 点击左侧某个扫描页即可检测该页；"检测全部页"处理整个
文件夹并逐页显示进度。左窗格显示带检测框的扫描页，右窗格显示提取出的照片
（已摆正、修边）。

**4. 检查照片。** 每张卡片带一个方向徽标（人脸 / 物体），手动裁剪过的照片
带"✎ crop"徽标，增强过的带"AI ✦"徽标。可用"全选"或卡片角上的复选框选择。

**5. 修正旋转。** ⟲ / ⟳ 旋转 90°，卡片徽标同步更新。

**6. 调整裁剪。** 点击卡片上的 **✎**：扫描页上显示该照片的四角手柄，
拖动角点后点"应用裁剪"（或取消）。完成后扫描页上以实心绿色框显示你的
裁剪区域，与右侧照片一致。手动裁剪过的照片在重新检测时会保留该裁剪。

**7. 拆分误检照片。** 若自动检测把两张照片合为一张：在扫描页上**右键**，
拖框圈选其中一张，调整角点后"应用裁剪" — 它会被提取为一张新照片出现在
右侧，并标记"✎ crop"。

**8. 增强。** 勾选 *AI 放大*、*AI 上色*、*AI 人脸* 的组合，按"增强选中项"。
进度条分阶段显示（加载模型、放大、上色、修复人脸）；"✕ 取消"可中止剩余
任务。完成后自动打开对比窗口。

**9. 对比与恢复。** 点击任意照片打开原图 ↔ 增强后并排对比；滚轮缩放、
拖动平移、双击复位。不满意？"恢复原图"丢弃增强结果 — 可随时换选项重新增强。

**10. 保存。** 选择 JPEG（小）或 PNG（无损），按"保存选中…"后选择目标
文件夹（对话框确认按钮显示"保存"）。文件命名为 `<扫描页名>_p<序号>.jpg/png`，
例如 `20260829095358_001_p0.jpg`；已增强的照片保存增强版本，否则保存原始
提取结果。

### 项目结构

```
photo-digitizer/
├── ui/                     前端（纯 HTML/JS/CSS，无打包器）
│   ├── index.html
│   ├── app.js
│   └── style.css
├── src-tauri/
│   ├── src/
│   │   ├── detect.rs       扫描页分割 → 照片四边形 → 透视裁剪
│   │   ├── orient.rs       旋转检测（YuNet 人脸、MobileNet-SSD 物体）
│   │   ├── enhance.rs      Real-ESRGAN 分块 + 上色流水线
│   │   ├── service.rs      持有全部模型句柄的单一工作线程
│   │   ├── cmds.rs         Tauri 命令层
│   │   ├── pipeline.rs     共享编排
│   │   └── bin/pipeline.rs CLI 测试工具（无 GUI）
│   ├── models/             下载的 AI 模型（git 忽略）
│   └── tauri.conf.json
└── scripts/download_models.sh
```

### CLI 测试工具

检测/增强核心同时提供命令行，便于脱离 GUI 验证流水线：

```bash
cd src-tauri
cargo run --bin pipeline -- detect ../../Scanned --out /tmp/out --debug
cargo run --bin pipeline -- enhance /tmp/out/sheet_p0.png --upscale --colorize --out /tmp/out
```

`detect` 输出每张照片 `<stem>_p<i>.png` 与带检测框的整页
`<stem>_debug.png`。

### 技术说明与调优

- 检测将每页缩放到 ≤1600 像素做分割（Canny + Otsu 策略合并、IoU 去重）；
  提取基于原分辨率整页做透视变换。
- 放大输入长边上限 640 像素（256 像素块、32 像素重叠、线性融合）→ 输出 ×4，
  例如 640×480 → 2560×1920。如需更高质量可在 `enhance.rs` 中调高
  `MAX_UPSCALE_INPUT`（成本按平方增长）。
- 上色在长边 ≤512 像素下运行；预测的色度上采样后按平均饱和度权重与原色度
  融合（褪色 → 几乎全用 AI 颜色，鲜艳 → 大部分保留原色）。
- 朝向：YuNet 关键点姿态检查每张人脸（眼在鼻上方、鼻在口上方 ⇒ ×3 权重），
  避免旋转变体误检获胜。SSD 物体检测仅在总分 ≥0.9 且为第二名 1.3 倍以上时
  才参与决策，否则照片保持 0°。
- 提取：透视变换后 `trim_white_borders` 裁掉近纯白的扫描边（≥98% 亮度的
  行/列，每侧上限 30%，至少保留 120 像素）。
- `ml-service` 工作线程独占所有模型句柄（OpenCV `Net` 非 `Send`）；Tauri
  应用通过通道与它通信，重推理不会阻塞 UI 线程。

### 模型清单

| 文件 | 用途 | 来源 |
|------|------|------|
| `yunet.onnx` | 人脸检测（摆向） | [opencv_zoo](https://github.com/opencv/opencv_zoo) |
| `ssd_mobilenet.caffemodel` + `.prototxt` | 物体检测（摆向后备） | [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD) |
| `esrgan_x4.onnx` | ×4 超分辨率 | [anakhiu/realesrgan-onnx](https://huggingface.co/anakhiu/realesrgan-onnx)（Real-ESRGAN x4plus） |
| `colorizer.caffemodel` + `.prototxt` + `points_in_hull.npy` | 上色 | [richzhang/colorization](https://github.com/richzhang/colorization)（v2） |
| `gfpgan_1.4.onnx` | 人脸修复 | [facefusion/models-3.0.0](https://huggingface.co/facefusion/models-3.0.0) |

---

*All processing is local. Your photos never leave your machine.*
*所有处理均在本地完成，照片不会离开你的电脑。*
