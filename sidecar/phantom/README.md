# Phantom upscale sidecar

Runs [dreamoving/Phantom](https://github.com/dreamoving/Phantom)'s PASD
diffusion super-resolution pipeline for the photo-digitizer app's
"Phantom AI" upscale engine, adapted from `gradio2.py`:

- **PASD** (pixel-aware stable diffusion, SD1.5-based) as the SR model —
  Phantom's routing sends old photos / humans there anyway, so BLIP
  captioning, CLIP scene routing, RealESRGAN/SwinIR and the MARCONet
  text module are dropped.
- **Phantom's VQA model** (MOS score) is kept: it drives Phantom's
  automatic parameters (process size, upscale factor, conditioning
  strength, noise level) — no knobs to turn.
- MPS/CPU ports of the CUDA-only bits (see below).

## Usage

The app calls this sidecar automatically when the `models/` folder is
complete; it appears as the "Phantom AI" engine option next to
Real-ESRGAN in the Enhance panel.

Setup (one-time; needs `uv`, ~6 GB download, ~9 GB disk):

```bash
cd sidecar/phantom
uv sync                       # torch MPS + diffusers 0.21.4 stack
./download_models.sh          # PASD 5.4 GB + SD1.5 fp16 + VQA (resumable)
```

Run standalone (one request per line on stdin):

```bash
echo '{"id":"t","input":"in.png","output":"out.png","scale":4}' \
  | .venv/bin/python phantom_upscale.py
```

Worked protocol: JSON lines on stdout (`ready` / `progress` /
`done` / `error` / `fatal`), diagnostics on stderr.

Expect: **~4 minutes per photo** on Apple Silicon (15 diffusion steps,
work area capped at ~0.8 MP); the model set loads in ~6 s and stays
loaded for batch runs. The first load reads ~5.5 GB of weights.

## Ports from the CUDA repo

Vendored under `phantom_src/` (commit `main` @ 2024-01-19):

| File | Change |
| --- | --- |
| `myutils/devices.py` | replaced A1111 excerpt (imported `modules` — crashed on macOS) with an MPS-aware stub |
| `myutils/vaehook.py` | `print()` redirected to stderr (worker stdout is the JSON protocol) |
| `myutils/vaehook.py` | nothing else — `torch.cuda.is_available()` guards are already there |
| `pipelines/pipeline_pasd.py` | `torch.cuda.empty_cache()` guarded by `is_available()` |
| `synthesis_vqa/utils/mos.py` | `torch.load(map_location=…)`, `Resize(antialias=False)` (MPS lacks the AA op) |
| `synthesis_vqa/vqa_model.py` | `.to('cuda')` → `.to(self.devide)` device passed at init; frame tensor float32 (MPS has no float64) |
| `basicsr/archs/rrdbnet_arch.py` | new: minimal RRDB shim (the import in `controlnet.py` is used only for an optional RRDB ControlNet variant) |

Plus, in `phantom_upscale.py`:

- **Chunked attention** (`ChunkedAttn2_0`): MPS has no flash-attention
  backend, so `F.scaled_dot_product_attention` at latent sizes above
  ~64×64 hits "Invalid buffer size" (the full QK^T matrix, e.g. 25 GB at
  122×162 latents). Chunking over the sequence dimension keeps every
  matmul inside Metal's buffer limit with the same math.
- `PYTORCH_ENABLE_MPS_FALLBACK=1`: UniPC scheduler's tiny
  `linalg.solve` has no MPS kernel; only that op falls back to CPU.
- Work-area cap (~0.8 MP) so a photo stays around 4 minutes; the output
  is resized to the requested ×4 scale, matching Phantom's own
  `resize_flag` behavior.
