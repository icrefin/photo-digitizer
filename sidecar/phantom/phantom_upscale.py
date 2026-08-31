"""Phantom (PASD) upscaling worker for photo-digitizer.

Runs the dreamoving/Phantom PASD pipeline (diffusion super-resolution)
plus Phantom's VQA-based automatic parameter sizing, adapted from
gradio2.py. BLIP captioning / CLIP scene routing / RealESRGAN / SwinIR /
MARCONet are dropped: the app routes old photos (people, low MOS) to
PASD anyway.

Protocol (one JSON object per line):
  stdin  request:  {"id": "...", "input": "/a.png", "output": "/b.png",
                    "scale": 4, "seed": 0}
  stdout events:   {"event": "ready", "device": "mps", "dtype": "float16"}
                   {"event": "progress", "id": "...", "pct": 0.42, "msg": "..."}
                   {"event": "done", "id": "...", "output": "...",
                    "width": W, "height": H, "sec": 12.3}
                   {"event": "error", "id": "...", "msg": "..."}
                   {"event": "fatal", "msg": "..."}   (then exit 1)
Diagnostics go to stderr. EOF on stdin exits cleanly.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# UniPC's tiny linalg.solve and a few other ops have no MPS kernel; the
# CPU fallback only fires for those (negligible cost).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "phantom_src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer  # noqa: E402
from diffusers import AutoencoderKL, UniPCMultistepScheduler  # noqa: E402

from models.pasd.controlnet import ControlNetModel  # noqa: E402
from models.pasd.unet_2d_condition import UNet2DConditionModel  # noqa: E402
from pipelines.pipeline_pasd import StableDiffusionControlNetPipeline  # noqa: E402
from myutils.wavelet_color_fix import wavelet_color_fix  # noqa: E402
from synthesis_vqa.vqa_model import VideoQualitySynthesisAnalysis  # noqa: E402
from diffusers.models.attention_processor import AttnProcessor2_0  # noqa: E402


class ChunkedAttn2_0(AttnProcessor2_0):
    """MPS variant of AttnProcessor2_0: chunked attention matmul.

    MPS has no flash-attention backend for scaled_dot_product_attention,
    so above latent resolutions of ~64×64 the full QK^T buffer (batch ×
    heads × seq × seq) exceeds Metal's buffer limit and the kernel raises
    "Invalid buffer size". Chunking over the sequence dimension computes
    the same rows (softmax is still over the full key/value rows) with
    each matmul far below the limit; fp32 intermediates mirror flash
    attention's precision.
    """

    def __init__(self, chunk_size: int = 2048):
        super().__init__()
        self.chunk_size = chunk_size

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, scale: float = 1.0):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None
            else encoder_hidden_states.shape)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        orig_dtype = query.dtype
        query = query.float()
        key = key.float()
        value = value.float()

        batch, heads, seq, dim = query.shape
        scale_v = dim ** -0.5
        out = torch.empty_like(query)
        for c0 in range(0, seq, self.chunk_size):
            c1 = min(c0 + self.chunk_size, seq)
            q = query[:, :, c0:c1]
            att = torch.matmul(q, key.transpose(-1, -2)) * scale_v
            att = att.softmax(dim=-1)
            out[:, :, c0:c1] = torch.matmul(att, value)

        hidden_states = out.to(orig_dtype).transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim)

        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

A_PROMPT = "clean, high-resolution, 8k, best quality, masterpiece"
N_PROMPT = ("dotted, noise, blur, lowres, oversmooth, longbody, bad anatomy, "
            "bad hands, missing fingers, extra digit, fewer digits, cropped, "
            "worst quality, low quality")
CFG = 7.5
STEPS = 15


def emit(**event):
    print(json.dumps(event), flush=True)


class Params:
    def __init__(self):
        self.init_latent_with_noise = False
        self.offset_noise_scale = 0.0
        self.num_inference_steps = STEPS
        self.added_noise_level = 400
        self.latent_tiled_size = 384
        self.latent_tiled_overlap = 8


class PhantomEngine:
    def __init__(self, models_dir: Path, device_name: str, dtype_name: str):
        sd = models_dir / "stable-diffusion-v1-5"
        ckpt = models_dir / "runs" / "pasd" / "checkpoint-100000"
        # vqa_cfg.json ships inside the vendored Phantom sources
        vqa_cfg = SCRIPT_DIR / "phantom_src" / "synthesis_vqa" / "vqa_cfg.json"
        mos_weights = models_dir / "synthesis_vqa" / "weights" / "mos_model_best.pth"
        for p in (sd, ckpt / "unet", ckpt / "controlnet", vqa_cfg, mos_weights):
            if not p.exists():
                raise FileNotFoundError(f"model component missing: {p}")

        if device_name == "auto":
            device_name = ("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu")
        self.device = torch.device(device_name)
        if dtype_name == "auto":
            dtype_name = "float16" if self.device.type in ("cuda", "mps") else "float32"
        self.dtype = {"float16": torch.float16, "float32": torch.float32}[dtype_name]

        scheduler = UniPCMultistepScheduler.from_pretrained(sd, subfolder="scheduler")
        text_encoder = CLIPTextModel.from_pretrained(
            sd, subfolder="text_encoder", variant="fp16",
            use_safetensors=True, torch_dtype=self.dtype)
        tokenizer = CLIPTokenizer.from_pretrained(sd, subfolder="tokenizer")
        vae = AutoencoderKL.from_pretrained(
            sd, subfolder="vae", variant="fp16",
            use_safetensors=True, torch_dtype=self.dtype)
        feature_extractor = CLIPImageProcessor.from_pretrained(sd / "feature_extractor")
        unet = UNet2DConditionModel.from_pretrained(
            ckpt, subfolder="unet", torch_dtype=self.dtype)
        controlnet = ControlNetModel.from_pretrained(
            ckpt, subfolder="controlnet", torch_dtype=self.dtype)

        for m in (vae, text_encoder, unet, controlnet):
            m.requires_grad_(False)
            m.to(self.device, dtype=self.dtype)

        self.pipeline = StableDiffusionControlNetPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            feature_extractor=feature_extractor, unet=unet,
            controlnet=controlnet, scheduler=scheduler,
            safety_checker=None, requires_safety_checker=False)
        self.pipeline._init_tiled_vae(encoder_tile_size=2048, decoder_tile_size=512)
        proc = ChunkedAttn2_0(chunk_size=2048)
        self.pipeline.unet.set_attn_processor(proc)
        self.pipeline.controlnet.set_attn_processor(proc)
        self.args = Params()

        self.vqa = VideoQualitySynthesisAnalysis(
            mos_weights=str(mos_weights), device=self.device,
            vqa_cfg_file=str(vqa_cfg))
        self._request_id = ""

    def _progress(self, step: int, _timestep, _latents):
        pct = 0.05 + 0.85 * (step + 1) / self.args.num_inference_steps
        emit(event="progress", id=self._request_id, pct=round(pct, 3),
             msg=f"denoising {step + 1}/{self.args.num_inference_steps}")

    def enhance_one(self, input_path: Path, output_path: Path, scale: int, seed: int):
        emit(event="progress", id=self._request_id, pct=0.0, msg="analyzing")
        t0 = time.time()
        image = Image.open(input_path).convert("RGB")
        ori_width, ori_height = image.size
        short_side, long_side = min(ori_width, ori_height), max(ori_width, ori_height)

        # Phantom's guard: very large inputs are downscaled first (the
        # in-app caller caps inputs to 640 px, but standalone use must be
        # robust: SD1.5 attention cost explodes past ~1080 px short side).
        if short_side > 1080 or long_side > 2160:
            tmp = min(1080 / short_side, 2160 / long_side)
            image = image.resize((round(ori_width * tmp), round(ori_height * tmp)))
            ori_width, ori_height = image.size
            short_side, long_side = min(ori_width, ori_height), max(ori_width, ori_height)

        image_info = {"task_id": 0}
        vqa_result = self.vqa.run(np.array(image), image_info, vqa_mode="mos")
        mos = float(vqa_result["vqa_mos_info"]["mos"])

        # Phantom's automatic sizing (gradio2.py PASD branch).
        if short_side < 120 or mos < 0.4:
            process_size = 384
        elif short_side <= 180:
            process_size = 512
        else:
            process_size = 768
        resize_preproc = transforms.Compose([
            transforms.Resize(process_size,
                              interpolation=transforms.InterpolationMode.BILINEAR)])

        if (short_side <= 270 or long_side <= 360) and long_side <= 540:
            rscale = min(scale, 4)
        elif (short_side <= 360 or long_side <= 480) and long_side <= 720:
            rscale = min(scale, 3)
        elif (short_side <= 540 or long_side <= 720) and long_side <= 1080:
            rscale = min(scale, 2)
        else:
            rscale = 1

        if mos < 0.4:
            input_image = image.resize(
                (int(image.size[0] * 0.5), int(image.size[1] * 0.5)))
            input_image = resize_preproc(input_image)
        else:
            input_image = image.resize(
                (image.size[0] * rscale, image.size[1] * rscale))
            if min(input_image.size) < process_size:
                input_image = resize_preproc(input_image)
        input_image = input_image.resize(
            (input_image.size[0] // 8 * 8, input_image.size[1] // 8 * 8))

        # MPS guard: SD1.5 attention cost grows with the latent area, and
        # Metal cannot allocate the full QK^T matrix (chunked attention
        # fixes the crash but not the FLOPs). Cap the work area so one
        # photo stays practical: ~0.8 MP ≈ 12k latent tokens ≈ 25 s/step.
        max_area = 800_000
        w, h = input_image.size
        if w * h > max_area:
            s = (max_area / (w * h)) ** 0.5
            input_image = input_image.resize(
                (round(w * s) // 8 * 8, round(h * s) // 8 * 8))
        width, height = input_image.size

        out_short = min(width, height)
        if out_short <= 720:
            self.args.init_latent_with_noise = False
            self.args.added_noise_level = 200
        elif out_short <= 1080:
            self.args.init_latent_with_noise = False
            self.args.added_noise_level = 400
        else:
            self.args.init_latent_with_noise = True
        alpha = 0.7 if mos < 0.4 else 1.0

        generator = torch.Generator(device=self.device.type)
        generator.manual_seed(seed)
        out = self.pipeline(
            self.args, A_PROMPT, input_image,
            num_inference_steps=self.args.num_inference_steps,
            generator=generator, height=height, width=width,
            guidance_scale=CFG, negative_prompt=N_PROMPT,
            conditioning_scale=alpha, eta=0.0,
            callback=self._progress, callback_steps=1).images[0]
        out = wavelet_color_fix(out, input_image)
        out = out.resize((ori_width * scale, ori_height * scale))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(output_path)
        emit(event="done", id=self._request_id, output=str(output_path),
             width=out.width, height=out.height,
             sec=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default=str(SCRIPT_DIR / "models"),
                    help="models directory produced by download_models.sh")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "float16", "float32"])
    opts = ap.parse_args()

    try:
        engine = PhantomEngine(Path(opts.models), opts.device, opts.dtype)
    except Exception as e:
        traceback.print_exc()
        emit(event="fatal", msg=str(e))
        sys.exit(1)
    emit(event="ready", device=str(engine.device), dtype=str(engine.dtype))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            emit(event="error", id=None, msg=f"bad request json: {e}")
            continue
        engine._request_id = str(req.get("id", ""))
        try:
            engine.enhance_one(
                Path(req["input"]), Path(req["output"]),
                int(req.get("scale", 4)), int(req.get("seed", 0)))
        except KeyError as e:
            emit(event="error", id=engine._request_id, msg=f"missing field {e}")
        except Exception as e:
            traceback.print_exc()
            emit(event="error", id=engine._request_id, msg=str(e))


if __name__ == "__main__":
    main()
