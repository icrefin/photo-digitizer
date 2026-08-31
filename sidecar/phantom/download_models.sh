#!/bin/bash
# Download all models needed by the Phantom PASD upscale sidecar.
#
# Layout produced (relative to this script's directory):
#   models/runs/pasd/checkpoint-100000/{unet,controlnet}   PASD weights (Aliyun OSS)
#   models/stable-diffusion-v1-5/{text_encoder,tokenizer,vae,scheduler,feature_extractor}
#   models/synthesis_vqa/weights/mos_model_best.pth        VQA model for automatic sizing
#   models/.ready                                          sentinel written on success
#
# Re-runnable: every download resumes with curl -C -.

set -euo pipefail
cd "$(dirname "$0")"
MODELS="$PWD/models"
mkdir -p "$MODELS"

if [ -f "$MODELS/.ready" ]; then
  echo "[download] models already present, nothing to do"
  exit 0
fi

fetch() { # fetch <url> <out>
  local url="$1" out="$2"
  if [ -f "$out" ] && [ -n "$3" ] && [ "$(stat -f%z "$out")" = "$3" ]; then
    echo "[download] have $(basename "$out")"
    return 0
  fi
  mkdir -p "$(dirname "$out")"
  # hf-mirror first (reachable from CN), huggingface.co as fallback
  local host_url="${url/HF_HOST/https://hf-mirror.com}"
  curl -fL --retry 3 --retry-delay 2 -C - -o "$out.part" "$host_url" || {
    rm -f "$out.part"
    host_url="${url/HF_HOST/https://huggingface.co}"
    curl -fL --retry 3 --retry-delay 2 -C - -o "$out.part" "$host_url"
  }
  mv "$out.part" "$out"
}

echo "[download] PASD checkpoint (5.4 GB, Aliyun OSS)"
fetch "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/public/phantom/checkpoints/pasd.zip" \
  "$MODELS/pasd.zip"

if [ ! -d "$MODELS/runs/pasd/checkpoint-100000" ]; then
  echo "[download] unzipping PASD checkpoint"
  ditto -x -k "$MODELS/pasd.zip" "$MODELS/runs"
  rm -f "$MODELS/pasd.zip"
fi
# The zip may carry optimizer/rng state; keep only what inference needs.
find "$MODELS/runs/pasd" -maxdepth 3 -type d \( -name optimizer -o -name rng_state* \) -exec rm -rf {} + 2>/dev/null || true

echo "[download] SD1.5 components (text_encoder/tokenizer/vae/scheduler/feature_extractor, fp16)"
SD="HF_HOST/sd-legacy/stable-diffusion-v1-5/resolve/main"
for f in \
  "text_encoder/config.json" \
  "text_encoder/model.fp16.safetensors" \
  "tokenizer/merges.txt" \
  "tokenizer/special_tokens_map.json" \
  "tokenizer/tokenizer_config.json" \
  "tokenizer/vocab.json" \
  "vae/config.json" \
  "vae/diffusion_pytorch_model.fp16.safetensors" \
  "scheduler/scheduler_config.json" \
  "feature_extractor/preprocessor_config.json"; do
  fetch "$SD/$f" "$MODELS/stable-diffusion-v1-5/$f"
done

echo "[download] Phantom VQA model (47 MB)"
fetch "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/public/phantom/checkpoints/mos_model_best.pth" \
  "$MODELS/synthesis_vqa/weights/mos_model_best.pth"

touch "$MODELS/.ready"
echo "[download] done"
