#!/usr/bin/env bash
# Downloads all AI models used by the app into src-tauri/models/
set -u
DIR="$(cd "$(dirname "$0")/.." && pwd)/src-tauri/models"
mkdir -p "$DIR"
cd "$DIR"

fetch () { # fetch <output> <url> [url2...]
  local out="$1"; shift
  if [ -s "$out" ]; then echo "[skip] $out exists"; return 0; fi
  for url in "$@"; do
    echo "[get ] $out  <-  $url"
    if curl -fsSL --retry 2 --connect-timeout 20 -o "$out.part" "$url"; then
      mv "$out.part" "$out"; return 0
    fi
    rm -f "$out.part"
  done
  echo "[FAIL] $out"; return 1
}

fetch yunet.onnx \
  "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
  "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

fetch ssd_mobilenet.prototxt \
  "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt" \
  "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v1.prototxt"

fetch ssd_mobilenet.caffemodel \
  "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel"

fetch colorizer.prototxt \
  "https://raw.githubusercontent.com/richzhang/colorization/caffe/models/colorization_deploy_v2.prototxt" \
  "https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_deploy_v2.prototxt"

fetch colorizer.caffemodel \
  "http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_model/colorization_release_v2.caffemodel" \
  "https://huggingface.co/datasets/pixel觅/colorization/resolve/main/colorization_release_v2.caffemodel"

fetch gfpgan_1.4.onnx \
  "https://hf-mirror.com/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx" \
  "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx"

fetch points_in_hull.npy \
  "https://raw.githubusercontent.com/richzhang/colorization/caffe/resources/points_in_hull.npy"

fetch esrgan_x4.onnx \
  "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.onnx" \
  "https://huggingface.co/qualcomm/Real-ESRGAN-x4plus/resolve/main/Real-ESRGAN-x4plus.onnx"

echo "--- results ---"
ls -la "$DIR"
