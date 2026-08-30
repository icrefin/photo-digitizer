//! AI enhancement: Real-ESRGAN ×4 super-resolution (ONNX Runtime) and
//! Zhang et al. real-time colorization (OpenCV DNN, Caffe model).

use crate::detect::mat_typed;
use crate::paths;
use anyhow::{bail, Context, Result};
use ndarray::{Array2, Array4};
use opencv::core::{Mat, MatTraitConst, Scalar, Size, Vector, CV_32F};
use opencv::dnn::{LayerTrait, NetTrait, NetTraitConst};
use opencv::{calib3d, core, dnn, imgcodecs, imgproc, objdetect};

const TILE: i32 = 256;
const OVERLAP: i32 = 32;
const STRIDE: i32 = TILE - OVERLAP;
const MAX_UPSCALE_INPUT: i32 = 640;
const MAX_COLORIZE_SIDE: i32 = 512;

pub struct Enhancers {
    esrgan: ort::session::Session,
    esrgan_input: String,
    color: Option<dnn::Net>,
    /// GFPGAN v1.4 (512×512 aligned-face restoration), optional.
    gfp: Option<ort::session::Session>,
    gfp_input: String,
    /// Separate YuNet instance for locating faces on (upscaled) photos.
    face_det: Option<core::Ptr<objdetect::FaceDetectorYN>>,
}

impl Enhancers {
    pub fn load() -> Result<Self> {
        paths::ensure_ort_dylib();
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let session = ort::session::Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .unwrap_or_else(|e| e.recover())
            .with_intra_threads(threads)
            .unwrap_or_else(|e| e.recover())
            .commit_from_file(paths::model_path("esrgan_x4.onnx")?)?;
        let esrgan_input = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "x".to_string());

        let color = match (
            paths::model_path("colorizer.prototxt"),
            paths::model_path("colorizer.caffemodel"),
            paths::model_path("points_in_hull.npy"),
        ) {
            (Ok(proto), Ok(weights), Ok(hull)) => {
                let net = match dnn::read_net_from_caffe(
                    proto.to_string_lossy().as_ref(),
                    weights.to_string_lossy().as_ref(),
                ) {
                    Ok(n) => Some(n),
                    Err(_) => None,
                };
                let hull_parsed = match load_hull_npy(&hull) {
                    Ok(h) => Some(h),
                    Err(_) => None,
                };
                let combined = match (net, hull_parsed) {
                    (Some(mut net), Some(hull)) => {
                        // The caffemodel ships without the class8_ab hull
                        // weights; the OpenCV sample injects them at runtime.
                        if let Err(e) = set_ab_weights(&mut net, &hull) {
                            eprintln!("warn: set class8_ab weights failed: {e}");
                        }
                        Some((net, hull))
                    }
                    _ => None,
                };
                combined
            }
            _ => {
                eprintln!("warn: colorizer model files unavailable, colorization disabled");
                None
            }
        };
        let color = color.map(|(net, _hull)| net);

        let gfp = match paths::model_path("gfpgan_1.4.onnx") {
            Ok(path) => match ort::session::Session::builder()?
                .with_optimization_level(
                    ort::session::builder::GraphOptimizationLevel::Level3,
                )
                .unwrap_or_else(|e| e.recover())
                .with_intra_threads(threads)
                .unwrap_or_else(|e| e.recover())
                .commit_from_file(path)
            {
                Ok(sess) => {
                    let input_name = sess
                        .inputs()
                        .first()
                        .map(|i| i.name().to_string())
                        .unwrap_or_else(|| "input".to_string());
                    Some((sess, input_name))
                }
                Err(e) => {
                    eprintln!("warn: GFPGAN failed to load: {e}");
                    None
                }
            },
            Err(_) => {
                eprintln!("warn: gfpgan_1.4.onnx unavailable, face restore disabled");
                None
            }
        };
        let (gfp, gfp_input) = match gfp {
            Some((sess, name)) => (Some(sess), name),
            None => (None, String::new()),
        };

        let face_det = match paths::model_path("yunet.onnx") {
            Ok(path) => objdetect::FaceDetectorYN::create(
                path.to_string_lossy().as_ref(),
                "",
                Size::new(320, 320),
                0.5,
                0.3,
                5000,
                0,
                0,
            )
            .ok(),
            Err(_) => None,
        };

        Ok(Self { esrgan: session, esrgan_input, color, gfp, gfp_input, face_det })
    }

    pub fn has_colorizer(&self) -> bool {
        self.color.is_some()
    }

    /// ×4 super-resolution with overlap-blended tiling. `progress` reports 0..1.
    pub fn upscale(&mut self, img: &Mat, progress: &mut dyn FnMut(f32), cancel: &dyn Fn() -> bool) -> Result<Mat> {
        let long = img.cols().max(img.rows());
        let base = if long > MAX_UPSCALE_INPUT {
            let s = MAX_UPSCALE_INPUT as f64 / long as f64;
            let mut r = Mat::default();
            imgproc::resize(
                img,
                &mut r,
                Size::new(
                    (img.cols() as f64 * s).round() as i32,
                    (img.rows() as f64 * s).round() as i32,
                ),
                0.0,
                0.0,
                imgproc::INTER_AREA,
            )?;
            r
        } else {
            img.clone()
        };
        let (bw, bh) = (base.cols() as usize, base.rows() as usize);

        let tiles_x = tile_count(bw);
        let tiles_y = tile_count(bh);
        let pad_w = (tiles_x - 1) * STRIDE as usize + TILE as usize - bw;
        let pad_h = (tiles_y - 1) * STRIDE as usize + TILE as usize - bh;

        let mut padded = Mat::default();
        core::copy_make_border(
            &base,
            &mut padded,
            0,
            pad_h as i32,
            0,
            pad_w as i32,
            core::BORDER_REPLICATE,
            Scalar::all(0.0),
        )?;
        let wp = bw + pad_w;
        let hp = bh + pad_h;
        let bytes = crate::detect::mat_bytes(&padded)?.to_vec();
        let stride_row = wp * 3;

        let mut acc = vec![0f32; wp * 4 * hp * 4 * 3];
        let mut wsum = vec![0f32; wp * 4 * hp * 4];
        let total = (tiles_x * tiles_y) as f32;
        let mut done = 0usize;

        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                if cancel() {
                    bail!("cancelled");
                }
                let (x0, y0) = (tx * STRIDE as usize, ty * STRIDE as usize);
                let mut input = vec![0f32; 3 * TILE as usize * TILE as usize];
                for c in 0..3usize {
                    for y in 0..TILE as usize {
                        let row = (y0 + y) * stride_row + x0 * 3;
                        for x in 0..TILE as usize {
                            // BGR -> RGB
                            input[c * TILE as usize * TILE as usize + y * TILE as usize + x] =
                                bytes[row + x * 3 + (2 - c)] as f32 / 255.0;
                        }
                    }
                }
                let arr = Array4::from_shape_vec((1, 3, TILE as usize, TILE as usize), input)
                    .context("tile shape")?;
                let tensor = ort::value::Tensor::from_array(arr)?;
                let outputs = self.esrgan.run(ort::inputs![self.esrgan_input.as_str() => tensor])?;
                let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
                let chw = shape.len() == 4 && shape[1] == 3;
                let ot = TILE as usize * 4;

                for y in 0..ot {
                    let wy = ramp(y, ot, ty == 0, ty == tiles_y - 1);
                    for x in 0..ot {
                        let wgt = wy * ramp(x, ot, tx == 0, tx == tiles_x - 1);
                        let gi = (y0 * 4 + y) * (wp * 4) + x0 * 4 + x;
                        for c in 0..3usize {
                            let v = if chw {
                                data[c * ot * ot + y * ot + x]
                            } else {
                                data[(y * ot + x) * 3 + c]
                            };
                            acc[gi * 3 + c] += v * wgt;
                        }
                        wsum[gi] += wgt;
                    }
                }
                done += 1;
                progress(done as f32 / total);
            }
        }

        // Assemble, crop the padding away, RGB -> BGR.
        let mut out_bytes = vec![0u8; bw * 4 * bh * 4 * 3];
        for y in 0..bh * 4 {
            for x in 0..bw * 4 {
                let gi = y * (wp * 4) + x;
                let denom = wsum[gi].max(1e-6);
                for (dst_c, src_c) in [(0usize, 2usize), (1, 1), (2, 0)] {
                    let v = acc[gi * 3 + src_c] / denom;
                    out_bytes[(y * bw * 4 + x) * 3 + dst_c] =
                        (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
        let flat = Mat::from_slice(&out_bytes)?;
        Ok(flat.reshape(3, (bh * 4) as i32)?.clone_pointee())
    }

    /// AI colorization. Faded photos are re-colored almost fully; photos that
    /// already carry strong color keep most of their original chroma.
    pub fn colorize(&mut self, img: &Mat) -> Result<Mat> {
        let net = match self.color.as_mut() {
            Some(n) => n,
            None => bail!("colorizer model not loaded"),
        };
        let (w, h) = (img.cols(), img.rows());
        let long = w.max(h) as f64;
        let s = (MAX_COLORIZE_SIDE as f64 / long).min(1.0);
        let (sw, sh) = ((w as f64 * s).round() as i32, (h as f64 * s).round() as i32);

        // Downscaled Lab.
        let mut small = Mat::default();
        imgproc::resize(img, &mut small, Size::new(sw, sh), 0.0, 0.0, imgproc::INTER_AREA)?;
        let mut small32 = Mat::default();
        small.convert_to(&mut small32, CV_32F, 1.0 / 255.0, 0.0)?;
        let mut lab = Mat::default();
        imgproc::cvt_color_def(&small32, &mut lab, imgproc::COLOR_BGR2Lab)?;
        let lab_ch = split3(&lab)?;
        let l_small = lab_ch.0;

        // Network input: L - 50, shape (1, 1, H, W).
        let mut l_in = Mat::default();
        core::subtract(&l_small, &Scalar::all(50.0), &mut l_in, &core::no_array(), -1)?;
        let blob = dnn::blob_from_image(&l_in, 1.0, Size::default(), Scalar::all(0.0), false, false, CV_32F)?;
        net.set_input(&blob, "", 1.0, Scalar::all(0.0))?;
        // Default output = class8_ab: 2-channel ab at 1/4 input resolution
        // (the hull weights injected in Enhancers::load are applied in-graph).
        let ab_mat = net.forward_single("")?;
        let dims = ab_mat.mat_size(); // MatSize derefs to &[i32]
        let (ow, oh) = (dims[3] as usize, dims[2] as usize);
        let ab_data = mat_typed::<f32>(&ab_mat)?;
        let plane = oh * ow;
        anyhow::ensure!(ab_data.len() >= 2 * plane, "unexpected ab output size");
        let a_small = ab_data[..plane].to_vec();
        let b_small = ab_data[plane..2 * plane].to_vec();

        // Full-resolution Lab.
        let mut full32 = Mat::default();
        img.convert_to(&mut full32, CV_32F, 1.0 / 255.0, 0.0)?;
        let mut lab_f = Mat::default();
        imgproc::cvt_color_def(&full32, &mut lab_f, imgproc::COLOR_BGR2Lab)?;
        let (l_full, a_full, b_full) = split3(&lab_f)?;

        // Resize predicted chroma to full res.
        let a_new = resize_f32(&a_small, ow as i32, oh as i32, w, h)?;
        let b_new = resize_f32(&b_small, ow as i32, oh as i32, w, h)?;

        // Blend with original chroma based on how colorful the original is.
        let wn = saturation_weight(img)?;
        let keep = 1.0 - wn;
        let mut a_mix = Mat::default();
        core::add_weighted(&a_new, wn as f64, &a_full, keep as f64, 0.0, &mut a_mix, -1)?;
        let mut b_mix = Mat::default();
        core::add_weighted(&b_new, wn as f64, &b_full, keep as f64, 0.0, &mut b_mix, -1)?;

        let mut merged = Vector::<Mat>::new();
        merged.push(l_full);
        merged.push(a_mix);
        merged.push(b_mix);
        let mut lab_out = Mat::default();
        core::merge(&merged, &mut lab_out)?;
        let mut bgr01 = Mat::default();
        imgproc::cvt_color_def(&lab_out, &mut bgr01, imgproc::COLOR_Lab2BGR)?;
        let mut out = Mat::default();
        bgr01.convert_to(&mut out, core::CV_8U, 255.0, 0.0)?;
        Ok(out)
    }
}

/// Split a 3-channel Mat into its channels (kept alive together in the tuple).
fn split3(m: &Mat) -> Result<(Mat, Mat, Mat)> {
    let mut ch = Vector::<Mat>::new();
    core::split(m, &mut ch)?;
    let c0 = ch.get(0)?;
    let c1 = ch.get(1)?;
    let c2 = ch.get(2)?;
    Ok((c0, c1, c2))
}

fn tile_count(size: usize) -> usize {
    if size <= TILE as usize {
        1
    } else {
        ((size - TILE as usize) as f32 / STRIDE as f32).ceil() as usize + 1
    }
}

fn ramp(i: usize, len: usize, first_edge: bool, last_edge: bool) -> f32 {
    let o = OVERLAP as usize;
    if first_edge && i < o {
        return 1.0;
    }
    if last_edge && i + o > len {
        return 1.0;
    }
    if i < o {
        (i + 1) as f32 / o as f32
    } else if i + o > len {
        (len - i) as f32 / o as f32
    } else {
        1.0
    }
}

fn resize_f32(data: &[f32], sw: i32, sh: i32, w: i32, h: i32) -> Result<Mat> {
    let flat = Mat::from_slice(data)?;
    let m = flat.reshape(1, sh)?;
    debug_assert_eq!(m.cols(), sw);
    let mut out = Mat::default();
    imgproc::resize(&m, &mut out, Size::new(w, h), 0.0, 0.0, imgproc::INTER_LINEAR)?;
    Ok(out)
}

/// Mean HSV saturation (0..1) → weight for AI chroma (faded → high).
fn saturation_weight(img: &Mat) -> Result<f32> {
    let mut hsv = Mat::default();
    imgproc::cvt_color_def(img, &mut hsv, imgproc::COLOR_BGR2HSV)?;
    let mut ch = Vector::<Mat>::new();
    core::split(&hsv, &mut ch)?;
    let s = ch.get(1)?;
    let mean = core::mean(&s, &core::no_array())?[0] / 255.0;
    Ok((1.0 - 3.0 * mean as f32).clamp(0.12, 1.0))
}

/// Inject the 313-point hull weights into the `class8_ab` layer.
fn set_ab_weights(net: &mut dnn::Net, hull: &Array2<f32>) -> Result<()> {
    struct LayerWrap<'a> {
        ptr: &'a mut core::Ptr<dnn::Layer>,
    }
    impl core::AlgorithmTraitConst for LayerWrap<'_> {
        fn as_raw_Algorithm(&self) -> *const std::ffi::c_void {
            self.ptr.inner_as_raw()
        }
    }
    impl core::AlgorithmTrait for LayerWrap<'_> {
        fn as_raw_mut_Algorithm(&mut self) -> *mut std::ffi::c_void {
            self.ptr.inner_as_raw_mut()
        }
    }
    impl dnn::LayerTraitConst for LayerWrap<'_> {
        fn as_raw_Layer(&self) -> *const std::ffi::c_void {
            self.ptr.inner_as_raw()
        }
    }
    impl dnn::LayerTrait for LayerWrap<'_> {
        fn as_raw_mut_Layer(&mut self) -> *mut std::ffi::c_void {
            self.ptr.inner_as_raw_mut()
        }
    }

    // class8_ab: the 2 x 313 x 1 x 1 hull weights
    let ab_id = net.get_layer_id("class8_ab")?;
    let mut ab_layer = net.get_layer(ab_id)?;
    let mut ab_wrap = LayerWrap { ptr: &mut ab_layer };
    let pts: Vec<f32> = hull.iter().copied().collect(); // row-major (313, 2)
    let flat = Mat::from_slice(&pts)?; // 1 x 626
    let shape2 = flat.reshape(1, 313)?; // 313 x 2
    let mut dims = Vector::<i32>::new();
    dims.push(2);
    dims.push(313);
    dims.push(1);
    dims.push(1);
    let blob = shape2.reshape_nd_vec(1, &dims)?.clone_pointee();
    let mut blobs = Vector::<Mat>::new();
    blobs.push(blob);
    ab_wrap.set_blobs(blobs);

    // conv8_313_rh: the 1 x 313 rebalance scale (2.606)
    let rh_id = net.get_layer_id("conv8_313_rh")?;
    let mut rh_layer = net.get_layer(rh_id)?;
    let mut rh_wrap = LayerWrap { ptr: &mut rh_layer };
    let gain = vec![2.606f32; 313];
    let gain_flat = Mat::from_slice(&gain)?; // 1 x 313 as 1D
    let gain_mat = gain_flat.reshape(1, 1)?.clone_pointee();
    let mut gain_blobs = Vector::<Mat>::new();
    gain_blobs.push(gain_mat);
    rh_wrap.set_blobs(gain_blobs);
    Ok(())
}

/// GFPGAN face restoration: detect faces on `img`, align each to a 512×512
/// crop, restore, warp back and feather-blend. `progress` reports (done, total).
/// Faces are processed so later faces never overlap earlier ones incorrectly
/// (all blends accumulate in float space, composited once at the end).
pub fn restore_faces(
    enh: &mut Enhancers,
    img: &Mat,
    progress: &mut dyn FnMut(usize, usize),
    cancel: &dyn Fn() -> bool,
) -> Result<Mat> {
    let (gfp, gfp_input) = match (enh.gfp.as_mut(), &enh.gfp_input) {
        (Some(g), name) => (g, name.clone()),
        (None, _) => return Ok(img.clone()),
    };
    let face_det = match enh.face_det.as_mut() {
        Some(f) => f,
        None => return Ok(img.clone()),
    };

    // Detect faces on a bounded-size copy for speed, then scale back.
    let detect_side = 960;
    let long = img.cols().max(img.rows()) as f64;
    let dscale = (detect_side as f64 / long).min(1.0);
    let mut det_img = Mat::default();
    imgproc::resize(
        img,
        &mut det_img,
        Size::new(
            (img.cols() as f64 * dscale).round() as i32,
            (img.rows() as f64 * dscale).round() as i32,
        ),
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;
    struct Face {
        // landmarks in full-image coords: [re, le, nose, rm, lm] as (x, y)
        pts: [(f32, f32); 5],
    }
    let mut faces: Vec<Face> = Vec::new();
    {
        use opencv::objdetect::FaceDetectorYNTrait;
        struct FaceDet<'a> {
            ptr: &'a mut core::Ptr<objdetect::FaceDetectorYN>,
        }
        impl objdetect::FaceDetectorYNTraitConst for FaceDet<'_> {
            fn as_raw_FaceDetectorYN(&self) -> *const std::ffi::c_void {
                self.ptr.inner_as_raw()
            }
        }
        impl objdetect::FaceDetectorYNTrait for FaceDet<'_> {
            fn as_raw_mut_FaceDetectorYN(&mut self) -> *mut std::ffi::c_void {
                self.ptr.inner_as_raw_mut()
            }
        }
        let mut det = FaceDet { ptr: face_det };
        det.set_input_size(det_img.size()?)?;
        let mut faces_mat = Mat::default();
        det.detect(&det_img, &mut faces_mat)?;
        if faces_mat.rows() > 0 {
            let data = crate::detect::mat_typed::<f32>(&faces_mat)?;
            let sx = img.cols() as f32 / det_img.cols() as f32;
            let sy = img.rows() as f32 / det_img.rows() as f32;
            for r in 0..faces_mat.rows() as usize {
                let b = r * 15;
                let conf = data[b + 14];
                if conf < 0.5 {
                    continue;
                }
                let pt = |i: usize| {
                    ((data[b + 2 * i + 4] * sx), (data[b + 2 * i + 5] * sy))
                };
                faces.push(Face {
                    pts: [pt(0), pt(1), pt(2), pt(3), pt(4)],
                });
            }
        }
    }
    if faces.is_empty() {
        return Ok(img.clone());
    }

    // ArcFace 112-px template scaled to 512.
    const S: f32 = 512.0 / 112.0;
    let tmpl: [(f32, f32); 5] = [
        (38.2946 * S, 51.6963 * S),
        (73.5318 * S, 51.5014 * S),
        (56.0252 * S, 71.7366 * S),
        (41.5493 * S, 92.3655 * S),
        (70.7299 * S, 92.2041 * S),
    ];

    // Feathered mask for blending the 512 output back: white center, soft edge.
    let center_px = (512 - 48) * (512 - 48);
    let center_data = vec![255u8; center_px];
    let center_flat = Mat::from_slice(&center_data)?;
    let center = center_flat.reshape(1, 512 - 48)?;
    let mut mask_u8 = Mat::default();
    core::copy_make_border(
        &center,
        &mut mask_u8,
        24,
        24,
        24,
        24,
        core::BORDER_CONSTANT,
        Scalar::all(0.0),
    )?;
    let mut mask = Mat::default();
    imgproc::gaussian_blur(
        &mask_u8,
        &mut mask,
        Size::new(41, 41),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let mut mask3 = Mat::default();
    imgproc::cvt_color_def(&mask, &mut mask3, imgproc::COLOR_GRAY2BGR)?;
    let mut mask32 = Mat::default();
    mask3.convert_to(&mut mask32, core::CV_32FC3, 1.0 / 255.0, 0.0)?;

    // Float copies of the working image for accumulation.
    let mut acc32 = Mat::default();
    img.convert_to(&mut acc32, core::CV_32FC3, 1.0, 0.0)?;
    let mut base32 = Mat::default();
    acc32.copy_to(&mut base32)?;

    let total = faces.len();
    for (i, face) in faces.iter().enumerate() {
        if cancel() {
            bail!("cancelled");
        }
        progress(i, total);

        let src: Vector<core::Point2f> = face
            .pts
            .iter()
            .map(|(x, y)| core::Point2f::new(*x, *y))
            .collect();
        let dst: Vector<core::Point2f> = tmpl
            .iter()
            .map(|(x, y)| core::Point2f::new(*x, *y))
            .collect();
        let mut inliers = core::Mat::default();
        let _ = &mut inliers;
        let m = match calib3d::estimate_affine_partial_2d_def(&src, &dst) {
            Ok(m) if !m.empty() => m,
            _ => {
                continue;
            }
        };

        // Warp the full image into the aligned 512×512 face crop.
        let mut crop = Mat::default();
        imgproc::warp_affine(
            img,
            &mut crop,
            &m,
            Size::new(512, 512),
            imgproc::INTER_LINEAR,
            core::BORDER_REFLECT_101,
            Scalar::all(0.0),
        )?;

        // BGR 8U crop -> RGB, NCHW, normalized to [-1, 1] (facefusion recipe)
        let bytes = crate::detect::mat_bytes(&crop)?.to_vec();
        let mut input = vec![0f32; 3 * 512 * 512];
        for c in 0..3usize {
            for y in 0..512usize {
                let row = y * 512 * 3;
                for x in 0..512usize {
                    let v = bytes[row + x * 3 + (2 - c)] as f32 / 255.0;
                    input[c * 512 * 512 + y * 512 + x] = (v - 0.5) / 0.5;
                }
            }
        }
        let arr = ndarray::Array4::from_shape_vec((1, 3, 512, 512), input)
            .context("gfp input shape")?;
        let tensor = ort::value::Tensor::from_array(arr)?;
        let outputs = gfp.run(ort::inputs![gfp_input.as_str() => tensor])?;
        let (oshape, odata) = outputs[0].try_extract_tensor::<f32>()?;

        // Normalize output to 0..1 (handle [-1,1] exports).
        let n = 512usize * 512usize * 3usize;
        let chw = oshape.len() == 4 && oshape[1] == 3;
        let (lo, _) = odata.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        let gain = if lo < -0.3 { 0.5 } else { 1.0 };
        let bias = if lo < -0.3 { 0.5 } else { 0.0 };

        let mut out_bytes = vec![0u8; n];
        for y in 0..512usize {
            for x in 0..512usize {
                // model output is RGB; keep channel order and convert after
                for dst_c in 0..3usize {
                    let v = if chw {
                        odata[dst_c * n / 3 + y * 512 + x]
                    } else {
                        odata[(y * 512 + x) * 3 + dst_c]
                    };
                    out_bytes[(y * 512 + x) * 3 + dst_c] =
                        ((v * gain + bias).clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
        let flat = Mat::from_slice(&out_bytes)?;
        let mut restored = flat.reshape(3, 512)?.clone_pointee();
        let mut restored_bgr = Mat::default();
        imgproc::cvt_color_def(&restored, &mut restored_bgr, imgproc::COLOR_RGB2BGR)?;
        restored = restored_bgr;

        // Warp restored + mask back to full size and blend.
        let m_inv = invert_affine(&m)?;
        let mut back = Mat::default();
        imgproc::warp_affine(
            &restored,
            &mut back,
            &m_inv,
            Size::new(img.cols(), img.rows()),
            imgproc::INTER_LINEAR,
            core::BORDER_CONSTANT,
            Scalar::all(0.0),
        )?;
        let mut back_mask = Mat::default();
        imgproc::warp_affine(
            &mask32,
            &mut back_mask,
            &m_inv,
            Size::new(img.cols(), img.rows()),
            imgproc::INTER_LINEAR,
            core::BORDER_CONSTANT,
            Scalar::all(0.0),
        )?;

        let mut back32 = Mat::default();
        back.convert_to(&mut back32, core::CV_32FC3, 1.0, 0.0)?;
        let mut inv_mask32 = Mat::default();
        core::subtract(
            &Scalar::all(1.0),
            &back_mask,
            &mut inv_mask32,
            &core::no_array(),
            -1,
        )?;
        // acc = acc * (1 - m) + back * m
        let mut t1 = Mat::default();
        core::multiply(&base32, &inv_mask32, &mut t1, 1.0, -1)?;
        let mut t2 = Mat::default();
        core::multiply(&back32, &back_mask, &mut t2, 1.0, -1)?;
        core::add(&t1, &t2, &mut acc32, &core::no_array(), -1)?;
    }
    progress(total, total);

    let mut out = Mat::default();
    acc32.convert_to(&mut out, core::CV_8U, 1.0, 0.0)?;
    Ok(out)
}

/// Invert a 2×3 affine matrix.
fn invert_affine(m: &Mat) -> Result<Mat> {
    let d = crate::detect::mat_typed::<f64>(m)?;
    let (a, b, tx, c, e, ty) = (d[0], d[1], d[2], d[3], d[4], d[5]);
    let det = a * e - b * c;
    anyhow::ensure!(det.abs() > 1e-12, "singular affine");
    let vals: Vec<f64> = vec![
        e / det,
        -b / det,
        (b * ty - e * tx) / det,
        -c / det,
        a / det,
        (c * tx - a * ty) / det,
    ];
    let flat = Mat::from_slice(&vals)?;
    Ok(flat.reshape(1, 2)?.clone_pointee())
}

/// Minimal .npy reader for the hull points file: (313, 2) as f64 or f32.
fn load_hull_npy(path: &std::path::Path) -> Result<Array2<f32>> {
    let bytes = std::fs::read(path)?;
    if !bytes.starts_with(b"\x93NUMPY") {
        bail!("not a numpy file: {}", path.display());
    }
    let major = bytes[6];
    let hdr_len: usize = if major == 1 {
        u16::from_le_bytes([bytes[8], bytes[9]]) as usize
    } else {
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize
    };
    let off = if major == 1 { 10 } else { 12 };
    let header = String::from_utf8_lossy(&bytes[off..off + hdr_len]);
    // header looks like: {'descr': '<i8', 'fortran_order': False, 'shape': (313, 2), }
    let parts: Vec<&str> = header.split('\'').collect();
    let descr = parts.get(3).copied().context("npy descr parse")?;
    let shape_part = header.split("'shape'").nth(1).context("npy shape key")?;
    let open = shape_part.find('(').context("npy shape '('")?;
    let close = shape_part.rfind(')').context("npy shape ')'")?;
    let dims: Vec<usize> = shape_part[open + 1..close]
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let data_off = off + hdr_len;
    let floats: Vec<f32> = match descr {
        "<f4" | "|f4" => bytes[data_off..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        "<f8" | "|f8" => bytes[data_off..]
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        "<i8" | "|i8" => bytes[data_off..]
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        other => bail!("unsupported npy dtype {other}"),
    };
    let arr = match dims.as_slice() {
        [313, 2] => Array2::from_shape_vec((313, 2), floats)?,
        [2, 313] => {
            let a = Array2::from_shape_vec((2, 313), floats)?;
            a.t().to_owned()
        }
        _ => bail!("unexpected hull shape {dims:?}"),
    };
    Ok(arr)
}

/// Save a Mat to disk (png or jpeg).
pub fn save_mat(mat: &Mat, path: &std::path::Path, jpeg_quality: i32) -> Result<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("png")
        .to_ascii_lowercase();
    let mut params = Vector::new();
    if ext == "jpg" || ext == "jpeg" {
        params.push(imgcodecs::IMWRITE_JPEG_QUALITY);
        params.push(jpeg_quality);
    }
    let ok = imgcodecs::imwrite(path.as_os_str(), mat, &params)?;
    anyhow::ensure!(ok, "imwrite failed: {}", path.display());
    Ok(())
}

/// Read a Mat from disk as BGR.
pub fn load_mat(path: &std::path::Path) -> Result<Mat> {
    let m = imgcodecs::imread(path.as_os_str(), imgcodecs::IMREAD_COLOR)?;
    anyhow::ensure!(!m.empty(), "failed to read image: {}", path.display());
    Ok(m)
}
