//! Orientation detection: try 4 rotations and score each with a face detector
//! (YuNet) and, if no face is found, a general object detector (MobileNet-SSD).
//! The rotation with the most confident detections is considered upright.

use crate::detect::{mat_typed, rotate_cw};
use anyhow::Result;
use opencv::core::{Mat, MatTraitConst, Ptr, Scalar, Size, CV_32F};
use opencv::objdetect::FaceDetectorYNTrait;
use opencv::core::Vector;
use opencv::{core, dnn, dnn::NetTrait, imgcodecs, imgproc, objdetect};

pub struct Detectors {
    pub face: Ptr<objdetect::FaceDetectorYN>,
    pub obj: Option<dnn::Net>,
}

/// Adapter giving `FaceDetectorYNTrait` method access over the raw `Ptr`.
struct FaceDet<'a> {
    ptr: &'a mut Ptr<objdetect::FaceDetectorYN>,
}

impl opencv::objdetect::FaceDetectorYNTraitConst for FaceDet<'_> {
    fn as_raw_FaceDetectorYN(&self) -> *const std::ffi::c_void {
        let p = self.ptr.inner_as_raw();
        p
    }
}

impl opencv::objdetect::FaceDetectorYNTrait for FaceDet<'_> {
    fn as_raw_mut_FaceDetectorYN(&mut self) -> *mut std::ffi::c_void {
        let p = self.ptr.inner_as_raw_mut();
        p
    }
}

const FACE_CONF: f32 = 0.3;
const OBJ_CONF: f32 = 0.35;
/// Face detection runs at (near) full resolution: downscaling shrank small
/// faces below YuNet's reliable range. Only huge crops are reduced.
const FACE_MAX_SIDE: i32 = 2400;
const FACE_MIN_SIDE: i32 = 1150;
const TIE_RATIO: f32 = 0.85;
const SSD_INPUT: i32 = 300;
/// A face whose landmarks form an upright pose (eyes above nose above mouth)
/// counts this much more than an unverified detection.
const POSE_GAIN: f32 = 3.0;
const FACE_MIN_SCORE: f32 = 1.4;
/// Object-based rotation needs total confidence and a margin over the runner-up.
const OBJECT_MIN_SCORE: f32 = 0.9;
const OBJECT_MARGIN: f32 = 1.3;

impl Detectors {
    pub fn load() -> Result<Self> {
        let face = objdetect::FaceDetectorYN::create(
            crate::paths::model_path("yunet.onnx")?.to_string_lossy().as_ref(),
            "",
            Size::new(320, 320),
            FACE_CONF,
            0.3,
            5000,
            0,
            0,
        )?;
        let obj = match (
            crate::paths::model_path("ssd_mobilenet.prototxt"),
            crate::paths::model_path("ssd_mobilenet.caffemodel"),
        ) {
            (Ok(proto), Ok(weights)) => dnn::read_net_from_caffe(
                proto.to_string_lossy().as_ref(),
                weights.to_string_lossy().as_ref(),
            )
            .ok(),
            _ => {
                eprintln!("warn: MobileNet-SSD unavailable, object-based orientation disabled");
                None
            }
        };
        Ok(Self { face, obj })
    }

    /// Return the clockwise rotation (0/90/180/270) that makes `crop` upright,
    /// plus which detector decided it ("face" / "object" / "none").
    pub fn best_rotation(&mut self, crop: &Mat) -> Result<(i32, &'static str)> {
        let variants = rotation_variants(crop, FACE_MAX_SIDE, FACE_MIN_SIDE)?;

        // 1) Faces, verified by landmark pose (eyes above nose above mouth,
        // eye line near-horizontal). The true face also yields the largest,
        // most stable detection box, so scores are size-normalized across
        // rotations before comparing.
        let mut per_rot: Vec<(i32, Vec<(f32, f64, bool)>)> = Vec::new();
        let mut max_area = 0.0f64;
        for (deg, img) in &variants {
            if let Some(dets) = self.face_detections(img)? {
                for (_, area, _) in &dets {
                    if *area > max_area {
                        max_area = *area;
                    }
                }
                per_rot.push((*deg, dets));
            }
        }
        let mut scores: Vec<(f32, i32)> = Vec::new();
        for (deg, dets) in &per_rot {
            let mut score = 0.0f32;
            for (conf, area, pose_ok) in dets {
                let size_factor = if max_area > 0.0 {
                    0.5 + 0.5 * (area / max_area) as f32
                } else {
                    1.0
                };
                score += conf * if *pose_ok { POSE_GAIN } else { 0.15 } * size_factor;
            }
            if score > 0.0 {
                scores.push((score, *deg));
            }
        }
        eprintln!("[orient] face scores={:?}", scores);
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        if let Some(&(best_score, best_deg)) = scores.first() {
            if best_score >= FACE_MIN_SCORE {
                let ties: Vec<i32> = scores
                    .iter()
                    .take_while(|(s, _)| *s >= best_score * TIE_RATIO)
                    .map(|&(_, deg)| deg)
                    .collect();
                let winner = ties.iter().copied().min().unwrap_or(best_deg);
                return Ok((if ties.len() == 1 { best_deg } else { winner }, "face"));
            }
        }

        // 2) General objects — only trust the detector when it is decisive;
        // otherwise keep 0° (the common case for scanner-placed photos).
        if let Some(net) = self.obj.as_mut() {
            let mut scores: Vec<(f32, i32)> = Vec::new();
            for (deg, img) in &variants {
                let score = object_score(net, img)?;
                if score > 0.0 {
                    scores.push((score, *deg));
                }
            }
            if scores.len() >= 2 {
                scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                if scores[0].0 >= OBJECT_MIN_SCORE
                    && scores[0].0 >= OBJECT_MARGIN * scores[1].0.max(0.01)
                {
                    return Ok((scores[0].1, "object"));
                }
            } else if let Some(&(score, deg)) = scores.first() {
                if score >= OBJECT_MIN_SCORE {
                    return Ok((deg, "object"));
                }
            }
        }
        Ok((0, "none"))
    }

    /// Pose-weighted face detections for one rotation, or None if no faces.
    /// A face counts as upright when its YuNet landmarks read eyes-above-
    /// nose-above-mouth with a near-horizontal eye line.
    fn face_detections(&mut self, img: &Mat) -> Result<Option<Vec<(f32, f64, bool)>>> {
        let mut det = FaceDet { ptr: &mut self.face };
        det.set_input_size(img.size()?)?;
        let mut faces = Mat::default();
        det.detect(img, &mut faces)?;
        if faces.rows() == 0 {
            return Ok(None);
        }
        let data = crate::detect::mat_typed::<f32>(&faces)?;
        let mut out = Vec::new();
        for r in 0..faces.rows() as usize {
            let b = r * 15;
            let conf = data[b + 14];
            if conf < FACE_CONF {
                continue;
            }
            // landmarks: 4..6 right eye, 6..8 left eye, 8..10 nose,
            // 10..12 right mouth corner, 12..14 left mouth corner
            let (rex, rey) = (data[b + 4], data[b + 5]);
            let (lex, ley) = (data[b + 6], data[b + 7]);
            let eyes_y = (rey + ley) * 0.5;
            let nose_y = data[b + 9];
            let mouth_y = (data[b + 11] + data[b + 13]) * 0.5;
            // the eye line must be near-horizontal for a truly upright face —
            // this rejects upside-down / sideways misdetections
            let eye_dx = (lex - rex).abs();
            let eye_dy = (ley - rey).abs();
            let horizontal = eye_dy.atan2(eye_dx.max(1.0)).to_degrees() < 35.0;
            let upright_pose =
                horizontal && eyes_y + 1.5 < nose_y && nose_y + 0.5 < mouth_y;
            let area = (data[b + 2] * data[b + 3]) as f64;
            // a real face is never a sizable fraction of the whole photo —
            // huge "detections" are texture false positives
            if area > 0.12 * (img.cols() * img.rows()) as f64 {
                continue;
            }
            out.push((conf, area, upright_pose));
        }
        Ok(Some(out))
    }
}

/// Diagnostic: per-rotation face detections with pose verdicts.
/// (rotation degrees cw, rotated & downscaled image) for 0/90/180/270.
fn rotation_variants(crop: &Mat, max_side: i32, min_side: i32) -> Result<Vec<(i32, Mat)>> {
    let long = crop.cols().max(crop.rows());
    let mut base = Mat::default();
    if long > max_side || long < min_side {
        let s = if long > max_side {
            max_side as f64 / long as f64
        } else {
            (min_side as f64 / long as f64).min(2.0)
        };
        imgproc::resize(
            crop,
            &mut base,
            Size::new(
                (crop.cols() as f64 * s).round() as i32,
                (crop.rows() as f64 * s).round() as i32,
            ),
            0.0,
            0.0,
            if s > 1.0 {
                imgproc::INTER_LINEAR
            } else {
                imgproc::INTER_AREA
            },
        )?;
    } else {
        base = crop.clone();
    }
    let mut out = vec![(0i32, base.clone())];
    for (deg, code) in [
        (90, core::ROTATE_90_CLOCKWISE),
        (180, core::ROTATE_180),
        (270, core::ROTATE_90_COUNTERCLOCKWISE),
    ] {
        // always rotate from the base: chaining from the previous variant
        // would silently skip some rotations (e.g. never testing 180°)
        let mut r = Mat::default();
        core::rotate(&base, &mut r, code)?;
        out.push((deg, r));
    }
    Ok(out)
}

fn object_score(net: &mut dnn::Net, img: &Mat) -> Result<f32> {
    let mut small = Mat::default();
    imgproc::resize(
        img,
        &mut small,
        Size::new(SSD_INPUT, SSD_INPUT),
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;
    let blob = dnn::blob_from_image(
        &small,
        1.0 / 127.5,
        Size::new(SSD_INPUT, SSD_INPUT),
        Scalar::all(127.5),
        false,
        false,
        CV_32F,
    )?;
    net.set_input(&blob, "", 1.0, Scalar::all(0.0))?;
    let out = net.forward_single("")?;
    let data = crate::detect::mat_typed::<f32>(&out)?;
    let n = data.len() / 7;
    let mut score = 0.0f32;
    for i in 0..n {
        let conf = data[i * 7 + 2];
        if conf > OBJ_CONF {
            score += conf;
        }
    }
    Ok(score)
}

pub fn debug_faces(detectors: &mut Detectors, crop: &Mat) -> Result<String> {
    let variants = rotation_variants(crop, FACE_MAX_SIDE, FACE_MIN_SIDE)?;
    let mut out = String::new();
    for (deg, img) in &variants {
        if std::env::var_os("ORIENT_DEBUG").is_some() {
            let r = imgcodecs::imwrite(
                &format!("/tmp/orient_dbg/variant_{deg}.png"),
                img,
                &Vector::new(),
            );
            eprintln!("[dump] variant_{deg}: {} {}x{}", r.is_ok(), img.cols(), img.rows());
        }
        match detectors.face_detections(img)? {
            Some(dets) => {
                out.push_str(&format!("rot {deg}°: {} face(s)\n", dets.len()));
                for (conf, area, pose_ok) in &dets {
                    out.push_str(&format!(
                        "  conf={conf:.2} area={area:.0} pose_ok={pose_ok}\n"
                    ));
                }
            }
            None => out.push_str(&format!("rot {deg}°: 0 face(s)\n")),
        }
    }
    Ok(out)
}

/// Public helper: rotate a crop upright with the chosen rotation.
pub fn upright(crop: &Mat, deg: i32) -> Result<Mat> {
    rotate_cw(crop, deg)
}
