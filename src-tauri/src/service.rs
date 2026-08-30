//! A single worker thread owns every model handle (OpenCV DNN nets are not
//! Send, ONNX Runtime sessions are cheap to keep) and processes jobs FIFO.

use crate::enhance::{self, Enhancers};
use crate::orient::Detectors;
use crate::paths;
use crate::pipeline;
use anyhow::{bail, Result};
use base64::Engine;
use opencv::core::{Mat, MatTraitConst, Point2f};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::OnceLock;

/// Set when the user cancels the running enhancement batch.
static CANCELLED: AtomicBool = AtomicBool::new(false);

pub fn cancel_enhance() {
    CANCELLED.store(true, Ordering::SeqCst);
}

pub fn reset_cancel() {
    CANCELLED.store(false, Ordering::SeqCst);
}

fn is_cancelled() -> bool {
    CANCELLED.load(Ordering::SeqCst)
}

/// Result of detecting one sheet: preview image + extracted photos.
#[derive(Clone)]
pub struct SheetDetected {
    pub sheet_b64: String,
    pub page_w: i32,
    pub page_h: i32,
    pub photos: Vec<DetectedPhoto>,
}

#[derive(Clone, serde::Serialize)]
pub struct DetectedPhoto {
    pub id: String,
    pub index: usize,
    pub rotation: i32,
    /// How the upright orientation was determined: face / object / none.
    pub method: String,
    pub width: i32,
    pub height: i32,
    pub extracted: PathBuf,
    pub thumb_b64: String,
    /// Page-space quad (TL, TR, BR, BL) the photo was cropped from.
    pub quad: Vec<f64>,
}

#[derive(Clone, serde::Serialize)]
pub struct EnhancedPhoto {
    pub id: String,
    pub enhanced: PathBuf,
    pub thumb_b64: String,
    pub width: i32,
    pub height: i32,
}

pub type ProgressCb = Box<dyn FnMut(usize, usize, &str) + Send>;

enum Msg {
    Detect {
        path: PathBuf,
        /// Page-space quads of photos the user manually cropped on this sheet
        /// (photo index, TL TR BR BL): their boxes override the fresh ones in
        /// the preview and re-detection must not resurrect auto crops for them.
        manual: Vec<(usize, Vec<f64>)>,
        reply: Sender<Result<SheetDetected>>,
    },
    ReExtract {
        path: PathBuf,
        quad: Vec<f64>,
        id: String,
        reply: Sender<Result<DetectedPhoto>>,
    },
    Rotate {
        id: String,
        deg: i32,
        reply: Sender<Result<DetectedPhoto>>,
    },
    Enhance {
        id: String,
        path: PathBuf,
        upscale: bool,
        colorize: bool,
        faces: bool,
        progress: ProgressCb,
        reply: Sender<Result<EnhancedPhoto>>,
    },
}

fn service() -> &'static Sender<Msg> {
    static S: OnceLock<Sender<Msg>> = OnceLock::new();
    S.get_or_init(|| {
        let (tx, rx) = channel::<Msg>();
        std::thread::Builder::new()
            .name("ml-service".into())
            .spawn(move || run_loop(rx))
            .expect("spawn ml-service thread");
        tx
    })
}

fn run_loop(rx: Receiver<Msg>) {
    let mut det: Option<Detectors> = None;
    let mut enh: Option<Enhancers> = None;
    while let Ok(msg) = rx.recv() {
        match msg {
            Msg::Detect { path, manual, reply } => {
                let r = (|| -> Result<SheetDetected> {
                    std::fs::create_dir_all(paths::jobs_dir())?;
                    let d = det.get_or_insert_with(|| {
                        Detectors::load().expect("failed to load detectors")
                    });
                    let page = enhance::load_mat(&path)?;
                    let (quads, photos) = pipeline::detect_and_extract(&page, d)?;
                    // Overlay manual crop boxes so the preview keeps showing
                    // the user's adjusted boundaries after a re-detection.
                    let mut vis_quads = quads.clone();
                    let mut vis_manual = vec![false; vis_quads.len()];
                    let mut manual = manual;
                    manual.sort_by_key(|(i, _)| *i);
                    for (idx, flat) in &manual {
                        if flat.len() != 8 {
                            continue;
                        }
                        let q: crate::detect::Quad = [
                            Point2f::new(flat[0] as f32, flat[1] as f32),
                            Point2f::new(flat[2] as f32, flat[3] as f32),
                            Point2f::new(flat[4] as f32, flat[5] as f32),
                            Point2f::new(flat[6] as f32, flat[7] as f32),
                        ];
                        if *idx < vis_quads.len() {
                            vis_quads[*idx] = q;
                            vis_manual[*idx] = true;
                        } else {
                            vis_quads.push(q);
                            vis_manual.push(true);
                        }
                    }
                    let sheet_jpg = pipeline::sheet_preview(&page, &vis_quads, &vis_manual)?;
                    let sheet_b64 =
                        base64::engine::general_purpose::STANDARD.encode(&sheet_jpg);
                    let stem = stem_of(&path);
                    let mut out = Vec::new();
                    for p in photos {
                        let id = format!("{}#{}", stem, p.index);
                        let extracted =
                            paths::jobs_dir().join(format!("{}_{}.png", stem, p.index));
                        enhance::save_mat(&p.mat, &extracted, 95)?;
                        let jpg = crate::detect::encode_jpeg(&p.mat, 560, 88)?;
                        let thumb_b64 =
                            base64::engine::general_purpose::STANDARD.encode(&jpg);
                        let quad: Vec<f64> = quads
                            .get(p.index)
                            .map(|q| q.iter().flat_map(|pt| [pt.x as f64, pt.y as f64]).collect())
                            .unwrap_or_default();
                        out.push(DetectedPhoto {
                            id,
                            index: p.index,
                            rotation: p.rotation,
                            method: p.method.to_string(),
                            width: p.mat.cols(),
                            height: p.mat.rows(),
                            extracted,
                            thumb_b64,
                            quad,
                        });
                    }
                    Ok(SheetDetected { sheet_b64, page_w: page.cols(), page_h: page.rows(), photos: out })
                })();
                let _ = reply.send(r);
            }
            Msg::Rotate { id, deg, reply } => {
                let r = (|| -> Result<DetectedPhoto> {
                    std::fs::create_dir_all(paths::jobs_dir())?;
                    let mut it = id.split('#');
                    let stem = it.next().unwrap_or("photo");
                    let idx = it.next().unwrap_or("0");
                    let path = paths::jobs_dir().join(format!("{stem}_{idx}.png"));
                    let img = enhance::load_mat(&path)?;
                    let deg = deg.rem_euclid(360);
                    let rotated = crate::detect::rotate_cw(&img, deg)?;
                    enhance::save_mat(&rotated, &path, 95)?;
                    let jpg = crate::detect::encode_jpeg(&rotated, 560, 88)?;
                    Ok(DetectedPhoto {
                        id,
                        index: 0,
                        rotation: deg,
                        method: "manual".into(),
                        width: rotated.cols(),
                        height: rotated.rows(),
                        extracted: path,
                        thumb_b64: base64::engine::general_purpose::STANDARD.encode(&jpg),
                        quad: Vec::new(),
                    })
                })();
                let _ = reply.send(r);
            }
            Msg::ReExtract { path, quad, id, reply } => {
                let r = (|| -> Result<DetectedPhoto> {
                    std::fs::create_dir_all(paths::jobs_dir())?;
                    let d = det.get_or_insert_with(|| {
                        Detectors::load().expect("failed to load detectors")
                    });
                    anyhow::ensure!(quad.len() == 8, "quad must have 8 coordinates");
                    let pts: Vec<Point2f> = quad
                        .chunks_exact(2)
                        .map(|c| Point2f::new(c[0] as f32, c[1] as f32))
                        .collect();
                    let page = enhance::load_mat(&path)?;
                    let quad_arr = [pts[0], pts[1], pts[2], pts[3]];
                    let crop = crate::detect::extract_photo(&page, &quad_arr)?;
                    let crop = crate::detect::trim_white_borders(&crop)?;
                    let (rotation, method) = d.best_rotation(&crop)?;
                    let mat = crate::detect::rotate_cw(&crop, rotation)?;
                    let stem = stem_of(&path);
                    let idx = id.rsplit('#').next().unwrap_or("0").to_string();
                    let extracted = paths::jobs_dir().join(format!("{}_{}.png", stem, idx));
                    enhance::save_mat(&mat, &extracted, 95)?;
                    let jpg = crate::detect::encode_jpeg(&mat, 560, 88)?;
                    let thumb_b64 = base64::engine::general_purpose::STANDARD.encode(&jpg);
                    let id2 = id.clone();
                    Ok(DetectedPhoto {
                        id: id2,
                        index: idx.parse().unwrap_or(0),
                        rotation,
                        method: method.to_string(),
                        width: mat.cols(),
                        height: mat.rows(),
                        extracted,
                        thumb_b64,
                        quad,
                    })
                })();
                let _ = reply.send(r);
            }
            Msg::Enhance { id, path, upscale, colorize, faces, mut progress, reply } => {
                let r = (|| -> Result<EnhancedPhoto> {
                    if is_cancelled() {
                        bail!("cancelled");
                    }
                    std::fs::create_dir_all(paths::jobs_dir())?;
                    let stages =
                        (upscale as usize + colorize as usize + faces as usize).max(1);
                    // Honest staging: the first model load takes ~1s, so say so
                    // instead of going silent until the first inference event.
                    progress(0, stages, "loading AI models…");
                    let e = enh.get_or_insert_with(|| {
                        Enhancers::load().expect("failed to load enhancers")
                    });
                    let img = enhance::load_mat(&path)?;
                    let mut stage = 0usize;
                    let mut mat = img;

                    if upscale {
                        progress(stage, stages, "starting upscale…");
                        let mut last = 0f32;
                        mat = e.upscale(
                            &mat,
                            &mut |f| {
                                if (f * 20.0) as usize > (last * 20.0) as usize || f >= 1.0 {
                                    last = f;
                                    progress(stage, stages, &format!("upscaling {f:.0}%"));
                                }
                            },
                            &|| is_cancelled(),
                        )?;
                        stage += 1;
                    }
                    if colorize {
                        if is_cancelled() {
                            bail!("cancelled");
                        }
                        progress(stage, stages, "colorizing…");
                        mat = e.colorize(&mat)?;
                    }
                    if faces {
                        if is_cancelled() {
                            bail!("cancelled");
                        }
                        progress(stage, stages, "restoring faces…");
                        let mut last = 0usize;
                        mat = enhance::restore_faces(e, &mat, &mut |done, total| {
                            if done > last || done == total {
                                last = done;
                                progress(
                                    stage,
                                    stages,
                                    &format!("restoring faces {done}/{total}"),
                                );
                            }
                        }, &|| is_cancelled())?;
                    }

                    let stem = stem_of(&path);
                    let out = paths::jobs_dir().join(format!("{stem}_enh.png"));
                    enhance::save_mat(&mat, &out, 95)?;
                    let jpg = crate::detect::encode_jpeg(&mat, 900, 90)?;
                    progress(stages, stages, "done");
                    Ok(EnhancedPhoto {
                        id,
                        enhanced: out,
                        thumb_b64: base64::engine::general_purpose::STANDARD.encode(&jpg),
                        width: mat.cols(),
                        height: mat.rows(),
                    })
                })();
                let _ = reply.send(r);
            }
        }
    }
}

fn stem_of(p: &Path) -> String {
    p.file_stem().and_then(|s| s.to_str()).unwrap_or("scan").to_string()
}

/// Run detection for one scan file; blocks until the service answers.
/// `manual` lists user-adjusted crops (photo index, page-space quad) whose
/// boxes must survive the fresh detection in the returned preview.
pub fn detect_file(path: PathBuf, manual: Vec<(usize, Vec<f64>)>) -> Result<SheetDetected> {
    let (tx, rx) = channel();
    service()
        .send(Msg::Detect { path, manual, reply: tx })
        .map_err(|_| anyhow::anyhow!("ml-service channel closed"))?;
    rx.recv()?
}

/// Re-crop a photo with a user-adjusted quad (page coords TL TR BR BL).
pub fn re_extract(path: PathBuf, quad: Vec<f64>, id: String) -> Result<DetectedPhoto> {
    let (tx, rx) = channel();
    service()
        .send(Msg::ReExtract { path, quad, id, reply: tx })
        .map_err(|_| anyhow::anyhow!("ml-service channel closed"))?;
    rx.recv()?
}

pub fn rotate_photo(id: String, deg: i32) -> Result<DetectedPhoto> {
    let (tx, rx) = channel();
    service()
        .send(Msg::Rotate { id, deg, reply: tx })
        .map_err(|_| anyhow::anyhow!("ml-service channel closed"))?;
    rx.recv()?
}

pub fn enhance_photo(
    id: String,
    path: PathBuf,
    upscale: bool,
    colorize: bool,
    faces: bool,
    progress: ProgressCb,
) -> Result<EnhancedPhoto> {
    let (tx, rx) = channel();
    service()
        .send(Msg::Enhance { id, path, upscale, colorize, faces, progress, reply: tx })
        .map_err(|_| anyhow::anyhow!("ml-service channel closed"))?;
    rx.recv()?
}

/// Recompute a thumbnail + metadata from a temp PNG (used after manual rotate).
#[allow(dead_code)]
pub fn photo_from_path(id: &str, path: &Path) -> Result<DetectedPhoto> {
    let img = enhance::load_mat(path)?;
    let jpg = crate::detect::encode_jpeg(&img, 560, 88)?;
    Ok(DetectedPhoto {
        id: id.to_string(),
        index: 0,
        rotation: 0,
        method: "manual".into(),
        width: img.cols(),
        height: img.rows(),
        extracted: path.to_path_buf(),
        thumb_b64: base64::engine::general_purpose::STANDARD.encode(&jpg),
        quad: Vec::new(),
    })
}

/// Encode a Mat as a base64 JPEG thumbnail.
pub fn thumb_b64(mat: &Mat, max_side: i32) -> Result<String> {
    let jpg = crate::detect::encode_jpeg(mat, max_side, 88)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(&jpg))
}
