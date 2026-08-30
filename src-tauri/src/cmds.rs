//! Tauri command surface + UI-facing state.

use crate::paths;
use crate::service::{self, ProgressCb};
use serde::Serialize;
use std::collections::HashMap;
use base64::Engine;
use opencv::core::Point2f;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, State};

const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "tif", "tiff", "bmp"];

#[derive(Default)]
pub struct AppState {
    jobs: Mutex<HashMap<String, Job>>,
}

struct Job {
    source_file: String,
    extracted: PathBuf,
    thumb_b64: String,
    width: i32,
    height: i32,
    rotation: i32,
    method: String,
    quad: Vec<f64>,
    enhanced: Option<EnhancedInfo>,
}

struct EnhancedInfo {
    path: PathBuf,
    thumb_b64: String,
    width: i32,
    height: i32,
}

#[derive(Serialize, Clone)]
pub struct PhotoMeta {
    id: String,
    source_file: String,
    thumb: String,
    width: i32,
    height: i32,
    rotation: i32,
    method: String,
    quad: Vec<f64>,
    enhanced: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    enh_thumb: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    enh_width: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    enh_height: Option<i32>,
}

#[derive(Serialize, Clone)]
pub struct DetectedSheet {
    sheet_b64: String,
    page_w: i32,
    page_h: i32,
    photos: Vec<PhotoMeta>,
}

#[derive(Serialize, Clone)]
struct FileDetected {
    file: String,
    done: usize,
    total: usize,
    sheet_b64: String,
    page_w: i32,
    page_h: i32,
    photos: Vec<PhotoMeta>,
}

#[derive(Serialize, Clone)]
struct PhotoEnhanced {
    id: String,
    meta: PhotoMeta,
}

fn e2s<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

fn list_images(dir: &Path) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(e2s)?;
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let lower = name.to_lowercase();
        if IMAGE_EXTS.iter().any(|x| lower.ends_with(x)) {
            out.push(name);
        }
    }
    out.sort();
    Ok(out)
}

fn job_meta(id: &str, job: &Job) -> PhotoMeta {
    PhotoMeta {
        id: id.to_string(),
        source_file: job.source_file.clone(),
        thumb: job.thumb_b64.clone(),
        width: job.width,
        height: job.height,
        rotation: job.rotation,
        method: job.method.clone(),
        quad: job.quad.clone(),
        enhanced: job.enhanced.is_some(),
        enh_thumb: job.enhanced.as_ref().map(|e| e.thumb_b64.clone()),
        enh_width: job.enhanced.as_ref().map(|e| e.width),
        enh_height: job.enhanced.as_ref().map(|e| e.height),
    }
}

fn make_job(source_file: &str, d: &crate::service::DetectedPhoto) -> Job {
    Job {
        source_file: source_file.to_string(),
        extracted: d.extracted.clone(),
        thumb_b64: d.thumb_b64.clone(),
        width: d.width,
        height: d.height,
        rotation: d.rotation,
        method: d.method.clone(),
        quad: d.quad.clone(),
        enhanced: None,
    }
}

#[tauri::command]
pub async fn list_scans(dir: String) -> Result<Vec<String>, String> {
    list_images(Path::new(&dir))
}

#[tauri::command]
pub async fn default_dir() -> Result<Option<String>, String> {
    Ok(paths::default_scan_dir())
}

#[tauri::command]
pub async fn detect_one(
    state: State<'_, AppState>,
    dir: String,
    file: String,
) -> Result<DetectedSheet, String> {
    let path = Path::new(&dir).join(&file);
    let detected = tauri::async_runtime::spawn_blocking(move || service::detect_file(path))
        .await
        .map_err(e2s)?
        .map_err(e2s)?;
    let mut map = state.jobs.lock().unwrap();
    let photos = detected
        .photos
        .iter()
        .map(|d| {
            let job = make_job(&file, d);
            let m = job_meta(&d.id, &job);
            map.insert(d.id.clone(), job);
            m
        })
        .collect();
    Ok(DetectedSheet {
        sheet_b64: detected.sheet_b64,
        page_w: detected.page_w,
        page_h: detected.page_h,
        photos,
    })
}

#[tauri::command]
pub async fn detect_all(
    app: AppHandle,
    state: State<'_, AppState>,
    dir: String,
) -> Result<usize, String> {
    paths::clean_jobs();
    std::fs::create_dir_all(paths::jobs_dir()).map_err(e2s)?;
    let files = list_images(Path::new(&dir))?;
    let total = files.len();
    for (i, file) in files.iter().enumerate() {
        let path = Path::new(&dir).join(file);
        let detected = tauri::async_runtime::spawn_blocking(move || service::detect_file(path))
            .await
            .map_err(e2s)?
            .map_err(e2s)?;
        let mut map = state.jobs.lock().unwrap();
        let metas = detected
            .photos
            .iter()
            .map(|d| {
                let job = make_job(file, d);
                let m = job_meta(&d.id, &job);
                map.insert(d.id.clone(), job);
                m
            })
            .collect();
        let sheet_b64 = detected.sheet_b64;
        let (page_w, page_h) = (detected.page_w, detected.page_h);
        drop(map);
        let _ = app.emit(
            "file-detected",
            FileDetected { file: file.clone(), done: i + 1, total, sheet_b64, page_w, page_h, photos: metas },
        );
    }
    Ok(total)
}

#[tauri::command]
pub async fn rotate_photo(
    state: State<'_, AppState>,
    id: String,
    deg: i32,
) -> Result<PhotoMeta, String> {
    let _extracted = {
        let map = state.jobs.lock().unwrap();
        map.get(&id).ok_or("unknown photo id")?.extracted.clone()
    };
    let id2 = id.clone();
    let d = tauri::async_runtime::spawn_blocking(move || service::rotate_photo(id2, deg))
        .await
        .map_err(e2s)?
        .map_err(e2s)?;
    let mut map = state.jobs.lock().unwrap();
    let job = map.get_mut(&id).ok_or("unknown photo id")?;
    job.rotation = (job.rotation + deg.rem_euclid(360)) % 360;
    job.thumb_b64 = d.thumb_b64.clone();
    job.width = d.width;
    job.height = d.height;
    job.method = "manual".into();
    job.enhanced = None; // previous enhancement no longer matches the rotated source
    Ok(job_meta(&id, job))
}

/// Re-crop a photo with a user-adjusted quad (page coords, TL TR BR BL).
/// Regenerate the sheet preview with a given set of quads (used after a
/// manual crop so the drawn boxes reflect the adjusted boundaries).
#[tauri::command]
pub async fn sheet_preview_with_quads(
    dir: String,
    file: String,
    quads: Vec<Vec<f64>>,
) -> Result<String, String> {
    let path = Path::new(&dir).join(&file);
    tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        let page = crate::enhance::load_mat(&path).map_err(e2s)?;
        let mut converted: Vec<[Point2f; 4]> = Vec::new();
        for q in &quads {
            if q.len() != 8 {
                return Err("quad must have 8 coordinates".into());
            }
            converted.push([
                Point2f::new(q[0] as f32, q[1] as f32),
                Point2f::new(q[2] as f32, q[3] as f32),
                Point2f::new(q[4] as f32, q[5] as f32),
                Point2f::new(q[6] as f32, q[7] as f32),
            ]);
        }
        let vis = crate::detect::draw_debug(&page, &converted).map_err(e2s)?;
        let jpg = crate::detect::encode_jpeg(&vis, 1000, 85).map_err(e2s)?;
        Ok(base64::engine::general_purpose::STANDARD.encode(&jpg))
    })
    .await
    .map_err(e2s)?
}

#[tauri::command]
pub async fn re_extract_photo(
    state: State<'_, AppState>,
    dir: String,
    file: String,
    id: String,
    quad: Vec<f64>,
) -> Result<PhotoMeta, String> {
    if quad.len() != 8 {
        return Err("quad must have 8 coordinates".into());
    }
    let path = Path::new(&dir).join(&file);
    let id2 = id.clone();
    let d = tauri::async_runtime::spawn_blocking(move || {
        service::re_extract(path, quad, id2)
    })
    .await
    .map_err(e2s)?
    .map_err(e2s)?;
    let mut map = state.jobs.lock().unwrap();
    let job = map.get_mut(&id).ok_or("unknown photo id")?;
    job.rotation = d.rotation;
    job.method = format!("manual ({})", d.method);
    job.quad = d.quad.clone();
    job.thumb_b64 = d.thumb_b64.clone();
    job.width = d.width;
    job.height = d.height;
    job.enhanced = None; // previous enhancement no longer matches the new crop
    Ok(job_meta(&id, job))
}

#[tauri::command]
pub async fn enhance_photos(
    app: AppHandle,
    state: State<'_, AppState>,
    ids: Vec<String>,
    upscale: bool,
    colorize: bool,
    faces: bool,
) -> Result<(Vec<PhotoMeta>, bool), String> {
    service::reset_cancel();
    let total = ids.len();
    let mut results = Vec::new();
    let mut cancelled = false;
    for (i, id) in ids.iter().enumerate() {
        let extracted = {
            let map = state.jobs.lock().unwrap();
            map.get(id).ok_or(format!("unknown photo id {id}"))?.extracted.clone()
        };
        let app2 = app.clone();
        let id2 = id.clone();
        let progress: ProgressCb = Box::new(move |step: usize, steps: usize, stage: &str| {
            let _ = app2.emit(
                "enhance-progress",
                serde_json::json!({
                    "id": id2, "done": i + 1, "total": total,
                    "stage": stage, "stepDone": step, "stepTotal": steps,
                }),
            );
        });
        let id3 = id.clone();
        let enhanced = tauri::async_runtime::spawn_blocking(move || {
            service::enhance_photo(id3, extracted, upscale, colorize, faces, progress)
        })
        .await
        .map_err(e2s)?;
        let enhanced = match enhanced {
            Ok(e) => e,
            Err(err) => {
                if err.to_string().contains("cancelled") {
                    cancelled = true;
                    break;
                }
                return Err(err.to_string());
            }
        };

        let mut map = state.jobs.lock().unwrap();
        let job = map.get_mut(id).ok_or("unknown photo id")?;
        job.enhanced = Some(EnhancedInfo {
            path: enhanced.enhanced.clone(),
            thumb_b64: enhanced.thumb_b64.clone(),
            width: enhanced.width,
            height: enhanced.height,
        });
        let meta = job_meta(id, job);
        let _ = app.emit("photo-enhanced", PhotoEnhanced { id: id.clone(), meta: meta.clone() });
        results.push(meta);
        if cancelled {
            break;
        }
    }
    Ok((results, cancelled))
}

#[tauri::command]
pub async fn cancel_enhance() -> Result<(), String> {
    service::cancel_enhance();
    Ok(())
}

#[tauri::command]
pub async fn save_photos(
    state: State<'_, AppState>,
    ids: Vec<String>,
    out_dir: String,
    format: String,
) -> Result<Vec<String>, String> {
    let plan: Vec<(PathBuf, PathBuf)> = {
        let map = state.jobs.lock().unwrap();
        ids.iter()
            .filter_map(|id| {
                let job = map.get(id)?;
                let stem = id.split('#').next().unwrap_or("photo");
                let idx = id.rsplit('#').next().unwrap_or("0");
                let src = job
                    .enhanced
                    .as_ref()
                    .map(|e| e.path.clone())
                    .unwrap_or_else(|| job.extracted.clone());
                let name = format!(
                    "{stem}_p{idx}.{}",
                    if format == "png" { "png" } else { "jpg" }
                );
                Some((src, Path::new(&out_dir).join(name)))
            })
            .collect()
    };
    if plan.is_empty() {
        return Ok(vec![]);
    }
    std::fs::create_dir_all(&out_dir).map_err(e2s)?;
    tauri::async_runtime::spawn_blocking(move || -> Result<Vec<String>, String> {
        let mut saved = Vec::new();
        for (src, dst) in plan {
            let mat = crate::enhance::load_mat(&src).map_err(e2s)?;
            crate::enhance::save_mat(&mat, &dst, 95).map_err(e2s)?;
            saved.push(dst.to_string_lossy().to_string());
        }
        Ok(saved)
    })
    .await
    .map_err(e2s)?
}

#[tauri::command]
pub async fn pick_folder(app: AppHandle) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    tauri::async_runtime::spawn_blocking(move || {
        let picked = app.dialog().file().blocking_pick_folder().map(|p| p.to_string());
        Ok::<Option<String>, String>(picked)
    })
    .await
    .map_err(e2s)?
}
