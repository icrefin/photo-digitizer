//! End-to-end pipeline shared by the Tauri app and the CLI test harness.

use crate::detect::{self, detect_quads, draw_debug, extract_photo, Quad};
use crate::orient::Detectors;
use anyhow::Result;
use opencv::core::Mat;
use std::path::Path;

pub struct ExtractedPhoto {
    pub index: usize,
    pub quad_count: usize,
    pub rotation: i32,
    pub method: &'static str,
    pub mat: Mat,
}

/// Detect, extract, upright and white-trim all photos on one scanned page.
/// Returns the quads (page coords) plus the extracted photos.
pub fn detect_and_extract(page: &Mat, det: &mut Detectors) -> Result<(Vec<Quad>, Vec<ExtractedPhoto>)> {
    let quads = detect_quads(page)?;
    let mut out = Vec::new();
    for (i, quad) in quads.iter().enumerate() {
        let crop = extract_photo(page, quad)?;
        // Trim scan margins BEFORE orientation: white borders shrink the face
        // relative to the detection resize and skew the landmark pose check.
        let crop = detect::trim_white_borders(&crop)?;
        if std::env::var_os("ORIENT_DEBUG").is_some() {
            let _ = crate::enhance::save_mat(
                &crop,
                std::path::Path::new(&format!("/tmp/orient_dbg/crop_{i}.png")),
                95,
            )?;
        }
        let (rotation, method) = det.best_rotation(&crop)?;
        let mat = detect::rotate_cw(&crop, rotation)?;
        out.push(ExtractedPhoto {
            index: i,
            quad_count: quads.len(),
            rotation,
            method,
            mat,
        });
    }
    Ok((quads, out))
}

pub fn process_scan(path: &Path, det: &mut Detectors) -> Result<Vec<ExtractedPhoto>> {
    let page = crate::enhance::load_mat(path)?;
    Ok(detect_and_extract(&page, det)?.1)
}

/// Downscaled sheet with detected quads drawn on top, as JPEG bytes.
pub fn sheet_preview(page: &Mat, quads: &[Quad]) -> Result<Vec<u8>> {
    let vis = draw_debug(page, quads)?;
    detect::encode_jpeg(&vis, 1000, 85)
}

/// Save a debug overlay image next to `out_dir`.
pub fn save_debug(path: &Path, out_dir: &Path) -> Result<Option<std::path::PathBuf>> {
    let page = crate::enhance::load_mat(path)?;
    let quads = detect_quads(&page)?;
    if quads.is_empty() {
        return Ok(None);
    }
    let vis = draw_debug(&page, &quads)?;
    let out = out_dir.join(format!(
        "{}_debug.png",
        path.file_stem().and_then(|s| s.to_str()).unwrap_or("scan")
    ));
    crate::enhance::save_mat(&vis, &out, 90)?;
    Ok(Some(out))
}
