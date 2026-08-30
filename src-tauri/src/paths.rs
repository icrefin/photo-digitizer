//! Model / temp file path resolution shared by the app and the CLI test harness.

use std::path::{Path, PathBuf};

/// Marker file used to identify the models directory.
const MARKER: &str = "yunet.onnx";

/// Locate the models directory. Search order:
/// 1. `$MODELS_DIR`
/// 2. walk up from the executable (covers `target/debug`, bundled resources)
/// 3. walk up from the CWD and `<cwd>/src-tauri` (covers running inside the project)
pub fn models_dir() -> anyhow::Result<PathBuf> {
    if let Ok(d) = std::env::var("MODELS_DIR") {
        let p = PathBuf::from(d);
        if p.join(MARKER).exists() {
            return Ok(p);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(Path::to_path_buf);
        while let Some(d) = dir {
            let cand = d.join("models");
            if cand.join(MARKER).exists() {
                return Ok(cand);
            }
            dir = d.parent().map(Path::to_path_buf);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        let mut dir = Some(cwd);
        while let Some(d) = dir {
            for cand in [d.join("models"), d.join("src-tauri").join("models")] {
                if cand.join(MARKER).exists() {
                    return Ok(cand);
                }
            }
            dir = d.parent().map(Path::to_path_buf);
        }
    }
    anyhow::bail!(
        "models directory not found (looked for {MARKER}); set MODELS_DIR to src-tauri/models"
    )
}

pub fn model_path(name: &str) -> anyhow::Result<PathBuf> {
    let p = models_dir()?.join(name);
    anyhow::ensure!(p.exists(), "model file missing: {}", p.display());
    Ok(p)
}

/// Point ONNX Runtime at the Homebrew dylib unless the caller set ORT_DYLIB_PATH.
/// Must run before the first `ort::Session` is created.
pub fn ensure_ort_dylib() {
    if std::env::var_os("ORT_DYLIB_PATH").is_some_and(|v| !v.is_empty()) {
        return;
    }
    let candidates = [
        "/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib",
        "/usr/local/opt/onnxruntime/lib/libonnxruntime.dylib",
    ];
    for c in candidates {
        if Path::new(c).exists() {
            std::env::set_var("ORT_DYLIB_PATH", c);
            return;
        }
    }
}

/// Temp directory holding full-res extracted / enhanced photos for this session.
pub fn jobs_dir() -> PathBuf {
    std::env::temp_dir().join("photo-digitizer-jobs")
}

pub fn clean_jobs(keep: &[PathBuf]) {
    let Ok(entries) = std::fs::read_dir(jobs_dir()) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if keep.iter().any(|k| k == &p) {
            continue;
        }
        let _ = if p.is_dir() {
            std::fs::remove_dir_all(&p)
        } else {
            std::fs::remove_file(&p)
        };
    }
}

/// Default folder shown in the app on launch.
pub fn default_scan_dir() -> Option<String> {
    let home = std::env::var("HOME").ok()?;
    let cand = format!("{home}/workspace/digitize_old_photos/Scanned");
    if Path::new(&cand).is_dir() {
        return Some(cand);
    }
    None
}
