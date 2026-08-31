//! Phantom (dreamoving/Phantom PASD) upscaling sidecar bridge.
//!
//! The sidecar lives in `sidecar/phantom` (see its README): a uv-managed
//! Python worker that loads SD1.5 + PASD once and answers JSON-line
//! requests on stdin/stdout. This module owns the child process on the
//! ml-service thread, maps its events to progress callbacks and honors
//! the shared cancel flag.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

/// How long to wait for the sidecar to load its models and report ready.
const READY_TIMEOUT: Duration = Duration::from_secs(240);
/// Cancel-flag polling granularity while waiting for sidecar events.
const POLL: Duration = Duration::from_millis(250);

pub struct Status {
    pub available: bool,
    pub models_ready: bool,
    pub detail: String,
}

#[derive(Serialize, Clone)]
pub struct StatusOut {
    pub available: bool,
    pub models_ready: bool,
    pub detail: String,
}

enum Event {
    Ready,
    Progress { pct: f32, msg: String },
    Done { output: String },
    Error { msg: String },
    Fatal { msg: String },
    Eof,
}

/// If `d` is (or contains at `sidecar/phantom/`) the sidecar directory,
/// return that sidecar directory.
fn probe_sidecar(d: &Path) -> Option<PathBuf> {
    if d.join("phantom_upscale.py").is_file() {
        return Some(d.to_path_buf());
    }
    let sub = d.join("sidecar").join("phantom");
    if sub.join("phantom_upscale.py").is_file() {
        return Some(sub);
    }
    None
}

/// Locate the sidecar directory (the one holding phantom_upscale.py).
/// Search order: `$PHANTOM_SIDECAR_DIR`, walk up from the executable,
/// walk up from the CWD. Returns None when the sidecar isn't installed.
fn sidecar_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("PHANTOM_SIDECAR_DIR") {
        if let Some(p) = probe_sidecar(Path::new(&d)) {
            return Some(p);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(Path::to_path_buf);
        while let Some(d) = dir {
            if let Some(p) = probe_sidecar(&d) {
                return Some(p);
            }
            dir = d.parent().map(Path::to_path_buf);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        let mut dir = Some(cwd);
        while let Some(d) = dir {
            if let Some(p) = probe_sidecar(&d) {
                return Some(p);
            }
            dir = d.parent().map(Path::to_path_buf);
        }
    }
    None
}

fn models_marker(dir: &Path) -> bool {
    dir.join("models").join(".ready").exists()
}

fn venv_python(dir: &Path) -> PathBuf {
    dir.join(".venv").join("bin").join("python")
}

/// Cheap availability probe (no process spawn): sidecar present, venv
/// python present, models downloaded.
pub fn probe() -> Status {
    match sidecar_dir() {
        None => Status {
            available: false,
            models_ready: false,
            detail: "sidecar not found (sidecar/phantom)".into(),
        },
        Some(dir) => {
            let models_ready = models_marker(&dir);
            let py = venv_python(&dir);
            if !py.exists() {
                Status {
                    available: false,
                    models_ready,
                    detail: "sidecar venv missing — run `uv sync` in sidecar/phantom".into(),
                }
            } else if !models_ready {
                Status {
                    available: false,
                    models_ready,
                    detail: "models missing — run download_models.sh in sidecar/phantom".into(),
                }
            } else {
                Status { available: true, models_ready, detail: String::new() }
            }
        }
    }
}

/// Long-lived sidecar process. Must stay on the ml-service thread; the
/// reader thread only feeds the event channel.
pub struct Worker {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<Event>,
    dead: bool,
}

impl Worker {
    /// Spawn the sidecar and wait for its ready event.
    pub fn spawn() -> Result<Worker> {
        let dir = sidecar_dir().context("Phantom sidecar directory not found")?;
        let py = venv_python(&dir);
        anyhow::ensure!(py.exists(), "sidecar python missing: {}", py.display());
        let mut child = Command::new(&py)
            .current_dir(&dir)
            .arg("phantom_upscale.py")
            .arg("--models")
            .arg(dir.join("models"))
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("failed to spawn {}", py.display()))?;

        let stdout = child.stdout.take().context("no stdout")?;
        let stdin = child.stdin.take().context("no stdin")?;
        let (tx, rx) = channel::<Event>();
        std::thread::Builder::new()
            .name("phantom-sidecar".into())
            .spawn(move || reader(stdout, tx))
            .expect("spawn phantom reader thread");

        let mut w = Worker { child, stdin, rx, dead: false };
        // Wait for ready (or any terminal event) before the first request.
        let started = Instant::now();
        loop {
            if started.elapsed() > READY_TIMEOUT {
                w.kill();
                bail!("Phantom sidecar did not become ready in {}s", READY_TIMEOUT.as_secs());
            }
            match w.rx.recv_timeout(POLL) {
                Ok(Event::Ready) => return Ok(w),
                Ok(Event::Fatal { msg }) => {
                    w.mark_dead();
                    bail!("Phantom sidecar failed to load: {msg}");
                }
                Ok(Event::Eof) => {
                    w.mark_dead();
                    bail!("Phantom sidecar exited during startup");
                }
                Ok(_) => {}
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => {
                    w.mark_dead();
                    bail!("Phantom sidecar reader stopped during startup");
                }
            }
        }
    }

    fn mark_dead(&mut self) {
        self.dead = true;
        let _ = self.child.wait();
    }

    /// Kill the child (cancel path). The next request respawns it.
    pub fn kill(&mut self) {
        if !self.dead {
            let _ = self.child.kill();
            self.mark_dead();
        }
    }

    /// Run one image through the sidecar. Blocking; reports progress as
    /// (0..1, stage message); bails on cancel (killing the child).
    pub fn upscale(
        &mut self,
        input: &Path,
        output: &Path,
        scale: u32,
        id: &str,
        progress: &mut dyn FnMut(f32, &str),
        cancel: &dyn Fn() -> bool,
    ) -> Result<()> {
        if self.dead {
            *self = Worker::spawn()?;
        }
        let req = serde_json::json!({
            "id": id,
            "input": input.to_string_lossy(),
            "output": output.to_string_lossy(),
            "scale": scale,
            "seed": 0,
        });
        if writeln!(self.stdin, "{req}").and_then(|_| self.stdin.flush()).is_err() {
            // Broken pipe: child died between requests; retry once fresh.
            *self = Worker::spawn()?;
            writeln!(self.stdin, "{req}")
                .and_then(|_| self.stdin.flush())
                .context("Phantom sidecar not accepting requests")?;
        }

        loop {
            if cancel() {
                self.kill();
                bail!("cancelled");
            }
            match self.rx.recv_timeout(POLL) {
                Ok(Event::Progress { pct, msg }) => progress(pct, &msg),
                Ok(Event::Done { output: out }) => {
                    let p = PathBuf::from(&out);
                    if p.exists() {
                        return Ok(());
                    }
                    bail!("Phantom sidecar reported done but {} is missing", out);
                }
                Ok(Event::Error { msg }) => bail!("Phantom: {msg}"),
                Ok(Event::Fatal { msg }) => {
                    self.mark_dead();
                    bail!("Phantom sidecar fatal: {msg}");
                }
                Ok(Event::Eof) => {
                    self.mark_dead();
                    bail!("Phantom sidecar exited unexpectedly");
                }
                Ok(Event::Ready) => {}
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => {
                    self.mark_dead();
                    bail!("Phantom sidecar stopped responding");
                }
            }
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.kill();
    }
}

fn reader(stdout: std::process::ChildStdout, tx: Sender<Event>) {
    let mut lines = BufReader::new(stdout).lines();
    while let Some(Ok(line)) = lines.next() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue; // tolerate stray output; diagnostics belong on stderr
        };
        let ev = match v.get("event").and_then(|e| e.as_str()) {
            Some("ready") => Event::Ready,
            Some("progress") => Event::Progress {
                pct: v.get("pct").and_then(|p| p.as_f64()).unwrap_or(0.0) as f32,
                msg: v.get("msg").and_then(|m| m.as_str()).unwrap_or("").into(),
            },
            Some("done") => Event::Done {
                output: v.get("output").and_then(|o| o.as_str()).unwrap_or("").into(),
            },
            Some("error") => Event::Error {
                msg: v.get("msg").and_then(|m| m.as_str()).unwrap_or("unknown").into(),
            },
            Some("fatal") => Event::Fatal {
                msg: v.get("msg").and_then(|m| m.as_str()).unwrap_or("unknown").into(),
            },
            _ => continue,
        };
        if tx.send(ev).is_err() {
            return;
        }
    }
    let _ = tx.send(Event::Eof);
}
