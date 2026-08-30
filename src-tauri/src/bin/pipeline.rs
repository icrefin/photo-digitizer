use anyhow::Result;

use opencv::core::MatTraitConst;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        eprintln!("usage: pipeline detect|enhance ...");
        std::process::exit(2);
    };
    let mut out_dir = PathBuf::from("_pipeline_out");
    let mut paths = vec![];
    let mut flags = std::collections::HashSet::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                i += 1;
                out_dir = PathBuf::from(&args[i]);
            }
            f if f.starts_with("--") => {
                flags.insert(f.to_string());
            }
            p => paths.push(PathBuf::from(p)),
        }
        i += 1;
    }
    std::fs::create_dir_all(&out_dir)?;

    match cmd.as_str() {
        "detect" => {
            let mut det = photo_digitizer_lib::orient::Detectors::load()?;
            for p in expand(&paths)? {
                let t0 = Instant::now();
                let photos = photo_digitizer_lib::pipeline::process_scan(&p, &mut det)?;
                println!(
                    "{} -> {} photo(s) in {:.2}s",
                    p.display(),
                    photos.len(),
                    t0.elapsed().as_secs_f64()
                );
                for ph in &photos {
                    let out = out_dir.join(format!(
                        "{}_p{}.png",
                        p.file_stem().and_then(|s| s.to_str()).unwrap_or("scan"),
                        ph.index
                    ));
                    photo_digitizer_lib::enhance::save_mat(&ph.mat, &out, 95)?;
                    println!(
                        "   #{idx} rot={rot}° via {method}, {w}x{h} -> {out}",
                        idx = ph.index,
                        rot = ph.rotation,
                        method = ph.method,
                        w = ph.mat.cols(),
                        h = ph.mat.rows(),
                        out = out.display()
                    );
                }
                if flags.contains("--debug") {
                    if let Some(dbg) = photo_digitizer_lib::pipeline::save_debug(&p, &out_dir)? {
                        println!("   debug: {}", dbg.display());
                    }
                }
            }
        }
        "enhance" => {
            let mut enh = photo_digitizer_lib::enhance::Enhancers::load()?;
            for p in expand(&paths)? {
                let img = photo_digitizer_lib::enhance::load_mat(&p)?;
                let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("photo");
                if flags.contains("--upscale") {
                    let t0 = Instant::now();
                    let up = enh.upscale(
                        &img,
                        &mut |f| {
                            print!("\rupscale {f:>5.0}%");
                            use std::io::Write as _;
                            std::io::stdout().flush().ok();
                        },
                        &|| false,
                    )?;
                    println!("\rupscale done in {:.1}s", t0.elapsed().as_secs_f64());
                    let out = out_dir.join(format!("{stem}_x4.png"));
                    photo_digitizer_lib::enhance::save_mat(&up, &out, 95)?;
                    println!("   {} -> {} ({}x{})", p.display(), out.display(), up.cols(), up.rows());
                }
                if flags.contains("--colorize") {
                    let t0 = Instant::now();
                    let col = enh.colorize(&img)?;
                    println!("colorize done in {:.1}s", t0.elapsed().as_secs_f64());
                    let out = out_dir.join(format!("{stem}_color.png"));
                    photo_digitizer_lib::enhance::save_mat(&col, &out, 95)?;
                    println!("   {}", out.display());
                }
                if flags.contains("--faces") {
                    let t0 = Instant::now();
                    let mut last = 0usize;
                    let col = photo_digitizer_lib::enhance::restore_faces(
                        &mut enh,
                        &img,
                        &mut |done, total| {
                            if done > last {
                                last = done;
                                print!("\rface {done}/{total}");
                                use std::io::Write as _;
                                std::io::stdout().flush().ok();
                            }
                        },
                        &|| false,
                    )?;
                    println!("\rfaces done in {:.1}s       ", t0.elapsed().as_secs_f64());
                    let out = out_dir.join(format!("{stem}_faces.png"));
                    photo_digitizer_lib::enhance::save_mat(&col, &out, 95)?;
                    println!("   {}", out.display());
                }
            }
        }
        "facedbg" => {
            let mut det = photo_digitizer_lib::orient::Detectors::load()?;
            for p in expand(&paths)? {
                let img = photo_digitizer_lib::enhance::load_mat(&p)?;
                println!("=== {} ({}x{})", p.display(), img.cols(), img.rows());
                println!("{}", photo_digitizer_lib::orient::debug_faces(&mut det, &img)?);
            }
        }
        other => {
            eprintln!("unknown command: {other}");
            std::process::exit(2);
        }
    }
    Ok(())
}

fn expand(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut out = vec![];
    for p in paths {
        if p.is_dir() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(p)?
                .flatten()
                .map(|e| e.path())
                .filter(|f| {
                    matches!(
                        f.extension().and_then(|e| e.to_str()),
                        Some("jpg" | "jpeg" | "png" | "tif" | "tiff" | "bmp")
                    )
                })
                .collect();
            files.sort();
            out.extend(files);
        } else {
            out.push(p.clone());
        }
    }
    Ok(out)
}
