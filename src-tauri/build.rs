fn main() {
    // Stamp the build time (unix seconds) so the binary carries its own
    // build timestamp as the app "version".
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    println!("cargo:rustc-env=PHOTO_DIGITIZER_BUILD_TS={ts}");
    tauri_build::build()
}
