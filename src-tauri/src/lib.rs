//! Library entry: registers modules and runs the Tauri app.

pub mod cmds;
pub mod detect;
pub mod enhance;
pub mod orient;
pub mod paths;
pub mod phantom;
pub mod pipeline;
pub mod service;

/// Build timestamp (unix seconds) used as the app's version string.
/// The executable's mtime is the honest build time and survives copies;
/// the build.rs stamp is a fallback for exotic layouts.
pub fn build_ts() -> u64 {
    std::env::current_exe()
        .and_then(|p| p.metadata())
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or_else(|| env!("PHOTO_DIGITIZER_BUILD_TS").parse().unwrap_or(0))
}

/// App menu. Mirrors tauri's default macOS menu but replaces the native
/// "About" panel (which shows the static Cargo version) with an item that
/// opens the app's own About dialog, so both entry points agree.
#[cfg(target_os = "macos")]
fn build_menu(app: &tauri::AppHandle) -> tauri::Result<tauri::menu::Menu<tauri::Wry>> {
    use tauri::menu::{Menu, MenuItem, PredefinedMenuItem, Submenu};

    let about = MenuItem::with_id(app, "about", "About Photo Digitizer", true, None::<&str>)?;
    let app_menu = Submenu::with_items(
        app,
        "Photo Digitizer",
        true,
        &[
            &about,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::services(app, None)?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::hide(app, None)?,
            &PredefinedMenuItem::hide_others(app, None)?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::quit(app, None)?,
        ],
    )?;
    let window_menu = Submenu::with_items(
        app,
        "Window",
        true,
        &[
            &PredefinedMenuItem::minimize(app, None)?,
            &PredefinedMenuItem::maximize(app, None)?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::close_window(app, None)?,
        ],
    )?;

    Menu::with_items(
        app,
        &[
            &app_menu,
            &Submenu::with_items(
                app,
                "File",
                true,
                &[&PredefinedMenuItem::close_window(app, None)?],
            )?,
            &Submenu::with_items(
                app,
                "Edit",
                true,
                &[
                    &PredefinedMenuItem::undo(app, None)?,
                    &PredefinedMenuItem::redo(app, None)?,
                    &PredefinedMenuItem::separator(app)?,
                    &PredefinedMenuItem::cut(app, None)?,
                    &PredefinedMenuItem::copy(app, None)?,
                    &PredefinedMenuItem::paste(app, None)?,
                    &PredefinedMenuItem::select_all(app, None)?,
                ],
            )?,
            &Submenu::with_items(
                app,
                "View",
                true,
                &[&PredefinedMenuItem::fullscreen(app, None)?],
            )?,
            &window_menu,
            &Submenu::with_items(app, "Help", true, &[])?,
        ],
    )
}

#[cfg(not(target_os = "macos"))]
fn build_menu(app: &tauri::AppHandle) -> tauri::Result<tauri::menu::Menu<tauri::Wry>> {
    tauri::menu::Menu::default(app)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    paths::clean_jobs(&[]);
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .menu(build_menu)
        .on_menu_event(|app, event| {
            if event.id().as_ref() == "about" {
                use tauri::Emitter;
                let _ = app.emit("show-about", ());
            }
        })
        .manage(cmds::AppState::default())
        .invoke_handler(tauri::generate_handler![
            cmds::list_scans,
            cmds::default_dir,
            cmds::detect_one,
            cmds::detect_all,
            cmds::rotate_photo,
            cmds::re_extract_photo,
            cmds::add_manual_photo,
            cmds::sheet_preview_with_quads,
            cmds::enhance_photos,
            cmds::reset_enhancement,
            cmds::save_photos,
            cmds::cancel_enhance,
            cmds::phantom_status,
            cmds::about_info,
            cmds::pick_folder,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
