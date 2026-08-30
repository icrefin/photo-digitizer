//! Library entry: registers modules and runs the Tauri app.

pub mod cmds;
pub mod detect;
pub mod enhance;
pub mod orient;
pub mod paths;
pub mod pipeline;
pub mod service;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    paths::clean_jobs();
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(cmds::AppState::default())
        .invoke_handler(tauri::generate_handler![
            cmds::list_scans,
            cmds::default_dir,
            cmds::detect_one,
            cmds::detect_all,
            cmds::rotate_photo,
            cmds::re_extract_photo,
            cmds::sheet_preview_with_quads,
            cmds::enhance_photos,
            cmds::save_photos,
            cmds::cancel_enhance,
            cmds::pick_folder,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
