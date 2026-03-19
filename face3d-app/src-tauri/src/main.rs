#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod commands;
mod error;
mod state;

use state::{GpuCacheState, PipelineProcess};

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(PipelineProcess::new())
        .manage(GpuCacheState::new())
        .invoke_handler(tauri::generate_handler![
            // Pipeline
            commands::pipeline::start_pipeline,
            commands::pipeline::stop_pipeline,
            commands::pipeline::is_pipeline_running,
            // Sessions
            commands::sessions::list_sessions,
            commands::sessions::get_session_files,
            commands::sessions::delete_session,
            commands::sessions::get_session_metrics,
            commands::sessions::get_image_counts,
            // GPU
            commands::gpu::get_gpu_info,
            // Sensors
            commands::sensors::get_sensor_summary,
            commands::sensors::get_sensor_data,
            // Models & gallery
            commands::models::get_model_path,
            commands::models::list_renders,
            commands::models::list_previews,
            commands::models::list_input_frames,
            commands::models::get_report_path,
            // Config & system
            commands::config::get_pipeline_config,
            commands::config::save_pipeline_config,
            commands::config::get_python_info,
            commands::config::get_system_info,
            commands::config::open_folder,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
