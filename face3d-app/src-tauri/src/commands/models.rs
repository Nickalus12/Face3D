use std::fs;
use std::path::Path;

use tauri::command;

use crate::error::{AppError, AppResult};
use crate::state::{PreviewFile, PROJECT_ROOT};

#[command]
pub async fn get_model_path(session_id: String) -> AppResult<String> {
    let output_dir = Path::new(PROJECT_ROOT)
        .join("data/output")
        .join(&session_id);

    // Prefer gaussians.ply, fall back to mesh.ply, then mesh.obj
    let candidates = ["gaussians.ply", "mesh.ply", "mesh.obj"];
    for name in &candidates {
        let path = output_dir.join(name);
        if path.exists() && path.is_file() {
            return Ok(path.to_string_lossy().into_owned());
        }
    }

    Err(AppError::NotFound(format!(
        "No model file found for session '{}'",
        session_id
    )))
}

#[command]
pub async fn list_renders(session_id: String) -> AppResult<Vec<String>> {
    let renders_dir = Path::new(PROJECT_ROOT)
        .join("data/output")
        .join(&session_id)
        .join("renders");
    Ok(list_image_files(&renders_dir))
}

#[command]
pub async fn list_previews(session_id: String) -> AppResult<Vec<PreviewFile>> {
    let previews_dir = Path::new(PROJECT_ROOT)
        .join("data/output")
        .join(&session_id)
        .join("previews");

    let mut result = Vec::new();
    if !previews_dir.exists() || !previews_dir.is_dir() {
        return Ok(result);
    }

    let entries = match fs::read_dir(&previews_dir) {
        Ok(e) => e,
        Err(_) => return Ok(result),
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if !is_image_extension(&path) {
            continue;
        }

        let name = entry.file_name().to_string_lossy().into_owned();
        let name_lower = name.to_lowercase();

        let preview_type = if name_lower.contains("depth") {
            "depth"
        } else if name_lower.contains("landmark") {
            "landmark"
        } else if name_lower.contains("mask") {
            "mask"
        } else if name_lower.contains("comparison") || name_lower.contains("compare") {
            "comparison"
        } else {
            "other"
        };

        result.push(PreviewFile {
            name,
            preview_type: preview_type.to_string(),
            path: path.to_string_lossy().into_owned(),
        });
    }

    // Also scan face_masks/ directory in processed/
    let masks_dir = Path::new(PROJECT_ROOT)
        .join("data/processed")
        .join(&session_id)
        .join("face_masks");
    if masks_dir.exists() && masks_dir.is_dir() {
        if let Ok(entries) = fs::read_dir(&masks_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && is_image_extension(&path) {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    result.push(PreviewFile {
                        name,
                        preview_type: "mask".to_string(),
                        path: path.to_string_lossy().into_owned(),
                    });
                }
            }
        }
    }

    // Also scan comparisons/ subdirectory in previews
    let comparisons_dir = previews_dir.join("comparisons");
    if comparisons_dir.exists() && comparisons_dir.is_dir() {
        if let Ok(entries) = fs::read_dir(&comparisons_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && is_image_extension(&path) {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    result.push(PreviewFile {
                        name,
                        preview_type: "comparison".to_string(),
                        path: path.to_string_lossy().into_owned(),
                    });
                }
            }
        }
    }

    result.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(result)
}

#[command]
pub async fn list_input_frames(session_id: String) -> AppResult<Vec<String>> {
    // Try frames_srgb first, fall back to frames
    let srgb_dir = Path::new(PROJECT_ROOT)
        .join("data/processed")
        .join(&session_id)
        .join("frames_srgb");

    if srgb_dir.exists() && srgb_dir.is_dir() {
        let files = list_image_files(&srgb_dir);
        if !files.is_empty() {
            return Ok(files);
        }
    }

    let frames_dir = Path::new(PROJECT_ROOT)
        .join("data/processed")
        .join(&session_id)
        .join("frames");
    Ok(list_image_files(&frames_dir))
}

#[command]
pub async fn get_report_path(session_id: String) -> AppResult<Option<String>> {
    let report = Path::new(PROJECT_ROOT)
        .join("data/output")
        .join(&session_id)
        .join("report.html");

    if report.exists() && report.is_file() {
        Ok(Some(report.to_string_lossy().into_owned()))
    } else {
        Ok(None)
    }
}

// ── Helpers ──────────────────────────────────────────────────────

fn is_image_extension(path: &Path) -> bool {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) => {
            let lower = ext.to_lowercase();
            lower == "png" || lower == "jpg" || lower == "jpeg"
        }
        None => false,
    }
}

fn list_image_files(dir: &Path) -> Vec<String> {
    let mut files = Vec::new();
    if !dir.exists() || !dir.is_dir() {
        return files;
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && is_image_extension(&path) {
                files.push(path.to_string_lossy().into_owned());
            }
        }
    }
    files.sort();
    files
}
