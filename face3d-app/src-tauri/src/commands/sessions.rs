use std::fs;
use std::path::Path;

use tauri::command;

use crate::error::{AppError, AppResult};
use crate::state::{Session, PROJECT_ROOT};

#[command]
pub async fn list_sessions() -> AppResult<Vec<Session>> {
    let output_dir = Path::new(PROJECT_ROOT).join("data/output");
    let mut sessions = Vec::new();

    if output_dir.exists() && output_dir.is_dir() {
        let entries = fs::read_dir(&output_dir)?;
        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            sessions.push(Session {
                id: name.clone(),
                name,
                has_gaussians: path.join("gaussians.ply").exists(),
                has_mesh: path.join("mesh.ply").exists() || path.join("mesh.obj").exists(),
                has_renders: path.join("renders").exists(),
            });
        }
    }

    sessions.sort_by(|a, b| b.id.cmp(&a.id)); // newest first
    Ok(sessions)
}

#[command]
pub async fn get_session_files(session_id: String) -> AppResult<Vec<(String, u64)>> {
    let output_dir = Path::new(PROJECT_ROOT)
        .join("data/output")
        .join(&session_id);

    let mut files = Vec::new();
    if output_dir.exists() {
        let entries = fs::read_dir(&output_dir)?;
        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = entry.path();
            if path.is_file() {
                let name = entry.file_name().to_string_lossy().into_owned();
                let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                files.push((name, size));
            }
        }
    }

    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

#[command]
pub async fn delete_session(session_id: String) -> AppResult<()> {
    // Validate that the session_id doesn't contain path traversal
    if session_id.contains("..") || session_id.contains('/') || session_id.contains('\\') {
        return Err(AppError::Session("Invalid session ID".into()));
    }

    let output_dir = Path::new(PROJECT_ROOT)
        .join("data/output")
        .join(&session_id);
    let processed_dir = Path::new(PROJECT_ROOT)
        .join("data/processed")
        .join(&session_id);

    let mut deleted_something = false;

    if output_dir.exists() {
        fs::remove_dir_all(&output_dir)
            .map_err(|e| AppError::Session(format!("Failed to delete output dir: {}", e)))?;
        deleted_something = true;
    }

    if processed_dir.exists() {
        fs::remove_dir_all(&processed_dir)
            .map_err(|e| AppError::Session(format!("Failed to delete processed dir: {}", e)))?;
        deleted_something = true;
    }

    if deleted_something {
        Ok(())
    } else {
        Err(AppError::NotFound(format!(
            "Session '{}' not found",
            session_id
        )))
    }
}

#[command]
pub async fn get_session_metrics(session_id: String) -> AppResult<String> {
    let metrics_path = Path::new(PROJECT_ROOT)
        .join("data/output")
        .join(&session_id)
        .join("metrics.jsonl");

    if metrics_path.exists() && metrics_path.is_file() {
        let content = fs::read_to_string(&metrics_path)?;
        // Convert JSONL to a JSON array string
        let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
        let json_array = format!("[{}]", lines.join(","));
        Ok(json_array)
    } else {
        Ok(String::new())
    }
}

#[command]
pub async fn get_image_counts(session_id: String) -> AppResult<(usize, usize, usize)> {
    let renders_dir = Path::new(PROJECT_ROOT)
        .join("data/output")
        .join(&session_id)
        .join("renders");
    let previews_dir = Path::new(PROJECT_ROOT)
        .join("data/output")
        .join(&session_id)
        .join("previews");
    let srgb_dir = Path::new(PROJECT_ROOT)
        .join("data/processed")
        .join(&session_id)
        .join("frames_srgb");
    let frames_dir = Path::new(PROJECT_ROOT)
        .join("data/processed")
        .join(&session_id)
        .join("frames");

    let render_count = count_image_files(&renders_dir);
    let preview_count = count_image_files(&previews_dir);
    let frames = count_image_files(&srgb_dir);
    let frame_count = if frames == 0 {
        count_image_files(&frames_dir)
    } else {
        frames
    };

    Ok((render_count, preview_count, frame_count))
}

/// Count image files (png, jpg, jpeg) in a directory.
fn count_image_files(dir: &Path) -> usize {
    if !dir.exists() || !dir.is_dir() {
        return 0;
    }
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    entries
        .flatten()
        .filter(|e| {
            let path = e.path();
            if !path.is_file() {
                return false;
            }
            match path.extension().and_then(|ext| ext.to_str()) {
                Some(ext) => {
                    let lower = ext.to_lowercase();
                    lower == "png" || lower == "jpg" || lower == "jpeg"
                }
                None => false,
            }
        })
        .count()
}
