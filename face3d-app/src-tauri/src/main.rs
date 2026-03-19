#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use std::{
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};
use tauri::{command, AppHandle, Emitter, Manager, State};

const PROJECT_ROOT: &str = "D:/Projects/3D";
const PYTHON_EXE: &str = "C:/Users/Nicka/miniconda3/envs/face3d/python.exe";
const CUDA_HOME: &str = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8";

// ── Data types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Session {
    id: String,
    name: String,
    has_gaussians: bool,
    has_mesh: bool,
    has_renders: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GpuInfo {
    name: String,
    memory_total: String,
    memory_used: String,
    utilization: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PipelineLog {
    line: String,
    level: String,
}

// ── State ────────────────────────────────────────────────────────

struct PipelineProcess(Arc<Mutex<Option<u32>>>); // Stores PID

// ── Commands ─────────────────────────────────────────────────────

#[command]
async fn list_sessions() -> Result<Vec<Session>, String> {
    let output_dir = Path::new(PROJECT_ROOT).join("data/output");
    let mut sessions = Vec::new();

    if output_dir.exists() && output_dir.is_dir() {
        for entry in fs::read_dir(&output_dir).map_err(|e| e.to_string())? {
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
async fn start_pipeline(
    content_dir: String,
    session: String,
    app: AppHandle,
    state: State<'_, PipelineProcess>,
) -> Result<(), String> {
    // Check if already running
    {
        let guard = state.0.lock().unwrap();
        if guard.is_some() {
            return Err("Pipeline already running".into());
        }
    }

    let script = format!("{}/scripts/run_pipeline.py", PROJECT_ROOT);

    let mut child = Command::new(PYTHON_EXE)
        .arg("-u") // unbuffered output
        .arg(&script)
        .arg("--content-dir")
        .arg(&content_dir)
        .arg("--session")
        .arg(&session)
        .env("CUDA_HOME", CUDA_HOME)
        .env("TORCH_CUDA_ARCH_LIST", "8.6")
        .current_dir(PROJECT_ROOT)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start pipeline: {}", e))?;

    let pid = child.id();
    {
        let mut guard = state.0.lock().unwrap();
        *guard = Some(pid);
    }

    // Stream stdout in a background thread
    let stdout = child.stdout.take().unwrap();
    let app_stdout = app.clone();
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(text) => {
                    let level = if text.contains("[ERROR]") {
                        "error"
                    } else if text.contains("[WARN") {
                        "warn"
                    } else {
                        "info"
                    };
                    let _ = app_stdout.emit("pipeline-log", PipelineLog {
                        line: text,
                        level: level.to_string(),
                    });
                }
                Err(_) => break,
            }
        }
    });

    // Stream stderr in a background thread
    let stderr = child.stderr.take().unwrap();
    let app_stderr = app.clone();
    std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            match line {
                Ok(text) => {
                    let _ = app_stderr.emit("pipeline-log", PipelineLog {
                        line: text,
                        level: "stderr".to_string(),
                    });
                }
                Err(_) => break,
            }
        }
    });

    // Wait for child to finish in background
    let state_clone = state.0.clone();
    let app_done = app.clone();
    std::thread::spawn(move || {
        let status = child.wait();
        let mut guard = state_clone.lock().unwrap();
        *guard = None;
        let exit_code = status.map(|s| s.code().unwrap_or(-1)).unwrap_or(-1);
        let _ = app_done.emit("pipeline-complete", exit_code);
    });

    Ok(())
}

#[command]
async fn stop_pipeline(state: State<'_, PipelineProcess>) -> Result<(), String> {
    let mut guard = state.0.lock().unwrap();
    if let Some(pid) = guard.take() {
        // Kill the process tree on Windows
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F", "/T"])
            .output();
        Ok(())
    } else {
        Err("No pipeline running".into())
    }
}

#[command]
async fn get_gpu_info() -> Result<GpuInfo, String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"])
        .output()
        .map_err(|e| e.to_string())?;

    let text = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = text.trim().split(", ").collect();

    if parts.len() >= 4 {
        Ok(GpuInfo {
            name: parts[0].trim().to_string(),
            memory_total: parts[1].trim().to_string(),
            memory_used: parts[2].trim().to_string(),
            utilization: parts[3].trim().to_string(),
        })
    } else {
        Err("Failed to parse nvidia-smi output".into())
    }
}

#[command]
async fn is_pipeline_running(state: State<'_, PipelineProcess>) -> Result<bool, String> {
    let guard = state.0.lock().unwrap();
    Ok(guard.is_some())
}

#[command]
async fn open_folder(path: String) -> Result<(), String> {
    Command::new("explorer")
        .arg(&path)
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[command]
async fn get_session_files(session_id: String) -> Result<Vec<(String, u64)>, String> {
    let output_dir = Path::new(PROJECT_ROOT)
        .join("data/output")
        .join(&session_id);

    let mut files = Vec::new();
    if output_dir.exists() {
        for entry in fs::read_dir(&output_dir).map_err(|e| e.to_string())? {
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

// ── Main ─────────────────────────────────────────────────────────

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(PipelineProcess(Arc::new(Mutex::new(None))))
        .invoke_handler(tauri::generate_handler![
            list_sessions,
            start_pipeline,
            stop_pipeline,
            get_gpu_info,
            is_pipeline_running,
            open_folder,
            get_session_files,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
