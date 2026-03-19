use serde::Serialize;

/// Unified error type for all Face3D backend commands.
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("Pipeline error: {0}")]
    Pipeline(String),

    #[error("File not found: {0}")]
    NotFound(String),

    #[error("GPU error: {0}")]
    Gpu(String),

    #[error("Config error: {0}")]
    Config(String),

    #[error("Sensor error: {0}")]
    Sensor(String),

    #[error("Session error: {0}")]
    Session(String),

    #[error("System error: {0}")]
    System(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("YAML error: {0}")]
    Yaml(String),

    #[error("Lock poisoned")]
    #[allow(dead_code)]
    LockPoisoned,
}

// Tauri requires command return errors to be Serialize.
// We serialize them as their Display string.
impl Serialize for AppError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        serializer.serialize_str(self.to_string().as_ref())
    }
}

/// Shorthand result alias used throughout the backend.
pub type AppResult<T> = Result<T, AppError>;
