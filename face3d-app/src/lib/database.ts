/**
 * Face3D Database Layer — powered by libSQL (Turso)
 *
 * Replaces filesystem scanning with a real database. Provides:
 * - Session CRUD with rich metadata
 * - Training metrics time-series storage
 * - Pipeline stage tracking
 * - Frame-level quality data
 * - Vector search for ML prior (FLAME shape embeddings)
 * - Quality scoring and comparison queries
 */

import { Database } from 'tauri-plugin-libsql-api';

// ── Singleton database instance ────────────────────────────────

let db: Database | null = null;

export async function getDb(): Promise<Database> {
  if (db) return db;
  db = await Database.load('sqlite:face3d.db');
  await runMigrations(db);
  return db;
}

export async function closeDb(): Promise<void> {
  if (db) {
    await db.close();
    db = null;
  }
}

// ── Schema / Migrations ────────────────────────────────────────

async function runMigrations(database: Database): Promise<void> {
  // Migration version tracking
  await database.execute(`
    CREATE TABLE IF NOT EXISTS _migrations (
      id INTEGER PRIMARY KEY,
      name TEXT NOT NULL,
      applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  const applied = await database.select<{ name: string }[]>(
    'SELECT name FROM _migrations ORDER BY id'
  );
  const appliedNames = new Set(applied.map(r => r.name));

  for (const migration of MIGRATIONS) {
    if (appliedNames.has(migration.name)) continue;
    console.log(`[DB] Running migration: ${migration.name}`);
    for (const stmt of migration.statements) {
      await database.execute(stmt);
    }
    await database.execute(
      'INSERT INTO _migrations (name) VALUES ($1)',
      [migration.name]
    );
  }
}

interface Migration {
  name: string;
  statements: string[];
}

const MIGRATIONS: Migration[] = [
  {
    name: '001_sessions',
    statements: [
      `CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        status TEXT NOT NULL DEFAULT 'pending',
        content_dir TEXT,
        video_paths TEXT,
        photo_count INTEGER DEFAULT 0,
        sensor_log_count INTEGER DEFAULT 0,
        has_gaussians INTEGER DEFAULT 0,
        has_mesh INTEGER DEFAULT 0,
        has_renders INTEGER DEFAULT 0,
        total_size_bytes INTEGER DEFAULT 0,
        quality_grade TEXT,
        quality_score REAL,
        notes TEXT
      )`,
    ],
  },
  {
    name: '002_pipeline_runs',
    statements: [
      `CREATE TABLE IF NOT EXISTS pipeline_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        finished_at DATETIME,
        status TEXT NOT NULL DEFAULT 'running',
        total_duration_s REAL,
        config_snapshot TEXT,
        error_message TEXT
      )`,
      `CREATE INDEX IF NOT EXISTS idx_pipeline_runs_session ON pipeline_runs(session_id)`,
    ],
  },
  {
    name: '003_stage_results',
    statements: [
      `CREATE TABLE IF NOT EXISTS stage_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
        session_id TEXT NOT NULL,
        stage_num INTEGER NOT NULL,
        stage_name TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        started_at DATETIME,
        finished_at DATETIME,
        duration_s REAL,
        gpu_peak_gb REAL,
        items_processed INTEGER,
        error_message TEXT,
        metrics_json TEXT
      )`,
      `CREATE INDEX IF NOT EXISTS idx_stage_results_run ON stage_results(run_id)`,
      `CREATE INDEX IF NOT EXISTS idx_stage_results_session ON stage_results(session_id, stage_num)`,
    ],
  },
  {
    name: '004_training_metrics',
    statements: [
      `CREATE TABLE IF NOT EXISTS training_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        run_id INTEGER,
        iteration INTEGER NOT NULL,
        loss REAL NOT NULL,
        psnr REAL,
        num_gaussians INTEGER,
        gpu_memory_gb REAL,
        lr_means REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
      )`,
      `CREATE INDEX IF NOT EXISTS idx_training_session ON training_metrics(session_id)`,
      `CREATE INDEX IF NOT EXISTS idx_training_iter ON training_metrics(session_id, iteration)`,
    ],
  },
  {
    name: '005_frame_quality',
    statements: [
      `CREATE TABLE IF NOT EXISTS frame_quality (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        frame_name TEXT NOT NULL,
        blur_score REAL,
        exposure_score REAL,
        face_confidence REAL,
        face_area_ratio REAL,
        selected INTEGER DEFAULT 1,
        rejection_reason TEXT
      )`,
      `CREATE INDEX IF NOT EXISTS idx_frame_quality_session ON frame_quality(session_id)`,
    ],
  },
  {
    name: '006_scan_embeddings',
    statements: [
      // FLAME shape embeddings for vector similarity search (ML prior)
      `CREATE TABLE IF NOT EXISTS scan_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        embedding_type TEXT NOT NULL DEFAULT 'flame_shape',
        embedding BLOB,
        metadata_json TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )`,
      `CREATE INDEX IF NOT EXISTS idx_embeddings_session ON scan_embeddings(session_id)`,
      // Vector index for similarity search (cosine distance)
      // NOTE: libsql_vector_idx requires the vector extension enabled
      // This will be created when the first embedding is inserted
    ],
  },
];

// ── Session Operations ─────────────────────────────────────────

export interface SessionRecord {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  status: string;
  content_dir: string | null;
  photo_count: number;
  sensor_log_count: number;
  has_gaussians: number;
  has_mesh: number;
  has_renders: number;
  total_size_bytes: number;
  quality_grade: string | null;
  quality_score: number | null;
  notes: string | null;
}

export async function upsertSession(session: {
  id: string;
  name: string;
  contentDir?: string;
  photoCount?: number;
  sensorLogCount?: number;
  hasGaussians?: boolean;
  hasMesh?: boolean;
  hasRenderers?: boolean;
  totalSizeBytes?: number;
  qualityGrade?: string;
  qualityScore?: number;
}): Promise<void> {
  const database = await getDb();
  await database.execute(
    `INSERT INTO sessions (id, name, content_dir, photo_count, sensor_log_count, has_gaussians, has_mesh, has_renders, total_size_bytes, quality_grade, quality_score)
     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
     ON CONFLICT(id) DO UPDATE SET
       name = $2,
       updated_at = CURRENT_TIMESTAMP,
       photo_count = COALESCE($4, photo_count),
       sensor_log_count = COALESCE($5, sensor_log_count),
       has_gaussians = COALESCE($6, has_gaussians),
       has_mesh = COALESCE($7, has_mesh),
       has_renders = COALESCE($8, has_renders),
       total_size_bytes = COALESCE($9, total_size_bytes),
       quality_grade = COALESCE($10, quality_grade),
       quality_score = COALESCE($11, quality_score)`,
    [
      session.id,
      session.name,
      session.contentDir ?? null,
      session.photoCount ?? 0,
      session.sensorLogCount ?? 0,
      session.hasGaussians ? 1 : 0,
      session.hasMesh ? 1 : 0,
      session.hasRenderers ? 1 : 0,
      session.totalSizeBytes ?? 0,
      session.qualityGrade ?? null,
      session.qualityScore ?? null,
    ]
  );
}

export async function listSessions(): Promise<SessionRecord[]> {
  const database = await getDb();
  return database.select<SessionRecord[]>(
    'SELECT * FROM sessions ORDER BY updated_at DESC'
  );
}

export async function getSession(id: string): Promise<SessionRecord | null> {
  const database = await getDb();
  const rows = await database.select<SessionRecord[]>(
    'SELECT * FROM sessions WHERE id = $1',
    [id]
  );
  return rows.length > 0 ? rows[0] : null;
}

export async function deleteSession(id: string): Promise<void> {
  const database = await getDb();
  await database.execute('DELETE FROM sessions WHERE id = $1', [id]);
}

export async function updateSessionStatus(
  id: string,
  status: string
): Promise<void> {
  const database = await getDb();
  await database.execute(
    'UPDATE sessions SET status = $1, updated_at = CURRENT_TIMESTAMP WHERE id = $2',
    [status, id]
  );
}

// ── Pipeline Run Operations ────────────────────────────────────

export async function startPipelineRun(
  sessionId: string,
  config?: object
): Promise<number> {
  const database = await getDb();
  const result = await database.execute(
    `INSERT INTO pipeline_runs (session_id, config_snapshot) VALUES ($1, $2)`,
    [sessionId, config ? JSON.stringify(config) : null]
  );
  await updateSessionStatus(sessionId, 'running');
  return result.lastInsertId;
}

export async function endPipelineRun(
  runId: number,
  status: 'complete' | 'failed' | 'cancelled',
  duration?: number,
  error?: string
): Promise<void> {
  const database = await getDb();
  await database.execute(
    `UPDATE pipeline_runs SET
      status = $1,
      finished_at = CURRENT_TIMESTAMP,
      total_duration_s = $2,
      error_message = $3
    WHERE id = $4`,
    [status, duration ?? null, error ?? null, runId]
  );
}

// ── Stage Result Operations ────────────────────────────────────

export async function recordStageResult(stage: {
  runId: number;
  sessionId: string;
  stageNum: number;
  stageName: string;
  status: string;
  durationS?: number;
  gpuPeakGb?: number;
  itemsProcessed?: number;
  error?: string;
  metrics?: object;
}): Promise<void> {
  const database = await getDb();
  await database.execute(
    `INSERT INTO stage_results (run_id, session_id, stage_num, stage_name, status, duration_s, gpu_peak_gb, items_processed, error_message, metrics_json, started_at, finished_at)
     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)`,
    [
      stage.runId,
      stage.sessionId,
      stage.stageNum,
      stage.stageName,
      stage.status,
      stage.durationS ?? null,
      stage.gpuPeakGb ?? null,
      stage.itemsProcessed ?? null,
      stage.error ?? null,
      stage.metrics ? JSON.stringify(stage.metrics) : null,
    ]
  );
}

export async function getStageResults(
  sessionId: string
): Promise<any[]> {
  const database = await getDb();
  return database.select(
    `SELECT * FROM stage_results WHERE session_id = $1 ORDER BY stage_num`,
    [sessionId]
  );
}

// ── Training Metrics Operations ────────────────────────────────

export async function recordTrainingMetric(metric: {
  sessionId: string;
  runId?: number;
  iteration: number;
  loss: number;
  psnr?: number;
  numGaussians?: number;
  gpuMemoryGb?: number;
  lrMeans?: number;
}): Promise<void> {
  const database = await getDb();
  await database.execute(
    `INSERT INTO training_metrics (session_id, run_id, iteration, loss, psnr, num_gaussians, gpu_memory_gb, lr_means)
     VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`,
    [
      metric.sessionId,
      metric.runId ?? null,
      metric.iteration,
      metric.loss,
      metric.psnr ?? null,
      metric.numGaussians ?? null,
      metric.gpuMemoryGb ?? null,
      metric.lrMeans ?? null,
    ]
  );
}

export async function getTrainingMetrics(
  sessionId: string
): Promise<any[]> {
  const database = await getDb();
  return database.select(
    `SELECT iteration, loss, psnr, num_gaussians, gpu_memory_gb, timestamp
     FROM training_metrics WHERE session_id = $1 ORDER BY iteration`,
    [sessionId]
  );
}

export async function getLatestTrainingMetric(
  sessionId: string
): Promise<any | null> {
  const database = await getDb();
  const rows = await database.select<any[]>(
    `SELECT * FROM training_metrics WHERE session_id = $1 ORDER BY iteration DESC LIMIT 1`,
    [sessionId]
  );
  return rows.length > 0 ? rows[0] : null;
}

// ── Frame Quality Operations ───────────────────────────────────

export async function recordFrameQuality(frames: {
  sessionId: string;
  frameName: string;
  blurScore?: number;
  exposureScore?: number;
  faceConfidence?: number;
  faceAreaRatio?: number;
  selected?: boolean;
  rejectionReason?: string;
}[]): Promise<void> {
  const database = await getDb();
  for (const frame of frames) {
    await database.execute(
      `INSERT OR REPLACE INTO frame_quality (session_id, frame_name, blur_score, exposure_score, face_confidence, face_area_ratio, selected, rejection_reason)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`,
      [
        frame.sessionId,
        frame.frameName,
        frame.blurScore ?? null,
        frame.exposureScore ?? null,
        frame.faceConfidence ?? null,
        frame.faceAreaRatio ?? null,
        frame.selected ? 1 : 0,
        frame.rejectionReason ?? null,
      ]
    );
  }
}

export async function getFrameQuality(sessionId: string): Promise<any[]> {
  const database = await getDb();
  return database.select(
    `SELECT * FROM frame_quality WHERE session_id = $1 ORDER BY frame_name`,
    [sessionId]
  );
}

// ── Vector Search (ML Prior) ───────────────────────────────────

export async function storeScanEmbedding(
  sessionId: string,
  embedding: number[],
  type: string = 'flame_shape',
  metadata?: object
): Promise<void> {
  const database = await getDb();
  // Store as vector32 format for libSQL vector search
  const vecStr = `[${embedding.join(',')}]`;
  await database.execute(
    `INSERT INTO scan_embeddings (session_id, embedding_type, embedding, metadata_json)
     VALUES ($1, $2, vector32($3), $4)`,
    [sessionId, type, vecStr, metadata ? JSON.stringify(metadata) : null]
  );
}

export async function findSimilarScans(
  queryEmbedding: number[],
  limit: number = 5
): Promise<{ session_id: string; distance: number }[]> {
  const database = await getDb();
  const vecStr = `[${queryEmbedding.join(',')}]`;
  try {
    // Use vector_distance_cos for cosine similarity search
    return await database.select(
      `SELECT session_id,
              vector_distance_cos(embedding, vector32($1)) as distance
       FROM scan_embeddings
       WHERE embedding_type = 'flame_shape'
       ORDER BY distance
       LIMIT $2`,
      [vecStr, limit]
    );
  } catch {
    // Vector extension may not be available — fall back gracefully
    console.warn('[DB] Vector search not available, returning empty results');
    return [];
  }
}

// ── Analytics Queries ──────────────────────────────────────────

export async function getSessionSummary(sessionId: string): Promise<any> {
  const database = await getDb();
  const [session] = await database.select<any[]>(
    'SELECT * FROM sessions WHERE id = $1', [sessionId]
  );
  const stageResults = await getStageResults(sessionId);
  const latestMetric = await getLatestTrainingMetric(sessionId);
  const frameStats = await database.select<any[]>(
    `SELECT
       COUNT(*) as total_frames,
       SUM(CASE WHEN selected = 1 THEN 1 ELSE 0 END) as selected_frames,
       AVG(blur_score) as avg_blur,
       AVG(face_confidence) as avg_face_conf
     FROM frame_quality WHERE session_id = $1`,
    [sessionId]
  );

  return {
    session,
    stages: stageResults,
    training: latestMetric,
    frameStats: frameStats[0] ?? null,
  };
}

export async function compareSessionQuality(): Promise<any[]> {
  const database = await getDb();
  return database.select(
    `SELECT s.id, s.name, s.quality_grade, s.quality_score,
            s.created_at, s.total_size_bytes,
            (SELECT MAX(psnr) FROM training_metrics WHERE session_id = s.id) as best_psnr,
            (SELECT COUNT(*) FROM frame_quality WHERE session_id = s.id AND selected = 1) as frame_count
     FROM sessions s
     ORDER BY s.quality_score DESC NULLS LAST`
  );
}
