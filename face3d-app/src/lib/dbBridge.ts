/**
 * Database Bridge — connects pipeline events and session actions to the DB.
 *
 * This module listens to pipeline events (stage changes, training metrics,
 * completion) and writes them to the libSQL database in real-time.
 * It also provides DB-backed versions of session operations.
 *
 * Call `initDbBridge()` once at app startup.
 */

import {
  getDb,
  upsertSession,
  updateSessionStatus,
  startPipelineRun,
  endPipelineRun,
  recordStageResult,
  recordTrainingMetric,
  listSessions as dbListSessions,
  deleteSession as dbDeleteSession,
  type SessionRecord,
} from './database';
import {
  listSessions as fsListSessions,
  getSessionFiles,
  type Session,
} from './api';

// ── State ──────────────────────────────────────────────────────

let initialized = false;
let currentRunId: number | null = null;
let currentSessionId: string | null = null;
let stageStartTimes: Record<number, number> = {};

// ── Initialization ─────────────────────────────────────────────

/**
 * Initialize the database and sync existing filesystem sessions into it.
 * Call once at app startup (App.tsx useEffect).
 */
export async function initDbBridge(): Promise<void> {
  if (initialized) return;

  try {
    // Initialize DB (creates tables via migrations)
    await getDb();
    console.log('[DB] Database initialized');

    // Sync filesystem sessions into DB (one-time migration)
    await syncFilesystemSessions();

    initialized = true;
    console.log('[DB] Bridge ready');
  } catch (e) {
    console.error('[DB] Failed to initialize:', e);
    // Non-fatal — app works without DB, just falls back to filesystem
  }
}

/**
 * Scan the filesystem for existing sessions and upsert them into the DB.
 * This ensures the DB has all sessions even if they were created before
 * the DB existed.
 */
async function syncFilesystemSessions(): Promise<void> {
  try {
    const fsSessions = await fsListSessions();
    if (!fsSessions || fsSessions.length === 0) return;

    for (const session of fsSessions) {
      // Calculate total size from session files
      let totalSize = 0;
      try {
        const files = await getSessionFiles(session.id);
        totalSize = files.reduce((sum, [, size]) => sum + size, 0);
      } catch {
        // ignore — file scanning may fail
      }

      await upsertSession({
        id: session.id,
        name: session.name,
        hasGaussians: session.has_gaussians,
        hasMesh: session.has_mesh,
        hasRenderers: session.has_renders,
        totalSizeBytes: totalSize,
      });
    }
    console.log(`[DB] Synced ${fsSessions.length} sessions from filesystem`);
  } catch (e) {
    console.warn('[DB] Filesystem sync failed (non-fatal):', e);
  }
}

// ── Pipeline Event Handlers ────────────────────────────────────

/**
 * Called when a pipeline run starts. Creates a pipeline_run record
 * and updates the session status.
 */
export async function onPipelineStart(
  sessionId: string,
  contentDir: string,
  config?: object
): Promise<void> {
  if (!initialized) return;
  try {
    currentSessionId = sessionId;
    stageStartTimes = {};

    // Ensure session exists in DB
    await upsertSession({
      id: sessionId,
      name: sessionId,
      contentDir,
    });

    // Create pipeline run record
    currentRunId = await startPipelineRun(sessionId, config);
    console.log(`[DB] Pipeline run ${currentRunId} started for ${sessionId}`);
  } catch (e) {
    console.error('[DB] Failed to record pipeline start:', e);
  }
}

/**
 * Called when a pipeline stage starts running.
 */
export async function onStageStart(stageNum: number, stageName: string): Promise<void> {
  if (!initialized || !currentRunId || !currentSessionId) return;
  stageStartTimes[stageNum] = Date.now();
  try {
    await recordStageResult({
      runId: currentRunId,
      sessionId: currentSessionId,
      stageNum,
      stageName,
      status: 'running',
    });
  } catch (e) {
    console.warn('[DB] Failed to record stage start:', e);
  }
}

/**
 * Called when a pipeline stage completes.
 */
export async function onStageComplete(
  stageNum: number,
  stageName: string,
  metrics?: object
): Promise<void> {
  if (!initialized || !currentRunId || !currentSessionId) return;
  const startTime = stageStartTimes[stageNum];
  const duration = startTime ? (Date.now() - startTime) / 1000 : undefined;
  try {
    await recordStageResult({
      runId: currentRunId,
      sessionId: currentSessionId,
      stageNum,
      stageName,
      status: 'complete',
      durationS: duration,
      metrics,
    });
  } catch (e) {
    console.warn('[DB] Failed to record stage completion:', e);
  }
}

/**
 * Called when a training metric is parsed from the log output.
 */
export async function onTrainingMetric(metric: {
  iteration: number;
  loss: number;
  psnr?: number;
  numGaussians?: number;
  gpuMemoryGb?: number;
}): Promise<void> {
  if (!initialized || !currentSessionId) return;
  try {
    await recordTrainingMetric({
      sessionId: currentSessionId,
      runId: currentRunId ?? undefined,
      ...metric,
    });
  } catch {
    // Training metrics are high-frequency — don't spam console on error
  }
}

/**
 * Called when the pipeline finishes (success or failure).
 */
export async function onPipelineComplete(
  exitCode: number,
  durationS?: number
): Promise<void> {
  if (!initialized || !currentRunId || !currentSessionId) return;
  const status = exitCode === 0 ? 'complete' : 'failed';
  try {
    await endPipelineRun(currentRunId, status as any, durationS);
    await updateSessionStatus(
      currentSessionId,
      exitCode === 0 ? 'complete' : 'error'
    );

    // If successful, refresh session metadata
    if (exitCode === 0) {
      try {
        const files = await getSessionFiles(currentSessionId);
        const totalSize = files.reduce((sum, [, size]) => sum + size, 0);
        const hasGaussians = files.some(([n]) => n.includes('gaussians') || n.endsWith('.ply'));
        const hasMesh = files.some(([n]) => n.includes('mesh'));

        await upsertSession({
          id: currentSessionId,
          name: currentSessionId,
          hasGaussians,
          hasMesh,
          totalSizeBytes: totalSize,
        });
      } catch {
        // non-fatal
      }
    }

    console.log(`[DB] Pipeline run ${currentRunId} finished: ${status}`);
  } catch (e) {
    console.error('[DB] Failed to record pipeline completion:', e);
  }

  currentRunId = null;
  currentSessionId = null;
  stageStartTimes = {};
}

// ── Session Operations (DB-backed) ─────────────────────────────

/**
 * List sessions from DB, falling back to filesystem if DB unavailable.
 */
export async function listSessionsFromDb(): Promise<Session[]> {
  if (!initialized) {
    return fsListSessions();
  }
  try {
    const dbSessions = await dbListSessions();
    return dbSessions.map(dbToSession);
  } catch {
    return fsListSessions();
  }
}

/**
 * Delete session from both DB and filesystem.
 */
export async function deleteSessionFromDb(id: string): Promise<void> {
  try {
    await dbDeleteSession(id);
  } catch {
    // DB delete may fail if session not in DB — that's OK
  }
}

// ── Helpers ────────────────────────────────────────────────────

function dbToSession(record: SessionRecord): Session {
  return {
    id: record.id,
    name: record.name,
    has_gaussians: record.has_gaussians === 1,
    has_mesh: record.has_mesh === 1,
    has_renders: record.has_renders === 1,
    // Extended fields from DB
    quality_grade: record.quality_grade ?? undefined,
    quality_score: record.quality_score ?? undefined,
    total_size_bytes: record.total_size_bytes ?? undefined,
    status: record.status ?? undefined,
    created_at: record.created_at ?? undefined,
  };
}

/**
 * Get rich session summary from DB (stages, training metrics, frame quality).
 */
export async function getSessionSummaryFromDb(sessionId: string): Promise<any | null> {
  if (!initialized) return null;
  try {
    const { getSessionSummary } = await import('./database');
    return await getSessionSummary(sessionId);
  } catch {
    return null;
  }
}

/**
 * Get training metrics for charting from DB.
 */
export async function getTrainingMetricsFromDb(sessionId: string): Promise<any[]> {
  if (!initialized) return [];
  try {
    const { getTrainingMetrics } = await import('./database');
    return await getTrainingMetrics(sessionId);
  } catch {
    return [];
  }
}

/**
 * Compare quality across all sessions.
 */
export async function compareSessionsFromDb(): Promise<any[]> {
  if (!initialized) return [];
  try {
    const { compareSessionQuality } = await import('./database');
    return await compareSessionQuality();
  } catch {
    return [];
  }
}
