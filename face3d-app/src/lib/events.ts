/**
 * Typed event subscription manager for Tauri backend events.
 *
 * Centralises all listen() calls, guarantees cleanup on unmount,
 * and falls back to mock event timers in dev mode (browser).
 */

import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { isTauri, type PipelineLogPayload } from "./api";

// ── Event payload types ─────────────────────────────────────────

export type { PipelineLogPayload };

export interface StageChangePayload {
  stage: number;
  name: string;
}

export interface TrainingMetricPayload {
  iter: number;
  loss?: number;
  psnr?: number;
  gaussians?: number;
}

export interface FileChangePayload {
  path: string;
  kind: "created" | "modified" | "removed";
}

// ── Event name constants ────────────────────────────────────────

export const EVENTS = {
  PIPELINE_LOG: "pipeline-log",
  PIPELINE_COMPLETE: "pipeline-complete",
  PIPELINE_STAGE: "pipeline-stage",
  TRAINING_METRIC: "training-metric",
  FILE_CHANGE: "file-change",
} as const;

// ── Subscription record ─────────────────────────────────────────

interface Subscription {
  event: string;
  unlisten: UnlistenFn;
}

// ── Event Manager ───────────────────────────────────────────────

class EventManager {
  private subscriptions: Subscription[] = [];
  private devTimers: ReturnType<typeof setInterval>[] = [];

  /**
   * Subscribe to a typed backend event.
   * Returns a cleanup function (also tracked internally for bulk cleanup).
   */
  async subscribe<T>(
    event: string,
    handler: (payload: T) => void,
  ): Promise<() => void> {
    if (!isTauri()) {
      // In dev mode, we just return a no-op. Dev mock events are
      // set up separately via startDevMockEvents().
      return () => {};
    }

    const unlisten = await listen<T>(event, (e) => handler(e.payload));
    const sub: Subscription = { event, unlisten };
    this.subscriptions.push(sub);

    return () => {
      unlisten();
      this.subscriptions = this.subscriptions.filter((s) => s !== sub);
    };
  }

  // ── Typed convenience methods ─────────────────────────────────

  onPipelineLog(
    handler: (payload: PipelineLogPayload) => void,
  ): Promise<() => void> {
    return this.subscribe<PipelineLogPayload>(EVENTS.PIPELINE_LOG, handler);
  }

  onPipelineComplete(
    handler: (exitCode: number) => void,
  ): Promise<() => void> {
    return this.subscribe<number>(EVENTS.PIPELINE_COMPLETE, handler);
  }

  onPipelineStageChange(
    handler: (payload: StageChangePayload) => void,
  ): Promise<() => void> {
    return this.subscribe<StageChangePayload>(EVENTS.PIPELINE_STAGE, handler);
  }

  onTrainingMetric(
    handler: (payload: TrainingMetricPayload) => void,
  ): Promise<() => void> {
    return this.subscribe<TrainingMetricPayload>(EVENTS.TRAINING_METRIC, handler);
  }

  onFileChange(
    handler: (payload: FileChangePayload) => void,
  ): Promise<() => void> {
    return this.subscribe<FileChangePayload>(EVENTS.FILE_CHANGE, handler);
  }

  // ── Bulk cleanup ──────────────────────────────────────────────

  /** Unsubscribe from all events. Call on component unmount or app shutdown. */
  unsubscribeAll(): void {
    for (const sub of this.subscriptions) {
      sub.unlisten();
    }
    this.subscriptions = [];
    this.stopDevMockEvents();
  }

  // ── Dev-mode mock event emitters ──────────────────────────────

  /**
   * Start emitting fake pipeline events for UI development.
   * Only activates outside Tauri (plain browser).
   */
  startDevMockEvents(handlers: {
    onLog?: (payload: PipelineLogPayload) => void;
    onComplete?: (exitCode: number) => void;
  }): void {
    if (isTauri()) return;

    this.stopDevMockEvents();

    const stages = [
      "Frame Extraction",
      "Color Correction",
      "Quality Filtering",
      "COLMAP SfM",
      "Depth Estimation",
      "Gaussian Training",
    ];

    let tick = 0;
    let stageIdx = 0;

    const timer = setInterval(() => {
      tick++;

      if (handlers.onLog) {
        if (tick % 8 === 0 && stageIdx < stages.length) {
          handlers.onLog({
            line: `=== Stage ${stageIdx + 1}: ${stages[stageIdx]} ===`,
            level: "info",
          });
          stageIdx++;
        } else {
          handlers.onLog({
            line: `[mock] Processing frame ${tick}...`,
            level: "info",
          });
        }
      }

      // Complete after ~30 ticks
      if (tick >= 30) {
        handlers.onLog?.({ line: "Pipeline complete", level: "info" });
        handlers.onComplete?.(0);
        this.stopDevMockEvents();
      }
    }, 500);

    this.devTimers.push(timer);
  }

  stopDevMockEvents(): void {
    for (const t of this.devTimers) {
      clearInterval(t);
    }
    this.devTimers = [];
  }
}

/** Singleton event manager. */
export const eventManager = new EventManager();

export default EventManager;
