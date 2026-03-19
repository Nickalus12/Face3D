/**
 * Sets up all pipeline event listeners (log, complete) and wires
 * them into the Zustand stores. Handles cleanup on unmount.
 *
 * Usage: call once in your root component (App.tsx).
 */

import { useEffect } from "react";
import { eventManager } from "../lib/events";
import { invalidateSessionCaches, isTauri } from "../lib/api";
import usePipelineStore from "../store/pipelineStore";
import useSessionStore from "../store/sessionStore";
import {
  initDbBridge,
  onPipelineStart,
  onStageComplete,
  onTrainingMetric,
  onPipelineComplete as dbOnPipelineComplete,
} from "../lib/dbBridge";

export function usePipelineEvents(): void {
  useEffect(() => {
    const cleanups: Array<() => void> = [];

    async function setup() {
      // Initialize database on app startup
      await initDbBridge();

      // Pipeline log -> parse into store + record to DB
      const unLog = await eventManager.onPipelineLog((payload) => {
        const pipelineState = usePipelineStore.getState();
        pipelineState.parseLogLine(payload.line, payload.level);

        // Record stage completions to DB
        const stageCompleteMatch = payload.line.match(/Stage\s+(\d+)\s+completed\s+in\s+([\d.]+)s/i);
        if (stageCompleteMatch) {
          const stageNum = parseInt(stageCompleteMatch[1], 10);
          const stage = pipelineState.stages.find(s => s.id === stageNum);
          onStageComplete(stageNum, stage?.name ?? `Stage ${stageNum}`);
        }

        // Record training metrics to DB
        const iterMatch = payload.line.match(/Iter\s+(\d+)/i);
        const lossMatch = payload.line.match(/Loss\s+([\d.]+)/i);
        const psnrMatch = payload.line.match(/PSNR\s+([\d.]+)/i);
        const gsMatch = payload.line.match(/GS\s+([\d,]+)/i);
        const gpuMatch = payload.line.match(/GPU\s+([\d.]+)\//i);

        if (iterMatch && lossMatch) {
          onTrainingMetric({
            iteration: parseInt(iterMatch[1], 10),
            loss: parseFloat(lossMatch[1]),
            psnr: psnrMatch ? parseFloat(psnrMatch[1]) : undefined,
            numGaussians: gsMatch ? parseInt(gsMatch[1].replace(/,/g, ''), 10) : undefined,
            gpuMemoryGb: gpuMatch ? parseFloat(gpuMatch[1]) : undefined,
          });
        }
      });
      cleanups.push(unLog);

      // Pipeline complete -> update stores, DB, invalidate caches
      const unComplete = await eventManager.onPipelineComplete((exitCode) => {
        const state = usePipelineStore.getState();
        const durationS = state.startedAt
          ? (Date.now() - state.startedAt) / 1000
          : undefined;

        state.onPipelineComplete(exitCode);

        // Record to DB
        dbOnPipelineComplete(exitCode, durationS);

        invalidateSessionCaches();
        useSessionStore.getState().fetchSessions();

        // Flash the window title briefly
        if (document.title) {
          const original = document.title;
          document.title = exitCode === 0 ? "Pipeline Complete!" : "Pipeline Failed";
          setTimeout(() => {
            document.title = original;
          }, 3000);
        }
      });
      cleanups.push(unComplete);
    }

    setup();

    // In dev mode (browser), start mock events if someone starts
    // a pipeline via the wizard. The mock events are self-stopping.
    if (!isTauri()) {
      // Dev mock events are opt-in via eventManager.startDevMockEvents()
      // — they are triggered from pipelineStore.startPipeline in dev mode.
    }

    return () => {
      for (const fn of cleanups) fn();
    };
  }, []);
}

export default usePipelineEvents;
