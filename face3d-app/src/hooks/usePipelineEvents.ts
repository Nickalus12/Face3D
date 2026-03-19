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

        // AI-powered post-pipeline analysis (async, non-blocking)
        const sessionId = state.sessionName;
        if (exitCode === 0) {
          // Quality critique on renders (if AI enabled)
          import('../lib/gemini').then(({ isGeminiEnabled, critiqueRenders }) => {
            isGeminiEnabled().then(enabled => {
              if (!enabled || !sessionId) return;
              // Load 5 turntable frames as base64 for critique
              import('../lib/api').then(({ listRenders }) => {
                listRenders(sessionId).then(async renders => {
                  if (renders.length < 3) return;
                  // Pick 5 evenly-spaced frames
                  const step = Math.max(1, Math.floor(renders.length / 5));
                  const picks = [0, step, step * 2, step * 3, Math.min(step * 4, renders.length - 1)];
                  try {
                    const { convertFileSrc } = await import('@tauri-apps/api/core');
                    const base64s: string[] = [];
                    for (const idx of picks) {
                      const src = convertFileSrc(renders[idx]);
                      const resp = await fetch(src);
                      const blob = await resp.blob();
                      const buffer = await blob.arrayBuffer();
                      base64s.push(btoa(String.fromCharCode(...new Uint8Array(buffer))));
                    }
                    const critique = await critiqueRenders(base64s, sessionId);
                    if (critique) {
                      console.log(`[AI] Quality critique: ${critique.grade} (${critique.score}/100) — ${critique.summary}`);
                    }
                  } catch (e) {
                    console.warn('[AI] Quality critique skipped:', e);
                  }
                }).catch(() => {});
              });
            }).catch(() => {});
          }).catch(() => {});
        } else {
          // Error diagnosis (if AI enabled)
          import('../lib/gemini').then(({ isGeminiEnabled, diagnoseError }) => {
            isGeminiEnabled().then(enabled => {
              if (!enabled) return;
              // Collect last 20 error log lines
              const errorLines = state.logs
                .filter(l => l.level === 'error' || l.message.includes('Error') || l.message.includes('Traceback'))
                .slice(-20)
                .map(l => l.message)
                .join('\n');
              if (!errorLines) return;
              const currentStage = state.stages.find(s => s.status === 'error')?.name ?? 'Unknown';
              diagnoseError(errorLines, currentStage, sessionId).then(diagnosis => {
                if (diagnosis) {
                  console.log(`[AI] Error diagnosis: ${diagnosis.summary}`);
                  console.log(`[AI] Fix: ${diagnosis.fix}`);
                }
              }).catch(() => {});
            }).catch(() => {});
          }).catch(() => {});
        }

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
