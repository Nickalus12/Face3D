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

export function usePipelineEvents(): void {
  useEffect(() => {
    const cleanups: Array<() => void> = [];

    async function setup() {
      // Pipeline log -> parse into store
      const unLog = await eventManager.onPipelineLog((payload) => {
        usePipelineStore.getState().parseLogLine(payload.line, payload.level);
      });
      cleanups.push(unLog);

      // Pipeline complete -> update stores, invalidate caches
      const unComplete = await eventManager.onPipelineComplete((exitCode) => {
        usePipelineStore.getState().onPipelineComplete(exitCode);
        invalidateSessionCaches();
        useSessionStore.getState().fetchSessions();

        // Flash the window title briefly
        if (document.title) {
          const original = document.title;
          document.title = "Pipeline Complete!";
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
