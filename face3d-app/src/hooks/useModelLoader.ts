/**
 * Loads a PLY model for the given session with progress tracking.
 *
 * Flow:
 *   1. Call getModelPath(sessionId) via the typed API
 *   2. Convert the local path to a Tauri asset URL
 *   3. Fetch + parse the .ply binary
 *   4. Return point cloud data with loading/error state
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { getModelPath, isTauri } from "../lib/api";
import { parsePlyBuffer, type PlyData } from "../lib/plyLoader";

export interface ModelLoaderState {
  model: PlyData | null;
  loading: boolean;
  error: string | null;
  progress: number; // 0-100
  reload: () => void;
}

export function useModelLoader(
  sessionId: string | null,
): ModelLoaderState {
  const [model, setModel] = useState<PlyData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const mountedRef = useRef(true);
  const abortRef = useRef<AbortController | null>(null);

  const load = useCallback(async () => {
    if (!sessionId) {
      setModel(null);
      setProgress(0);
      return;
    }

    // Abort any in-flight request
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError(null);
    setProgress(0);

    try {
      const modelPath = await getModelPath(sessionId);
      if (!mountedRef.current || controller.signal.aborted) return;

      if (!modelPath) {
        setLoading(false);
        return;
      }

      // Build URL — in Tauri, use convertFileSrc; in dev, just use path
      let url: string;
      if (isTauri()) {
        // Dynamic import to avoid breaking dev mode
        const { convertFileSrc } = await import("@tauri-apps/api/core");
        url = convertFileSrc(modelPath);
      } else {
        // Dev mode — no real model to load
        setLoading(false);
        return;
      }

      setProgress(10);

      const response = await fetch(url, { signal: controller.signal });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const contentLength = response.headers.get("content-length");
      const totalBytes = contentLength ? parseInt(contentLength, 10) : 0;

      // Stream the response for progress reporting
      const reader = response.body?.getReader();
      if (!reader) {
        // Fallback: no streaming
        const buffer = await response.arrayBuffer();
        if (!mountedRef.current || controller.signal.aborted) return;
        setProgress(90);
        const data = parsePlyBuffer(buffer);
        setModel(data);
        setProgress(100);
        setLoading(false);
        return;
      }

      const chunks: Uint8Array[] = [];
      let received = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (controller.signal.aborted) return;
        chunks.push(value);
        received += value.length;

        if (totalBytes > 0) {
          setProgress(10 + Math.round((received / totalBytes) * 80));
        }
      }

      if (!mountedRef.current || controller.signal.aborted) return;

      // Merge chunks into a single buffer
      const merged = new Uint8Array(received);
      let offset = 0;
      for (const chunk of chunks) {
        merged.set(chunk, offset);
        offset += chunk.length;
      }

      setProgress(90);
      const data = parsePlyBuffer(merged.buffer);
      if (!mountedRef.current) return;
      setModel(data);
      setProgress(100);
    } catch (e) {
      if (!mountedRef.current) return;
      if ((e as Error).name === "AbortError") return;
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    mountedRef.current = true;
    load();
    return () => {
      mountedRef.current = false;
      abortRef.current?.abort();
    };
  }, [load]);

  return { model, loading, error, progress, reload: load };
}

export default useModelLoader;
