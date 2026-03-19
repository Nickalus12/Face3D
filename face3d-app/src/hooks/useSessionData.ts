/**
 * Loads session files, metrics, and render counts for the given session.
 *
 * Returns an AsyncState-shaped object so the consuming component can
 * show loading / error / data states uniformly.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import {
  getSessionFiles,
  getSessionMetrics,
  getImageCounts,
  listRenders,
} from "../lib/api";

export interface SessionData {
  files: { name: string; size: number }[];
  metrics: string;
  imageCounts: [number, number, number];
  renderPaths: string[];
  loading: boolean;
  error: string | null;
  lastFetched: number | null;
  refetch: () => void;
}

export function useSessionData(sessionId: string | null): SessionData {
  const [files, setFiles] = useState<{ name: string; size: number }[]>([]);
  const [metrics, setMetrics] = useState("");
  const [imageCounts, setImageCounts] = useState<[number, number, number]>([0, 0, 0]);
  const [renderPaths, setRenderPaths] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastFetched, setLastFetched] = useState<number | null>(null);

  const mountedRef = useRef(true);

  const fetchAll = useCallback(async () => {
    if (!sessionId) {
      setFiles([]);
      setMetrics("");
      setImageCounts([0, 0, 0]);
      setRenderPaths([]);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const [rawFiles, rawMetrics, counts, renders] = await Promise.all([
        getSessionFiles(sessionId),
        getSessionMetrics(sessionId),
        getImageCounts(sessionId),
        listRenders(sessionId),
      ]);

      if (!mountedRef.current) return;

      setFiles(rawFiles.map(([name, size]) => ({ name, size })));
      setMetrics(rawMetrics);
      setImageCounts(counts);
      setRenderPaths(renders);
      setLastFetched(Date.now());
    } catch (e) {
      if (!mountedRef.current) return;
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    mountedRef.current = true;
    fetchAll();
    return () => {
      mountedRef.current = false;
    };
  }, [fetchAll]);

  return {
    files,
    metrics,
    imageCounts,
    renderPaths,
    loading,
    error,
    lastFetched,
    refetch: fetchAll,
  };
}

export default useSessionData;
