/**
 * Loads sensor data for a session with built-in caching (via the
 * IPC cache in api.ts). Returns an AsyncState-shaped object.
 */

import { useState, useEffect, useRef } from "react";
import { getSensorData, getSensorSummary, type SensorSummary } from "../lib/api";

export interface SensorDataResult {
  summary: SensorSummary | null;
  data: number[][] | null;
  loading: boolean;
  error: string | null;
}

export function useSensorData(
  sessionId: string | null,
  sensorType: string,
): SensorDataResult {
  const [summary, setSummary] = useState<SensorSummary | null>(null);
  const [data, setData] = useState<number[][] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;

    if (!sessionId) {
      setSummary(null);
      setData(null);
      return;
    }

    setLoading(true);
    setError(null);

    Promise.all([
      getSensorSummary(sessionId),
      getSensorData(sessionId, sensorType),
    ])
      .then(([s, d]) => {
        if (!mountedRef.current) return;
        setSummary(s);
        setData(d);
      })
      .catch((e) => {
        if (!mountedRef.current) return;
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (mountedRef.current) setLoading(false);
      });

    return () => {
      mountedRef.current = false;
    };
  }, [sessionId, sensorType]);

  return { summary, data, loading, error };
}

export default useSensorData;
