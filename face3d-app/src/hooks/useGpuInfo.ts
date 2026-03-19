/**
 * Reactive GPU info hook.
 *
 * Polls the backend at the given interval and returns the latest GPU
 * info. Automatically pauses when the component unmounts.
 */

import { useState, useEffect, useRef } from "react";
import { getGpuInfo, type GpuInfo } from "../lib/api";

export function useGpuInfo(intervalMs: number = 5000): GpuInfo | null {
  const [gpu, setGpu] = useState<GpuInfo | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;

    // Initial fetch
    getGpuInfo().then((info) => {
      if (mountedRef.current) setGpu(info);
    });

    const id = setInterval(async () => {
      const info = await getGpuInfo();
      if (mountedRef.current) setGpu(info);
    }, intervalMs);

    return () => {
      mountedRef.current = false;
      clearInterval(id);
    };
  }, [intervalMs]);

  return gpu;
}

export default useGpuInfo;
