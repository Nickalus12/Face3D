import { useEffect, useRef, useState } from 'react';
import usePipelineStore from '../store/pipelineStore';

export default function StatusBar() {
  const { status, currentStage, gpuInfo, metrics } = usePipelineStore();

  const isRunning = status === 'running';
  const latestMetric = metrics.length > 0 ? metrics[metrics.length - 1] : null;

  // Parse GPU info strings
  const gpuName = gpuInfo?.name ?? '--';
  const vramUsedMB = gpuInfo?.memory_used ? parseFloat(gpuInfo.memory_used) : 0;
  const vramTotalMB = gpuInfo?.memory_total ? parseFloat(gpuInfo.memory_total) : 0;
  const vramUsedGB = (vramUsedMB / 1024).toFixed(1);
  const vramTotalGB = Math.round(vramTotalMB / 1024);
  const vramPercent = vramTotalMB > 0 ? (vramUsedMB / vramTotalMB) * 100 : 0;

  // GPU utilization
  const utilization = gpuInfo?.utilization ? parseInt(gpuInfo.utilization, 10) : 0;
  const utilColor =
    utilization > 80
      ? 'text-red-400'
      : utilization > 50
      ? 'text-amber-400'
      : 'text-emerald-400';

  // VRAM bar color
  const vramBarColor =
    vramPercent > 80
      ? 'bg-red-500'
      : vramPercent > 60
      ? 'bg-amber-500'
      : 'bg-emerald-500';

  // Gaussian count formatting
  const gsCount = latestMetric?.gaussians;
  const gsDisplay = gsCount
    ? gsCount >= 1_000_000
      ? `${(gsCount / 1_000_000).toFixed(2)}M`
      : gsCount >= 1_000
      ? `${Math.round(gsCount / 1_000)}K`
      : gsCount.toString()
    : '--';

  const statusLabel = isRunning
    ? `Stage ${currentStage}/14`
    : status === 'complete'
    ? 'Complete'
    : status === 'error'
    ? 'Error'
    : 'Idle';

  // FPS counter
  const [fps, setFps] = useState(0);
  const frameRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(performance.now());
  const frameCountRef = useRef<number>(0);

  useEffect(() => {
    const tick = (now: number) => {
      frameCountRef.current++;
      const delta = now - lastTimeRef.current;
      if (delta >= 1000) {
        setFps(Math.round((frameCountRef.current * 1000) / delta));
        frameCountRef.current = 0;
        lastTimeRef.current = now;
      }
      frameRef.current = requestAnimationFrame(tick);
    };
    frameRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameRef.current);
  }, []);

  // Separator dot
  const Dot = () => (
    <span className="text-zinc-700 text-[6px] leading-none select-none">&#x2022;</span>
  );

  return (
    <div className="h-[30px] gradient-border-top bg-[#08080a] flex items-center px-3 justify-between text-xs font-mono tracking-wide text-zinc-500 shrink-0 z-50 select-none">
      {/* Left side metrics */}
      <div className="flex items-center gap-2.5">
        {/* Pipeline Status */}
        <div className="flex items-center gap-1.5 bg-white/[0.03] px-2 py-0.5 rounded text-zinc-300">
          <span className="relative flex h-1.5 w-1.5">
            {isRunning && (
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60" />
            )}
            <span
              className={`relative inline-flex rounded-full h-1.5 w-1.5 ${
                isRunning
                  ? 'bg-emerald-500'
                  : status === 'complete'
                  ? 'bg-blue-500'
                  : status === 'error'
                  ? 'bg-red-500'
                  : 'bg-zinc-600'
              }`}
            />
          </span>
          <span className="tracking-tight">
            {isRunning ? (
              <span className="animated-dots">{statusLabel}</span>
            ) : (
              statusLabel
            )}
          </span>
        </div>

        <Dot />

        <div className="flex items-center gap-1.5 hover:text-zinc-300 transition-colors cursor-default">
          <span className="text-zinc-600">GPU</span>
          <span className="text-zinc-300">{gpuName}</span>
        </div>

        <Dot />

        {/* GPU Utilization */}
        <div className="flex items-center gap-1.5 hover:text-zinc-300 transition-colors cursor-default">
          <span className="text-zinc-600">Util</span>
          <span className={utilColor}>
            {gpuInfo ? `${utilization}%` : '--'}
          </span>
        </div>

        <Dot />

        {/* VRAM with inline mini progress bar */}
        <div className="flex items-center gap-1.5 hover:text-zinc-300 transition-colors cursor-default">
          <span className="text-zinc-600">VRAM</span>
          <span className="text-zinc-300">
            {gpuInfo ? `${vramUsedGB}/${vramTotalGB}GB` : '--'}
          </span>
          {gpuInfo && (
            <div className="w-12 h-[4px] bg-zinc-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${vramBarColor}`}
                style={{ width: `${Math.min(100, vramPercent)}%` }}
              />
            </div>
          )}
        </div>

        <Dot />

        <div className="flex items-center gap-1.5 hover:text-zinc-300 transition-colors cursor-default">
          <span className="text-zinc-600">GS</span>
          <span className="text-zinc-300">{gsDisplay}</span>
        </div>

        <Dot />

        <div className="flex items-center gap-1.5 hover:text-zinc-300 transition-colors cursor-default">
          <span className="text-zinc-600">PSNR</span>
          <span className="text-indigo-400 font-bold">
            {latestMetric?.psnr ? latestMetric.psnr.toFixed(2) : '--'}
          </span>
        </div>
      </div>

      {/* Right side info */}
      <div className="flex items-center gap-2.5 text-zinc-600">
        <span className="text-zinc-500 tabular-nums">{fps} fps</span>
        <Dot />
        <span>Face3D v2.0.4-beta</span>
      </div>
    </div>
  );
}
