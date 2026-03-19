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

  return (
    <div className="h-[22px] bg-[#0a0a0b] border-t border-white/5 flex items-center px-3 justify-between text-[10px] font-mono tracking-wide text-zinc-500 shrink-0 z-50 select-none">
      {/* Left side metrics */}
      <div className="flex items-center gap-3">
        {/* Pipeline Status */}
        <div className="flex items-center gap-2 bg-white/5 px-2 py-0.5 rounded text-zinc-300">
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
          <span className="tracking-tight">{statusLabel}</span>
        </div>

        <div className="w-[1px] h-3 bg-white/10" />

        <div className="flex items-center gap-1.5 hover:text-zinc-300 transition-colors cursor-default">
          <span className="text-zinc-600">GPU</span>
          <span className="text-zinc-300">{gpuName}</span>
        </div>

        <div className="w-[1px] h-3 bg-white/10" />

        <div className="flex items-center gap-1.5 hover:text-zinc-300 transition-colors cursor-default">
          <span className="text-zinc-600">VRAM</span>
          <span className="text-zinc-300">
            {gpuInfo ? `${vramUsedGB}/${vramTotalGB}GB` : '--'}
          </span>
        </div>

        <div className="w-[1px] h-3 bg-white/10" />

        <div className="flex items-center gap-1.5 hover:text-zinc-300 transition-colors cursor-default">
          <span className="text-zinc-600">GS</span>
          <span className="text-zinc-300">{gsDisplay}</span>
        </div>

        <div className="w-[1px] h-3 bg-white/10" />

        <div className="flex items-center gap-1.5 hover:text-zinc-300 transition-colors cursor-default">
          <span className="text-zinc-600">PSNR</span>
          <span className="text-indigo-400 font-bold">
            {latestMetric?.psnr ? latestMetric.psnr.toFixed(2) : '--'}
          </span>
        </div>
      </div>

      {/* Right side info */}
      <div className="flex items-center gap-4 text-zinc-600">
        <span className="hover:text-zinc-400 transition-colors cursor-pointer">
          {gpuInfo?.utilization ? `Util: ${gpuInfo.utilization}` : ''}
        </span>
        <div className="w-[1px] h-3 bg-white/10" />
        <span>Face3D v2.0.4-beta</span>
      </div>
    </div>
  );
}
