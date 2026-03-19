

interface StatusBarProps {
  isRunning?: boolean;
}

export default function StatusBar({ isRunning = true }: StatusBarProps) {
  return (
    <div className="h-[22px] bg-[#0a0a0b] border-t border-white/5 flex items-center px-3 justify-between text-[10px] font-mono tracking-wide text-zinc-500 shrink-0 z-50 select-none">
      {/* Left side metrics */}
      <div className="flex items-center gap-3">
        {/* Pipeline Status */}
        <div className="flex items-center gap-2 bg-white/5 px-2 py-0.5 rounded text-zinc-300">
          <span className="relative flex h-1.5 w-1.5">
            {isRunning && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60"></span>}
            <span className={`relative inline-flex rounded-full h-1.5 w-1.5 ${isRunning ? 'bg-emerald-500' : 'bg-zinc-600'}`}></span>
          </span>
          <span className="tracking-tight">Stage 6/14</span>
        </div>

        <div className="w-[1px] h-3 bg-white/10" />
        
        <div className="flex items-center gap-1.5 hover:text-zinc-300 transition-colors cursor-default">
          <span className="text-zinc-600">GPU</span>
          <span className="text-zinc-300">RTX 3080</span>
        </div>

        <div className="w-[1px] h-3 bg-white/10" />

        <div className="flex items-center gap-1.5 hover:text-zinc-300 transition-colors cursor-default">
          <span className="text-zinc-600">VRAM</span>
          <span className="text-zinc-300">2.1/16GB</span>
        </div>

        <div className="w-[1px] h-3 bg-white/10" />

        <div className="flex items-center gap-1.5 hover:text-zinc-300 transition-colors cursor-default">
          <span className="text-zinc-600">GS</span>
          <span className="text-zinc-300">450K</span>
        </div>

        <div className="w-[1px] h-3 bg-white/10" />

        <div className="flex items-center gap-1.5 hover:text-zinc-300 transition-colors cursor-default">
          <span className="text-zinc-600">IT/S</span>
          <span className="text-indigo-400 font-bold">28.4</span>
        </div>
      </div>
      
      {/* Right side info */}
      <div className="flex items-center gap-4 text-zinc-600">
        <span className="hover:text-zinc-400 transition-colors cursor-pointer">Memory: 4.2GB</span>
        <div className="w-[1px] h-3 bg-white/10" />
        <span>Face3D v2.0.4-beta</span>
      </div>
    </div>
  );
}
