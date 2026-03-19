import React, { useState } from 'react';
import { 
  Play, Square, Pause, UploadCloud, 
  TerminalSquare, Sparkles, ChevronUp 
} from 'lucide-react';

export const PipelinePanel: React.FC = () => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [status, setStatus] = useState<'idle' | 'running' | 'paused'>('idle');
  
  // Mock Progress
  const percent = 68;
  const radius = 28;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percent / 100) * circumference;

  if (isCollapsed) {
    return (
      <button 
        onClick={() => setIsCollapsed(false)}
        className="fixed right-6 top-24 z-30 p-3 bg-zinc-900/90 backdrop-blur border border-zinc-800 rounded-full shadow-2xl text-zinc-400 hover:text-emerald-400 hover:border-emerald-500/50 transition-all group"
      >
        <Play size={20} className="group-hover:scale-110 transition-transform" />
      </button>
    );
  }

  return (
    <div className="fixed right-6 top-24 w-[340px] bg-[#0a0a0b]/95 backdrop-blur-xl border border-zinc-800 rounded-2xl z-30 shadow-[0_0_40px_rgba(0,0,0,0.8)] flex flex-col overflow-hidden animate-in fade-in slide-in-from-right-4 duration-300">
      
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-zinc-800/80 bg-zinc-900/30">
        <div className="flex items-center gap-2">
          <TerminalSquare size={16} className="text-emerald-500" />
          <h3 className="text-zinc-100 font-medium text-sm">Control Panel</h3>
        </div>
        <button onClick={() => setIsCollapsed(true)} className="text-zinc-500 hover:text-zinc-300 transition-colors">
          <ChevronUp size={16} />
        </button>
      </div>

      <div className="p-5 space-y-6">
        
        {/* Input Configuration */}
        <div className="space-y-3">
          <label className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Input Directory</label>
          <div className="relative group cursor-pointer">
            <div className="absolute inset-0 bg-zinc-800/50 rounded-xl group-hover:bg-zinc-800 transition-colors" />
            <div className="relative border border-dashed border-zinc-700 group-hover:border-emerald-500/50 rounded-xl p-4 flex flex-col items-center justify-center gap-2 text-center transition-colors">
              <UploadCloud size={20} className="text-zinc-400 group-hover:text-emerald-400 transition-colors" />
              <p className="text-xs text-zinc-400 group-hover:text-zinc-300">Drop images here or <span className="text-emerald-400">browse</span></p>
            </div>
          </div>
        </div>

        {/* Session Name */}
        <div className="space-y-3">
          <label className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Session Name</label>
          <div className="flex gap-2">
            <input 
              type="text" 
              placeholder="e.g. bust_scan_01"
              className="flex-1 bg-zinc-900/50 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/50 transition-all"
            />
            <button className="p-2 bg-zinc-900/50 border border-zinc-800 rounded-lg text-zinc-400 hover:text-emerald-400 hover:border-emerald-500/50 transition-colors" title="Auto-generate">
              <Sparkles size={16} />
            </button>
          </div>
        </div>

        {/* Controls & Progress */}
        <div className="p-4 bg-zinc-900/40 border border-zinc-800 rounded-xl flex items-center justify-between">
          <div className="flex gap-2">
            <button 
              onClick={() => setStatus(status === 'running' ? 'paused' : 'running')}
              className={`p-3 rounded-lg flex items-center justify-center transition-all ${
                status === 'running' 
                  ? 'bg-amber-500/10 text-amber-500 border border-amber-500/20 hover:bg-amber-500/20' 
                  : 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20 hover:bg-emerald-500/20'
              }`}
            >
              {status === 'running' ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}
            </button>
            <button 
              onClick={() => setStatus('idle')}
              disabled={status === 'idle'}
              className="p-3 rounded-lg bg-red-500/5 text-red-500 border border-red-500/10 hover:bg-red-500/10 hover:border-red-500/20 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
            >
              <Square size={18} fill="currentColor" />
            </button>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-right">
              <div className="text-xs text-zinc-400">Stage 7/15</div>
              <div className="text-sm font-medium text-emerald-400 animate-pulse">Running...</div>
            </div>
            
            {/* Circular Progress */}
            <div className="relative w-14 h-14">
              <svg className="w-14 h-14 transform -rotate-90">
                <circle cx="28" cy="28" r={radius} stroke="currentColor" strokeWidth="4" fill="transparent" className="text-zinc-800" />
                <circle 
                  cx="28" cy="28" r={radius} 
                  stroke="currentColor" 
                  strokeWidth="4" 
                  fill="transparent" 
                  strokeDasharray={circumference} 
                  strokeDashoffset={strokeDashoffset} 
                  className="text-emerald-500 transition-all duration-500 ease-out" 
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-zinc-200">
                {percent}%
              </div>
            </div>
          </div>
        </div>

        {/* Mini Log */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-zinc-500">
            <span className="font-semibold uppercase tracking-wider">Recent Logs</span>
            <span>04:32 elapsed</span>
          </div>
          <div className="bg-[#050505] border border-zinc-800/80 rounded-lg p-3 font-mono text-[10px] text-zinc-400 space-y-1 h-20 overflow-hidden relative">
            <p><span className="text-zinc-600">[INFO]</span> Optimizing sh harmonics...</p>
            <p><span className="text-zinc-600">[INFO]</span> Iteration 4500/7000</p>
            <p><span className="text-zinc-600">[INFO]</span> Loss: 0.0412 PSNR: 28.4</p>
            <p className="text-emerald-400"><span className="text-emerald-500/50">[INFO]</span> Gaussians: 1,204,501</p>
            <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-[#050505] to-transparent pointer-events-none" />
          </div>
        </div>

      </div>
    </div>
  );
};
