import { useState } from 'react';
import Sidebar from './components/Sidebar';
import Viewer3D from './components/Viewer3D';
import StatusBar from './components/StatusBar';
import { ChevronRight, ChevronLeft, Terminal, Clock, Layers, Maximize2, Play, Pause, MoreHorizontal } from 'lucide-react';

export default function App() {
  const [isPanelOpen, setIsPanelOpen] = useState(true);
  const [isConsoleOpen, setIsConsoleOpen] = useState(false);
  const [isRunning, setIsRunning] = useState(true);

  return (
    <div className="flex flex-col h-screen w-screen bg-[#0a0a0b] text-zinc-300 font-sans overflow-hidden selection:bg-indigo-500/30">
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />

        {/* Collapsible Detail Panel */}
        <div 
          className={`relative transition-all duration-400 ease-[cubic-bezier(0.16,1,0.3,1)] bg-[#0e0e11] border-r border-zinc-800/40 flex flex-col z-20 shrink-0 ${
            isPanelOpen ? 'w-[280px] opacity-100' : 'w-0 opacity-0 overflow-hidden'
          }`}
        >
          {/* Panel Header */}
          <div className="h-14 flex items-center justify-between px-4 border-b border-zinc-800/40 shrink-0">
            <h2 className="text-xs font-semibold tracking-widest text-zinc-400 uppercase">Session Info</h2>
            <button className="text-zinc-500 hover:text-zinc-200 transition-colors">
              <MoreHorizontal size={16} />
            </button>
          </div>

          {/* Panel Content */}
          <div className="flex-1 overflow-y-auto p-4 space-y-6 scrollbar-hide">
            {/* Subject Info */}
            <div>
              <div className="text-[10px] font-bold tracking-wider text-zinc-500 uppercase mb-3">Target Subject</div>
              <div className="flex items-center gap-3 bg-white/[0.02] p-3 rounded-lg border border-white/5">
                <div className="w-10 h-10 bg-gradient-to-br from-indigo-500/20 to-purple-500/20 rounded-md border border-indigo-500/30 flex items-center justify-center">
                  <span className="text-indigo-400 text-xs font-bold">SUB</span>
                </div>
                <div>
                  <div className="text-sm font-medium text-zinc-200">Patient_042_A</div>
                  <div className="text-[11px] text-zinc-500 font-mono mt-0.5">ID: f3-8992-abc</div>
                </div>
              </div>
            </div>

            {/* Pipeline Status */}
            <div>
              <div className="text-[10px] font-bold tracking-wider text-zinc-500 uppercase mb-3">Pipeline Overview</div>
              <div className="space-y-2 relative before:absolute before:inset-y-3 before:left-[11px] before:w-[2px] before:bg-zinc-800/50">
                {/* Stage 1 */}
                <div className="relative flex items-center gap-3">
                  <div className="w-6 h-6 rounded-full bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center z-10 shrink-0">
                    <div className="w-2 h-2 rounded-full bg-emerald-400" />
                  </div>
                  <div className="flex-1 p-2.5 bg-white/[0.02] rounded-lg border border-white/5 text-sm flex justify-between items-center">
                    <span className="text-zinc-300">Alignment</span>
                    <span className="text-[10px] text-zinc-500 font-mono">1.2s</span>
                  </div>
                </div>
                {/* Stage 2 */}
                <div className="relative flex items-center gap-3">
                  <div className="w-6 h-6 rounded-full bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center z-10 shrink-0 ring-4 ring-[#0e0e11]">
                    <div className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />
                  </div>
                  <div className="flex-1 p-2.5 bg-indigo-500/5 rounded-lg border border-indigo-500/20 text-sm flex justify-between items-center shadow-[0_0_15px_rgba(99,102,241,0.05)]">
                    <span className="text-indigo-300 font-medium">Splatting</span>
                    <span className="text-[10px] text-indigo-400 font-mono animate-pulse">Running</span>
                  </div>
                </div>
                {/* Stage 3 */}
                <div className="relative flex items-center gap-3">
                  <div className="w-6 h-6 rounded-full bg-zinc-800/50 border border-zinc-700/50 flex items-center justify-center z-10 shrink-0">
                    <div className="w-2 h-2 rounded-full bg-zinc-600" />
                  </div>
                  <div className="flex-1 p-2.5 bg-transparent border border-transparent text-sm flex justify-between items-center opacity-50 text-zinc-500">
                    <span>Mesh Gen</span>
                    <span className="text-[10px] font-mono">Wait</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Quick Stats */}
            <div>
              <div className="text-[10px] font-bold tracking-wider text-zinc-500 uppercase mb-3">Live Metrics</div>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-white/[0.02] border border-white/5 p-3 rounded-lg flex flex-col gap-1">
                  <Layers size={14} className="text-blue-400" />
                  <span className="text-lg font-medium text-zinc-200 mt-1">450K</span>
                  <span className="text-[10px] text-zinc-500 uppercase">Gaussians</span>
                </div>
                <div className="bg-white/[0.02] border border-white/5 p-3 rounded-lg flex flex-col gap-1">
                  <Clock size={14} className="text-amber-400" />
                  <span className="text-lg font-medium text-zinc-200 mt-1">1m 24s</span>
                  <span className="text-[10px] text-zinc-500 uppercase">Elapsed</span>
                </div>
              </div>
            </div>
          </div>

          {/* Master Control */}
          <div className="p-4 border-t border-zinc-800/40 bg-zinc-900/20 shrink-0">
             <button 
                onClick={() => setIsRunning(!isRunning)}
                className={`w-full py-2.5 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-all shadow-lg ${
                  isRunning 
                    ? 'bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/20 shadow-red-500/5' 
                    : 'bg-indigo-500 text-white hover:bg-indigo-600 border border-indigo-400 shadow-indigo-500/20'
                }`}
             >
                {isRunning ? <Pause size={16} /> : <Play size={16} fill="currentColor" />}
                {isRunning ? 'Pause Pipeline' : 'Resume Pipeline'}
             </button>
          </div>
        </div>

        {/* Main Center Area */}
        <div className="flex-1 relative flex flex-col min-w-0 bg-[#0a0a0b]">
          {/* Panel Toggle button */}
          <button 
            onClick={() => setIsPanelOpen(!isPanelOpen)} 
            className="absolute top-5 left-4 z-20 p-1.5 bg-[#111113]/80 backdrop-blur-xl border border-white/10 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800/80 transition-all shadow-lg"
          >
            {isPanelOpen ? <ChevronLeft size={16}/> : <ChevronRight size={16}/>}
          </button>

          {/* 3D Viewport */}
          <div className="flex-1 relative rounded-tl-xl overflow-hidden border-t border-l border-white/5">
            <Viewer3D />
          </div>

          {/* Console Trigger Button (when closed) */}
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20">
             <button 
               onClick={() => setIsConsoleOpen(!isConsoleOpen)} 
               className={`flex items-center gap-2 px-4 py-1.5 bg-[#111113]/90 backdrop-blur-md border border-white/10 rounded-full text-[11px] font-medium transition-all shadow-xl ${
                 isConsoleOpen ? 'opacity-0 pointer-events-none' : 'opacity-100 text-zinc-400 hover:text-white hover:scale-105'
               }`}
             >
                <Terminal size={14} className="text-indigo-400" /> View Logs
             </button>
          </div>

          {/* Animated Bottom Console */}
          <div 
            className={`absolute bottom-0 left-0 right-0 transition-transform duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] z-30 flex flex-col bg-[#0d0d0f]/95 backdrop-blur-2xl border-t border-white/10 rounded-t-2xl shadow-[0_-20px_40px_rgba(0,0,0,0.5)] ${
              isConsoleOpen ? 'translate-y-0' : 'translate-y-full'
            }`} 
            style={{ height: '240px' }}
          >
            <div className="flex items-center justify-between px-4 py-3 border-b border-white/5 shrink-0">
               <div className="flex items-center gap-2">
                 <Terminal size={14} className="text-zinc-400" />
                 <span className="text-xs font-semibold tracking-wider text-zinc-300">SYSTEM CONSOLE</span>
               </div>
               <div className="flex items-center gap-2">
                 <button className="text-zinc-500 hover:text-zinc-300 p-1"><Maximize2 size={14} /></button>
                 <button onClick={() => setIsConsoleOpen(false)} className="text-zinc-500 hover:text-white p-1 bg-white/5 rounded-md"><ChevronRight size={14} className="rotate-90" /></button>
               </div>
            </div>
            <div className="p-4 font-mono text-[11px] leading-relaxed text-zinc-400 flex-1 overflow-y-auto scrollbar-hide space-y-1">
              <div className="text-zinc-600">[10:42:01] Initialization started...</div>
              <div className="text-zinc-600">[10:42:02] Loading model checkpoints: face_recon_v3.ckpt</div>
              <div className="text-emerald-500">[10:42:04] Successfully loaded 128MB. GPU Memory: 2.1GB allocated.</div>
              <div className="text-zinc-400">[10:42:05] Optimizing point cloud... Initial points: 245,190</div>
              <div className="text-zinc-500 pl-4">Iter 100/1000: Loss = 0.0453 | Densification triggered</div>
              <div className="text-zinc-500 pl-4">Iter 200/1000: Loss = 0.0210</div>
              <div className="text-indigo-400 animate-pulse pl-4">Iter 300/1000: Loss = 0.0154 | Splatting active..._</div>
            </div>
          </div>
        </div>
      </div>

      <StatusBar isRunning={isRunning} />
    </div>
  );
}
