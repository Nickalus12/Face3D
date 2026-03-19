import React, { useState } from 'react';
import { 
  ChevronLeft, ChevronRight, HardDrive, Clock, Activity, 
  CheckCircle2, CircleDashed, AlertCircle, PlayCircle, Box, Database
} from 'lucide-react';

type ViewMode = 'home' | 'pipeline' | 'metrics';

interface DetailPanelProps {
  currentView?: ViewMode;
  isExpanded?: boolean;
  onToggle?: () => void;
}

export const DetailPanel: React.FC<DetailPanelProps> = ({ 
  currentView = 'pipeline', 
  isExpanded = true,
  onToggle 
}) => {
  const [expandedStage, setExpandedStage] = useState<number | null>(3); // Mock current stage

  const STAGES = [
    { id: 1, name: 'Load Images', status: 'done', time: '00:02' },
    { id: 2, name: 'Extract Features', status: 'done', time: '01:14' },
    { id: 3, name: 'Match Features', status: 'running', time: '04:32' },
    { id: 4, name: 'SfM Initialization', status: 'queued', time: '--:--' },
    { id: 5, name: 'Camera Calibration', status: 'queued', time: '--:--' },
    // Truncated for brevity, normally 15 stages
  ];

  return (
    <div 
      className={`fixed left-0 top-16 bottom-0 bg-[#0a0a0b] border-r border-zinc-800 transition-all duration-300 ease-in-out z-20 flex flex-col shadow-2xl shadow-black
      ${isExpanded ? 'w-[280px] translate-x-0' : 'w-[280px] -translate-x-full'}`}
    >
      {/* Toggle Button (visible when closed, attached to edge) */}
      <button 
        onClick={onToggle}
        className={`absolute top-6 -right-8 w-8 h-10 bg-[#0a0a0b] border border-l-0 border-zinc-800 rounded-r-lg flex items-center justify-center text-zinc-400 hover:text-emerald-400 transition-colors z-30 ${isExpanded ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}
      >
        <ChevronRight size={18} />
      </button>

      {/* Header */}
      <div className="h-14 flex items-center justify-between px-4 border-b border-zinc-800/80 bg-zinc-900/20">
        <h2 className="text-zinc-100 font-semibold text-sm tracking-wide uppercase">
          {currentView === 'home' && 'Sessions'}
          {currentView === 'pipeline' && 'Training Pipeline'}
          {currentView === 'metrics' && 'Performance'}
        </h2>
        <button onClick={onToggle} className="text-zinc-500 hover:text-zinc-300 transition-colors">
          <ChevronLeft size={18} />
        </button>
      </div>

      {/* Content Area with Custom Scrollbar */}
      <div className="flex-1 overflow-y-auto [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:bg-zinc-800 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-track]:bg-transparent p-4">
        
        {/* PIPELINE VIEW */}
        {currentView === 'pipeline' && (
          <div className="relative pl-2">
            <div className="absolute left-4 top-2 bottom-6 w-[1px] bg-zinc-800" />
            
            <div className="space-y-6">
              {STAGES.map((stage) => (
                <div key={stage.id} className="relative z-10 flex flex-col gap-2">
                  <div 
                    className="flex items-start gap-3 cursor-pointer group"
                    onClick={() => setExpandedStage(expandedStage === stage.id ? null : stage.id)}
                  >
                    <div className={`mt-0.5 bg-[#0a0a0b] rounded-full p-0.5 ${
                      stage.status === 'done' ? 'text-emerald-500' : 
                      stage.status === 'running' ? 'text-amber-400 animate-pulse' : 
                      'text-zinc-600'
                    }`}>
                      {stage.status === 'done' && <CheckCircle2 size={16} />}
                      {stage.status === 'running' && <PlayCircle size={16} />}
                      {stage.status === 'queued' && <CircleDashed size={16} />}
                      {stage.status === 'error' && <AlertCircle size={16} className="text-red-500" />}
                    </div>
                    <div className="flex-1">
                      <div className="flex justify-between items-center">
                        <span className={`text-sm font-medium ${stage.status === 'running' ? 'text-zinc-100' : 'text-zinc-400'} group-hover:text-zinc-200 transition-colors`}>
                          {stage.name}
                        </span>
                        <span className="text-xs text-zinc-600 font-mono">{stage.time}</span>
                      </div>
                    </div>
                  </div>

                  {/* Expandable Details */}
                  {expandedStage === stage.id && (
                    <div className="ml-8 p-3 rounded-lg bg-zinc-900/40 border border-zinc-800/50 text-xs text-zinc-400 space-y-2 animate-in slide-in-from-top-2 duration-200">
                      <div className="flex justify-between border-b border-zinc-800 pb-1">
                        <span>Memory</span>
                        <span className="text-zinc-200">2.4 GB</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Points</span>
                        <span className="text-zinc-200">142,050</span>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* METRICS VIEW */}
        {currentView === 'metrics' && (
          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-zinc-900/30 border border-zinc-800 hover:border-zinc-700 transition-colors group">
              <div className="flex items-center gap-2 text-zinc-500 mb-2">
                <Activity size={14} />
                <span className="text-xs uppercase tracking-wider font-semibold">Best PSNR</span>
              </div>
              <div className="text-2xl font-light text-emerald-400 group-hover:scale-105 origin-left transition-transform">34.21 dB</div>
            </div>
            
            <div className="p-4 rounded-xl bg-zinc-900/30 border border-zinc-800 hover:border-zinc-700 transition-colors group">
              <div className="flex items-center gap-2 text-zinc-500 mb-2">
                <Box size={14} />
                <span className="text-xs uppercase tracking-wider font-semibold">Gaussians</span>
              </div>
              <div className="text-2xl font-light text-zinc-100 group-hover:scale-105 origin-left transition-transform">1.2M</div>
            </div>

            <div className="p-4 rounded-xl bg-zinc-900/30 border border-zinc-800 hover:border-zinc-700 transition-colors group">
              <div className="flex items-center gap-2 text-zinc-500 mb-2">
                <Database size={14} />
                <span className="text-xs uppercase tracking-wider font-semibold">Peak VRAM</span>
              </div>
              <div className="text-2xl font-light text-zinc-100 group-hover:scale-105 origin-left transition-transform">11.4 GB</div>
            </div>
          </div>
        )}

        {/* HOME VIEW (Sessions) */}
        {currentView === 'home' && (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="group p-3 rounded-xl bg-zinc-900/30 border border-zinc-800 hover:border-emerald-500/30 hover:bg-zinc-900/60 cursor-pointer transition-all">
                <div className="flex justify-between items-start mb-2">
                  <span className="text-sm font-medium text-zinc-200 group-hover:text-emerald-400 transition-colors">Session_{i}</span>
                  <span className="text-[10px] text-zinc-500">Oct {24 - i}</span>
                </div>
                <div className="flex gap-2 mb-3">
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Mesh</span>
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Splat</span>
                </div>
                <div className="flex items-center gap-1.5 text-xs text-zinc-500">
                  <HardDrive size={12} />
                  <span>450 MB</span>
                  <span className="mx-1">•</span>
                  <Clock size={12} />
                  <span>45m</span>
                </div>
              </div>
            ))}
          </div>
        )}

      </div>
    </div>
  );
};
