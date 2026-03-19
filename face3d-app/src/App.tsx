import { useEffect, useRef, useState } from 'react';
import Sidebar from './components/Sidebar';
import Viewer3D from './components/Viewer3D';
import StatusBar from './components/StatusBar';
import { ChevronRight, ChevronLeft, Terminal, Maximize2, Play, Pause, MoreHorizontal, Layers, Clock } from 'lucide-react';
import useSessionStore from './store/sessionStore';
import usePipelineStore from './store/pipelineStore';
import { onPipelineLog, onPipelineComplete } from './lib/tauri';

export default function App() {
  const [isPanelOpen, setIsPanelOpen] = useState(true);
  const [isConsoleOpen, setIsConsoleOpen] = useState(false);
  const consoleRef = useRef<HTMLDivElement>(null);

  const { fetchSessions } = useSessionStore();
  const {
    status,
    stages,
    logs,
    metrics,
    startedAt,
    fetchGpuInfo,
    stopPipeline,
  } = usePipelineStore();

  const isRunning = status === 'running' || status === 'stopping';

  // ── Bootstrap on mount ──────────────────────────────────────
  useEffect(() => {
    fetchSessions();
    fetchGpuInfo();
  }, [fetchSessions, fetchGpuInfo]);

  // ── Tauri event listeners ───────────────────────────────────
  useEffect(() => {
    const unlisteners: Array<Promise<() => void>> = [];

    const logUnsub = onPipelineLog((payload) => {
      usePipelineStore.getState().parseLogLine(payload.line, payload.level);
    });
    if (logUnsub) unlisteners.push(logUnsub);

    const completeUnsub = onPipelineComplete((exitCode) => {
      usePipelineStore.getState().onPipelineComplete(exitCode);
      // Refresh sessions to pick up new output files
      useSessionStore.getState().fetchSessions();
    });
    if (completeUnsub) unlisteners.push(completeUnsub);

    return () => {
      unlisteners.forEach((p) => p.then((unsub) => unsub()));
    };
  }, []);

  // ── Poll GPU info while pipeline is running ─────────────────
  useEffect(() => {
    if (!isRunning) return;
    const interval = setInterval(() => {
      fetchGpuInfo();
    }, 5000);
    return () => clearInterval(interval);
  }, [isRunning, fetchGpuInfo]);

  // ── Auto-scroll console ─────────────────────────────────────
  useEffect(() => {
    if (isConsoleOpen && consoleRef.current) {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    }
  }, [logs.length, isConsoleOpen]);

  // ── Elapsed time ────────────────────────────────────────────
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!startedAt || !isRunning) return;
    const tick = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
    return () => clearInterval(tick);
  }, [startedAt, isRunning]);

  const formatElapsed = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}m ${sec.toString().padStart(2, '0')}s`;
  };

  // Latest metrics
  const latestMetric = metrics.length > 0 ? metrics[metrics.length - 1] : null;
  const gaussianCount = latestMetric?.gaussians;

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
            {/* Pipeline Status */}
            <div>
              <div className="text-[10px] font-bold tracking-wider text-zinc-500 uppercase mb-3">Pipeline Overview</div>
              <div className="space-y-2 relative before:absolute before:inset-y-3 before:left-[11px] before:w-[2px] before:bg-zinc-800/50">
                {stages.map((stage) => {
                  const isCurrent = stage.status === 'running';
                  const isDone = stage.status === 'complete';
                  const isError = stage.status === 'error';

                  return (
                    <div key={stage.id} className="relative flex items-center gap-3">
                      <div
                        className={`w-6 h-6 rounded-full flex items-center justify-center z-10 shrink-0 ${
                          isDone
                            ? 'bg-emerald-500/20 border border-emerald-500/30'
                            : isCurrent
                            ? 'bg-indigo-500/20 border border-indigo-500/30 ring-4 ring-[#0e0e11]'
                            : isError
                            ? 'bg-red-500/20 border border-red-500/30'
                            : 'bg-zinc-800/50 border border-zinc-700/50'
                        }`}
                      >
                        <div
                          className={`w-2 h-2 rounded-full ${
                            isDone
                              ? 'bg-emerald-400'
                              : isCurrent
                              ? 'bg-indigo-400 animate-pulse'
                              : isError
                              ? 'bg-red-400'
                              : 'bg-zinc-600'
                          }`}
                        />
                      </div>
                      <div
                        className={`flex-1 p-2.5 rounded-lg text-sm flex justify-between items-center ${
                          isDone
                            ? 'bg-white/[0.02] border border-white/5'
                            : isCurrent
                            ? 'bg-indigo-500/5 border border-indigo-500/20 shadow-[0_0_15px_rgba(99,102,241,0.05)]'
                            : isError
                            ? 'bg-red-500/5 border border-red-500/20'
                            : 'bg-transparent border border-transparent opacity-50'
                        }`}
                      >
                        <span
                          className={
                            isCurrent
                              ? 'text-indigo-300 font-medium'
                              : isDone
                              ? 'text-zinc-300'
                              : isError
                              ? 'text-red-300'
                              : 'text-zinc-500'
                          }
                        >
                          {stage.name}
                        </span>
                        <span
                          className={`text-[10px] font-mono ${
                            isCurrent
                              ? 'text-indigo-400 animate-pulse'
                              : isDone
                              ? 'text-zinc-500'
                              : isError
                              ? 'text-red-400'
                              : 'text-zinc-600'
                          }`}
                        >
                          {isDone
                            ? 'Done'
                            : isCurrent
                            ? 'Running'
                            : isError
                            ? 'Error'
                            : 'Wait'}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Quick Stats */}
            <div>
              <div className="text-[10px] font-bold tracking-wider text-zinc-500 uppercase mb-3">Live Metrics</div>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-white/[0.02] border border-white/5 p-3 rounded-lg flex flex-col gap-1">
                  <Layers size={14} className="text-blue-400" />
                  <span className="text-lg font-medium text-zinc-200 mt-1">
                    {gaussianCount
                      ? gaussianCount >= 1_000_000
                        ? `${(gaussianCount / 1_000_000).toFixed(1)}M`
                        : gaussianCount >= 1_000
                        ? `${Math.round(gaussianCount / 1_000)}K`
                        : gaussianCount.toString()
                      : '--'}
                  </span>
                  <span className="text-[10px] text-zinc-500 uppercase">Gaussians</span>
                </div>
                <div className="bg-white/[0.02] border border-white/5 p-3 rounded-lg flex flex-col gap-1">
                  <Clock size={14} className="text-amber-400" />
                  <span className="text-lg font-medium text-zinc-200 mt-1">
                    {isRunning && startedAt ? formatElapsed(elapsed) : '--'}
                  </span>
                  <span className="text-[10px] text-zinc-500 uppercase">Elapsed</span>
                </div>
              </div>
            </div>
          </div>

          {/* Master Control */}
          <div className="p-4 border-t border-zinc-800/40 bg-zinc-900/20 shrink-0">
            <button
              onClick={() => {
                if (isRunning) {
                  stopPipeline();
                }
                // Start is handled from PipelinePanel / PipelineControl
              }}
              disabled={!isRunning}
              className={`w-full py-2.5 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-all shadow-lg ${
                isRunning
                  ? 'bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/20 shadow-red-500/5'
                  : 'bg-zinc-800 text-zinc-500 border border-zinc-700/50 cursor-not-allowed'
              }`}
            >
              {isRunning ? <Pause size={16} /> : <Play size={16} fill="currentColor" />}
              {isRunning ? 'Stop Pipeline' : 'Idle'}
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
            {isPanelOpen ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
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
                isConsoleOpen
                  ? 'opacity-0 pointer-events-none'
                  : 'opacity-100 text-zinc-400 hover:text-white hover:scale-105'
              }`}
            >
              <Terminal size={14} className="text-indigo-400" /> View Logs
              {logs.length > 0 && (
                <span className="ml-1 px-1.5 py-0.5 bg-indigo-500/20 text-indigo-300 rounded-full text-[9px] font-bold">
                  {logs.length}
                </span>
              )}
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
                <span className="text-[10px] text-zinc-600 ml-2">{logs.length} lines</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => usePipelineStore.getState().clearLogs()}
                  className="text-[10px] text-zinc-500 hover:text-zinc-300 px-2 py-1 bg-white/5 rounded-md transition-colors"
                >
                  Clear
                </button>
                <button className="text-zinc-500 hover:text-zinc-300 p-1">
                  <Maximize2 size={14} />
                </button>
                <button
                  onClick={() => setIsConsoleOpen(false)}
                  className="text-zinc-500 hover:text-white p-1 bg-white/5 rounded-md"
                >
                  <ChevronRight size={14} className="rotate-90" />
                </button>
              </div>
            </div>
            <div
              ref={consoleRef}
              className="p-4 font-mono text-[11px] leading-relaxed text-zinc-400 flex-1 overflow-y-auto scrollbar-hide space-y-0.5"
            >
              {logs.length === 0 ? (
                <div className="text-zinc-600 italic">No logs yet. Start the pipeline to see output.</div>
              ) : (
                logs.map((log, i) => (
                  <div
                    key={i}
                    className={
                      log.level === 'error'
                        ? 'text-red-400'
                        : log.level === 'warn'
                        ? 'text-amber-400'
                        : log.level === 'stderr'
                        ? 'text-orange-300/70'
                        : 'text-zinc-400'
                    }
                  >
                    <span className="text-zinc-600">[{log.timestamp}]</span> {log.message}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      <StatusBar />
    </div>
  );
}
