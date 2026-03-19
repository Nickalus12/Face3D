import { useEffect, useRef, useState, useCallback } from 'react';
import Sidebar, { type ViewId } from './components/Sidebar';
import Viewer3D from './components/Viewer3D';
import { Gallery } from './components/Gallery';
import SensorPanel from './components/SensorPanel';
import CameraTrajectory from './components/CameraTrajectory';
import CompareView from './components/CompareView';
import SettingsPanel from './components/SettingsPanel';
import StatusBar from './components/StatusBar';
import KeyboardShortcuts from './components/KeyboardShortcuts';
import { ChevronRight, ChevronLeft, Terminal, Maximize2, Play, Pause } from 'lucide-react';
import useSessionStore from './store/sessionStore';
import usePipelineStore from './store/pipelineStore';
import { onPipelineLog, onPipelineComplete } from './lib/tauri';
import ToastContainer from './components/Toast';
import { DetailPanel } from './components/DetailPanel';

export default function App() {
  const [activeView, setActiveView] = useState<ViewId>('view');
  const [isPanelOpen, setIsPanelOpen] = useState(true);
  const [isConsoleOpen, setIsConsoleOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const consoleRef = useRef<HTMLDivElement>(null);

  const { fetchSessions } = useSessionStore();
  const {
    status,
    logs,
    fetchGpuInfo,
    stopPipeline,
  } = usePipelineStore();

  const isRunning = status === 'running' || status === 'stopping';

  // ── Bootstrap on mount ──────────────────────────────────────
  useEffect(() => {
    const init = async () => {
      await fetchSessions();
      await fetchGpuInfo();
      setIsLoading(false);
    };
    init();
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
      // Flash window title bar on completion
      if (document.title) {
        const original = document.title;
        document.title = 'Pipeline Complete!';
        setTimeout(() => { document.title = original; }, 3000);
      }
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

  // ── Keyboard shortcut handlers ──────────────────────────────
  const handleViewChange = useCallback((view: ViewId) => {
    setActiveView(view);
  }, []);

  const handleToggleConsole = useCallback(() => {
    setIsConsoleOpen((prev) => !prev);
  }, []);

  return (
    <div className="flex flex-col h-screen w-screen bg-[#0a0a0b] text-zinc-300 font-sans overflow-hidden selection:bg-indigo-500/30 noise-overlay">
      {/* Keyboard shortcuts listener */}
      <KeyboardShortcuts
        onViewChange={handleViewChange}
        onToggleConsole={handleToggleConsole}
      />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar activeView={activeView} onViewChange={setActiveView} />

        {/* Collapsible Detail Panel */}
        <div
          className={`relative transition-all duration-400 ease-[cubic-bezier(0.16,1,0.3,1)] bg-[#0e0e11] border-r border-zinc-800/40 flex flex-col z-20 shrink-0 ${
            isPanelOpen ? 'w-[280px] opacity-100' : 'w-0 opacity-0 overflow-hidden'
          }`}
        >
          <DetailPanel isExpanded={isPanelOpen} onToggle={() => setIsPanelOpen(!isPanelOpen)} />

          {/* Master Control */}
          <div className="p-4 border-t border-zinc-800/40 bg-zinc-900/20 shrink-0">
            <button
              onClick={() => {
                if (isRunning) {
                  stopPipeline();
                }
              }}
              disabled={!isRunning}
              className={`w-full py-2.5 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-all duration-200 shadow-lg active:scale-[0.98] ${
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
            className="absolute top-5 left-4 z-20 p-1.5 bg-[#111113]/80 backdrop-blur-xl border border-white/10 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800/80 active:scale-90 transition-all duration-150 shadow-lg shadow-black/20"
          >
            {isPanelOpen ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
          </button>

          {/* Main Content — switches on active view */}
          <div className="flex-1 relative rounded-tl-xl overflow-hidden border-t border-l border-white/5">
            {activeView === 'gallery' ? (
              <Gallery />
            ) : activeView === 'settings' ? (
              <SettingsPanel />
            ) : activeView === 'sensors' ? (
              <div className="h-full overflow-y-auto scrollbar-hide">
                <SensorPanel />
                <div className="px-6 pb-6">
                  <CameraTrajectory />
                </div>
              </div>
            ) : activeView === 'compare' ? (
              <CompareView />
            ) : (
              <Viewer3D />
            )}
          </div>

          {/* Console Trigger Button (when closed) */}
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20">
            <button
              onClick={() => setIsConsoleOpen(!isConsoleOpen)}
              className={`flex items-center gap-2 px-4 py-1.5 bg-[#111113]/90 backdrop-blur-md border border-white/10 rounded-full text-[11px] font-medium transition-all duration-200 shadow-xl shadow-black/20 active:scale-95 ${
                isConsoleOpen
                  ? 'opacity-0 pointer-events-none'
                  : 'opacity-100 text-zinc-400 hover:text-white hover:scale-105 hover:bg-zinc-800/80'
              }`}
            >
              <Terminal size={14} className="text-indigo-400" /> View Logs
              <kbd className="text-zinc-600 text-[9px] font-mono ml-1">Ctrl+`</kbd>
              {logs.length > 0 && (
                <span className="ml-1 px-1.5 py-0.5 bg-indigo-500/20 text-indigo-300 rounded-full text-[9px] font-bold tabular-nums">
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
                <span className="text-[10px] text-zinc-600 ml-2 tabular-nums">{logs.length} lines</span>
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
      <ToastContainer />
    </div>
  );
}
