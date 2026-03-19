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
import TopBar from './components/TopBar';
import WelcomeScreen from './components/WelcomeScreen';
import { TrainingMetrics } from './components/TrainingMetrics';
import {
  ChevronRight,
  ChevronLeft,
  Terminal,
  Maximize2,
  Play,
  Pause,
  GripVertical,
} from 'lucide-react';
import useSessionStore from './store/sessionStore';
import usePipelineStore from './store/pipelineStore';
import { onPipelineLog, onPipelineComplete, openFolder } from './lib/tauri';
import ToastContainer from './components/Toast';
import { DetailPanel } from './components/DetailPanel';
import NewScanWizard from './components/NewScanWizard';

export default function App() {
  const [activeView, setActiveView] = useState<ViewId>('view');
  const [isPanelOpen, setIsPanelOpen] = useState(true);
  const [panelWidth, setPanelWidth] = useState(280);
  const [isConsoleOpen, setIsConsoleOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [wizardOpen, setWizardOpen] = useState(false);
  const [prevView, setPrevView] = useState<ViewId | null>(null);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const consoleRef = useRef<HTMLDivElement>(null);
  const isResizingRef = useRef(false);

  const { fetchSessions, currentSession, selectSession } = useSessionStore();
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
      useSessionStore.getState().fetchSessions();
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

  // ── View change with crossfade transition ───────────────────
  const handleViewChange = useCallback((view: ViewId) => {
    if (view === activeView) return;
    setIsTransitioning(true);
    setPrevView(activeView);
    setTimeout(() => {
      setActiveView(view);
      setTimeout(() => {
        setIsTransitioning(false);
        setPrevView(null);
      }, 150);
    }, 10);
  }, [activeView]);

  const handleToggleConsole = useCallback(() => {
    setIsConsoleOpen((prev) => !prev);
  }, []);

  // ── Draggable panel resize ──────────────────────────────────
  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isResizingRef.current = true;
    const startX = e.clientX;
    const startWidth = panelWidth;

    const onMouseMove = (moveEvent: MouseEvent) => {
      if (!isResizingRef.current) return;
      const delta = moveEvent.clientX - startX;
      const newWidth = Math.max(220, Math.min(startWidth + delta, 450));
      setPanelWidth(newWidth);
    };

    const onMouseUp = () => {
      isResizingRef.current = false;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, [panelWidth]);

  const handleOpenProjectFolder = useCallback(() => {
    openFolder('data/output');
  }, []);

  const handleNewScan = useCallback(() => {
    setWizardOpen(true);
  }, []);

  const handleSelectSessionFromWelcome = useCallback((session: any) => {
    selectSession(session);
    setActiveView('view');
  }, [selectSession]);

  // Determine if we should show the welcome screen
  const showWelcome = !currentSession && !isLoading;

  return (
    <div className="flex flex-col h-screen w-screen bg-[#0a0a0b] text-zinc-300 font-sans overflow-hidden selection:bg-indigo-500/30 noise-overlay">
      {/* Keyboard shortcuts listener */}
      <KeyboardShortcuts
        onViewChange={handleViewChange}
        onToggleConsole={handleToggleConsole}
      />

      {/* Top toolbar */}
      <TopBar
        activeView={activeView}
        onViewChange={handleViewChange}
        onNewScan={handleNewScan}
        onOpenFolder={handleOpenProjectFolder}
      />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar activeView={activeView} onViewChange={handleViewChange} />

        {/* Collapsible Detail Panel with draggable resize */}
        <div
          className={`relative bg-[#0e0e11] border-r border-zinc-800/40 flex flex-col z-20 shrink-0 transition-opacity duration-200 ease-out ${
            isPanelOpen ? 'opacity-100' : 'w-0 opacity-0 overflow-hidden'
          }`}
          style={isPanelOpen ? { width: `${panelWidth}px` } : undefined}
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

          {/* Resize handle — draggable right edge */}
          {isPanelOpen && (
            <div
              onMouseDown={handleResizeStart}
              className="absolute top-0 right-0 w-1.5 h-full cursor-col-resize group z-30 flex items-center justify-center hover:bg-indigo-500/10 transition-colors duration-150"
            >
              <div className="w-px h-8 bg-zinc-700/50 group-hover:bg-indigo-500/50 transition-colors duration-150 rounded-full" />
            </div>
          )}
        </div>

        {/* Main Center Area */}
        <div className="flex-1 relative flex flex-col min-w-0 bg-[#0a0a0b]">
          {/* Panel Toggle button */}
          <button
            onClick={() => setIsPanelOpen(!isPanelOpen)}
            className="absolute top-3 left-3 z-20 p-1.5 bg-[#111113]/80 backdrop-blur-xl border border-white/10 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800/80 active:scale-90 transition-all duration-150 shadow-lg shadow-black/20"
          >
            {isPanelOpen ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
          </button>

          {/* Main Content — switches on active view with crossfade */}
          <div className="flex-1 relative overflow-hidden">
            <div
              className={`absolute inset-0 transition-opacity duration-150 ease-out ${
                isTransitioning ? 'opacity-0' : 'opacity-100'
              }`}
            >
              {showWelcome ? (
                <WelcomeScreen
                  onNewScan={handleNewScan}
                  onSelectSession={handleSelectSessionFromWelcome}
                />
              ) : activeView === 'gallery' ? (
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
              ) : activeView === 'metrics' ? (
                <TrainingMetrics />
              ) : (
                <Viewer3D />
              )}
            </div>
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
            className={`absolute bottom-0 left-0 right-0 transition-all duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] z-30 flex flex-col bg-[#0d0d0f]/95 backdrop-blur-2xl border-t border-white/10 rounded-t-2xl shadow-[0_-20px_40px_rgba(0,0,0,0.5)] ${
              isConsoleOpen
                ? 'translate-y-0 opacity-100'
                : 'translate-y-full opacity-0'
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
                  className="text-[10px] text-zinc-500 hover:text-zinc-300 px-2 py-1 bg-white/5 hover:bg-white/[0.08] rounded-md transition-all duration-150 active:scale-95"
                >
                  Clear
                </button>
                <button className="text-zinc-500 hover:text-zinc-300 p-1 hover:bg-white/5 rounded-md transition-all duration-150 active:scale-90">
                  <Maximize2 size={14} />
                </button>
                <button
                  onClick={() => setIsConsoleOpen(false)}
                  className="text-zinc-500 hover:text-white p-1 bg-white/5 hover:bg-white/[0.08] rounded-md transition-all duration-150 active:scale-90"
                >
                  <ChevronRight size={14} className="rotate-90" />
                </button>
              </div>
            </div>
            <div
              ref={consoleRef}
              className="p-4 font-mono text-[11px] leading-6 text-zinc-400 flex-1 overflow-y-auto scrollbar-hide space-y-0.5"
            >
              {logs.length === 0 ? (
                <div className="text-zinc-600 italic">No logs yet. Start the pipeline to see output.</div>
              ) : (
                logs.map((log, i) => {
                  const isStageTransition = log.message.match(/(?:===\s*Stage|Running\s+stage|Stage\s+\d+\s+completed)/i);
                  const isStageNameLine = log.message.match(/^Stage\s+\d+[:\s]/i);
                  const isError = log.level === 'error';

                  if (isStageTransition) {
                    return (
                      <div key={i} className="my-1.5">
                        <div className="h-px bg-gradient-to-r from-transparent via-emerald-500/20 to-transparent" />
                        <div className="text-emerald-400 font-bold text-[10px] tracking-wider uppercase py-0.5">
                          {log.message}
                        </div>
                      </div>
                    );
                  }

                  // Format timestamp to HH:MM:SS
                  const shortTimestamp = log.timestamp.slice(0, 8);

                  return (
                    <div
                      key={i}
                      className={`${
                        isError
                          ? 'text-red-400 border-l-2 border-red-500/30 pl-2'
                          : log.level === 'warn'
                          ? 'text-amber-400'
                          : log.level === 'stderr'
                          ? 'text-orange-300/70'
                          : isStageNameLine
                          ? 'text-emerald-300 font-bold'
                          : 'text-zinc-400'
                      }`}
                    >
                      <span className="text-zinc-700 opacity-60 text-[10px]">[{shortTimestamp}]</span> {log.message}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      </div>

      <StatusBar />
      <ToastContainer />

      {/* New Scan Wizard Modal */}
      <NewScanWizard isOpen={wizardOpen} onClose={() => setWizardOpen(false)} />
    </div>
  );
}
