import { useEffect, useRef, useState, useCallback, lazy, Suspense } from 'react';
import Sidebar, { type ViewId } from './components/Sidebar';
import StatusBar from './components/StatusBar';
import KeyboardShortcuts from './components/KeyboardShortcuts';
import TopBar from './components/TopBar';
import WelcomeScreen from './components/WelcomeScreen';

// Lazy-load heavy components (Three.js, recharts, etc.) — only load when needed
const Viewer3D = lazy(() => import('./components/Viewer3D'));
const Gallery = lazy(() => import('./components/Gallery').then(m => ({ default: m.Gallery })));
const SettingsPanel = lazy(() => import('./components/SettingsPanel'));
const TrainingMetrics = lazy(() => import('./components/TrainingMetrics').then(m => ({ default: m.TrainingMetrics })));
const PipelineView = lazy(() => import('./components/PipelineView'));
import {
  ChevronRight,
  ChevronLeft,
  Terminal,
  Maximize2,
  Play,
  Pause,
} from 'lucide-react';
import useSessionStore from './store/sessionStore';
import usePipelineStore from './store/pipelineStore';
import useToastStore from './store/toastStore';
import { bindToast } from './lib/api';
import { usePipelineEvents } from './hooks/usePipelineEvents';
import ToastContainer from './components/Toast';
import { DetailPanel } from './components/DetailPanel';
import NewScanWizard from './components/NewScanWizard';

export default function App() {
  const [activeView, setActiveView] = useState<ViewId>('view');
  const [isPanelOpen, setIsPanelOpen] = useState(true);
  const [panelWidth, setPanelWidth] = useState(240);
  const [isConsoleOpen, setIsConsoleOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [wizardOpen, setWizardOpen] = useState(false);
  const [prevView, setPrevView] = useState<ViewId | null>(null);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const consoleRef = useRef<HTMLDivElement>(null);
  const isResizingRef = useRef(false);

  const { fetchSessions, currentSession, selectSession, clearSession } = useSessionStore();
  const {
    status,
    logs,
    fetchGpuInfo,
    stopPipeline,
  } = usePipelineStore();
  const addToast = useToastStore((s) => s.addToast);

  const isRunning = status === 'running' || status === 'stopping';

  // ── Wire toast store into the IPC layer ───────────────────────
  useEffect(() => {
    bindToast(addToast);
  }, [addToast]);

  // ── Pipeline event listeners (typed, auto-cleanup) ────────────
  usePipelineEvents();

  // ── Bootstrap on mount ──────────────────────────────────────
  useEffect(() => {
    const init = async () => {
      await fetchSessions();
      await fetchGpuInfo();
      setIsLoading(false);
    };
    init();
  }, [fetchSessions, fetchGpuInfo]);

  // ── Auto-switch to pipeline view when running ──────────────
  useEffect(() => {
    if (isRunning && activeView !== 'pipeline') {
      handleViewChange('pipeline');
    }
  }, [isRunning]);

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
    if (view === activeView && view !== 'home') return;
    // Clicking "Sessions" / "home" should deselect current session → show welcome
    if (view === 'home') {
      clearSession();
    }
    setIsTransitioning(true);
    setPrevView(activeView);
    setTimeout(() => {
      setActiveView(view);
      setTimeout(() => {
        setIsTransitioning(false);
        setPrevView(null);
      }, 150);
    }, 10);
  }, [activeView, clearSession]);

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

  const handleNewScan = useCallback(() => {
    setWizardOpen(true);
  }, []);

  const handleSelectSessionFromWelcome = useCallback((session: any) => {
    selectSession(session);
    // Route to the best view that has actual content
    if (session.has_gaussians) {
      setActiveView('view');  // Has a 3D model — show it
    } else if (session.has_renders) {
      setActiveView('gallery');  // Has renders but no model — show gallery
    } else {
      setActiveView('pipeline');  // Nothing yet — show pipeline status
    }
  }, [selectSession]);

  // Determine if we should show the welcome screen
  const showWelcome = !currentSession && !isLoading;

  return (
    <div className="flex flex-col h-screen w-screen bg-surface text-zinc-300 font-sans overflow-hidden selection:bg-indigo-500/30 noise-overlay">
      {/* Keyboard shortcuts listener */}
      <KeyboardShortcuts
        onViewChange={handleViewChange}
        onToggleConsole={handleToggleConsole}
      />

      {/* Top toolbar */}
      <TopBar
        activeView={activeView}
        onViewChange={handleViewChange}
      />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar activeView={activeView} onViewChange={handleViewChange} />

        {/* Collapsible Detail Panel with draggable resize */}
        <div
          className={`relative bg-surface-raised border-r border-border flex flex-col z-20 shrink-0 transition-opacity duration-200 ease-out ${
            isPanelOpen ? 'opacity-100' : 'w-0 opacity-0 overflow-hidden'
          }`}
          style={isPanelOpen ? { width: `${panelWidth}px` } : undefined}
        >
          <DetailPanel isExpanded={isPanelOpen} onSelectSession={handleSelectSessionFromWelcome} />

          {/* Master Control */}
          <div className="p-3 border-t border-border bg-zinc-900/20 shrink-0">
            <button
              onClick={() => {
                if (isRunning) {
                  stopPipeline();
                }
              }}
              disabled={!isRunning}
              className={`w-full py-2.5 rounded-button text-xs font-semibold flex items-center justify-center gap-2 transition-all duration-200 active:scale-[0.98] ${
                isRunning
                  ? 'bg-red-500/10 text-red-400 hover:bg-red-500/15 border border-red-500/20 shadow-lg shadow-red-500/5'
                  : 'bg-zinc-800/60 text-zinc-500 border border-zinc-700/40 cursor-not-allowed'
              }`}
            >
              {isRunning ? <Pause size={14} /> : <Play size={14} fill="currentColor" />}
              {isRunning ? 'Stop Pipeline' : 'Idle'}
            </button>
          </div>

          {/* Resize handle -- draggable right edge */}
          {isPanelOpen && (
            <div
              onMouseDown={handleResizeStart}
              className="absolute top-0 right-0 w-1.5 h-full cursor-col-resize group z-30 flex items-center justify-center hover:bg-indigo-500/10 transition-colors duration-150"
            >
              <div className="w-px h-8 bg-zinc-700/50 group-hover:bg-indigo-500/50 transition-colors duration-150 rounded-full" />
            </div>
          )}
        </div>

        {/* Panel toggle — lives between panel and content, never overlaps either */}
        <button
          onClick={() => setIsPanelOpen(!isPanelOpen)}
          className="w-4 h-full shrink-0 flex items-center justify-center bg-surface hover:bg-zinc-800/60 border-r border-border text-zinc-700 hover:text-zinc-300 transition-all duration-200 cursor-pointer group"
          title={isPanelOpen ? 'Collapse panel (Ctrl+B)' : 'Expand panel (Ctrl+B)'}
        >
          <div className="transition-transform duration-200 group-hover:scale-110">
            {isPanelOpen ? <ChevronLeft size={11} /> : <ChevronRight size={11} />}
          </div>
        </button>

        {/* Main Center Area */}
        <div className="flex-1 relative flex flex-col min-w-0 bg-surface">
          {/* Main Content -- switches on active view with crossfade */}
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
              ) : (
                <Suspense fallback={
                  <div className="flex-1 flex items-center justify-center bg-[#0a0a0b]">
                    <div className="w-6 h-6 border-2 border-zinc-700 border-t-indigo-500 rounded-full animate-spin" />
                  </div>
                }>
                  {activeView === 'pipeline' ? (
                    <PipelineView />
                  ) : activeView === 'gallery' ? (
                    <Gallery />
                  ) : activeView === 'settings' ? (
                    <SettingsPanel />
                  ) : activeView === 'metrics' ? (
                    <TrainingMetrics />
                  ) : (
                    <Viewer3D />
                  )}
                </Suspense>
              )}
            </div>
          </div>

          {/* Console Trigger Button (when closed) */}
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20">
            <button
              onClick={() => setIsConsoleOpen(!isConsoleOpen)}
              className={`flex items-center gap-2 px-4 py-2 toolbar-glass rounded-full text-xs font-medium transition-all duration-300 active:scale-95 ${
                isConsoleOpen
                  ? 'opacity-0 pointer-events-none translate-y-2'
                  : 'opacity-100 text-zinc-400 hover:text-white hover:scale-[1.03]'
              }`}
            >
              <Terminal size={14} className="text-indigo-400" /> View Logs
              <kbd className="text-zinc-600 text-[10px] font-mono ml-1 px-1.5 py-0.5 bg-white/[0.03] rounded">Ctrl+`</kbd>
              {logs.length > 0 && (
                <span className="ml-1 px-1.5 py-0.5 bg-indigo-500/15 text-indigo-300 rounded-full text-[10px] font-bold tabular-nums">
                  {logs.length}
                </span>
              )}
            </button>
          </div>

          {/* Animated Bottom Console */}
          <div
            className={`absolute bottom-0 left-0 right-0 transition-all duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] z-30 flex flex-col bg-[#0b0b0d]/95 backdrop-blur-2xl border-t border-white/8 rounded-t-2xl shadow-[0_-20px_50px_rgba(0,0,0,0.6)] ${
              isConsoleOpen
                ? 'translate-y-0 opacity-100'
                : 'translate-y-full opacity-0'
            }`}
            style={{ height: '260px' }}
          >
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-white/5 shrink-0">
              <div className="flex items-center gap-2.5">
                <div className="flex items-center gap-1.5">
                  <div className="w-2.5 h-2.5 rounded-full bg-red-500/60" />
                  <div className="w-2.5 h-2.5 rounded-full bg-amber-500/60" />
                  <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/60" />
                </div>
                <span className="text-xs font-semibold tracking-wider text-zinc-400 ml-1">CONSOLE</span>
                <span className="text-[11px] text-zinc-600 tabular-nums">{logs.length} lines</span>
              </div>
              <div className="flex items-center gap-1.5">
                <button
                  onClick={() => usePipelineStore.getState().clearLogs()}
                  className="text-[11px] text-zinc-500 hover:text-zinc-300 px-2.5 py-1 bg-white/[0.03] hover:bg-white/[0.06] rounded-md transition-all duration-150 active:scale-95 border border-white/5"
                >
                  Clear
                </button>
                <button className="text-zinc-500 hover:text-zinc-300 p-1.5 hover:bg-white/[0.04] rounded-md transition-all duration-150 active:scale-90">
                  <Maximize2 size={13} />
                </button>
                <button
                  onClick={() => setIsConsoleOpen(false)}
                  className="text-zinc-500 hover:text-white p-1.5 bg-white/[0.03] hover:bg-white/[0.06] rounded-md transition-all duration-150 active:scale-90 border border-white/5"
                >
                  <ChevronRight size={13} className="rotate-90" />
                </button>
              </div>
            </div>
            <div
              ref={consoleRef}
              className="px-4 py-3 font-mono text-xs leading-[1.7] text-zinc-400 flex-1 overflow-y-auto scrollbar-hide space-y-px"
            >
              {logs.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-zinc-600">
                  <Terminal size={20} className="mb-2 opacity-30" />
                  <span className="text-xs">No logs yet. Start the pipeline to see output.</span>
                </div>
              ) : (
                logs.map((log, i) => {
                  const isStageTransition = log.message.match(/(?:===\s*Stage|Running\s+stage|Stage\s+\d+\s+completed)/i);
                  const isStageNameLine = log.message.match(/^Stage\s+\d+[:\s]/i);
                  const isError = log.level === 'error';
                  const isSuccess = log.message.match(/(?:complete|success|finished|done)/i) && !isError;

                  if (isStageTransition) {
                    return (
                      <div key={i} className="my-2">
                        <div className="h-px bg-gradient-to-r from-transparent via-emerald-500/25 to-transparent" />
                        <div className="text-emerald-400 font-bold text-[11px] tracking-wider uppercase py-1 flex items-center gap-2">
                          <div className="w-1 h-1 rounded-full bg-emerald-400" />
                          {log.message}
                        </div>
                      </div>
                    );
                  }

                  const shortTimestamp = log.timestamp.slice(0, 8);

                  return (
                    <div
                      key={i}
                      className={`flex items-start gap-2 py-px rounded-sm transition-colors duration-100 hover:bg-white/[0.01] ${
                        isError
                          ? 'border-l-2 border-red-500/40 pl-2 bg-red-500/[0.02]'
                          : ''
                      }`}
                    >
                      <span className="text-zinc-700 text-[11px] shrink-0 tabular-nums select-none w-[52px]">{shortTimestamp}</span>
                      <span className={`flex-1 ${
                        isError
                          ? 'text-red-400'
                          : log.level === 'warn'
                          ? 'text-amber-400/90'
                          : log.level === 'stderr'
                          ? 'text-orange-300/60'
                          : isStageNameLine
                          ? 'text-emerald-300 font-semibold'
                          : isSuccess
                          ? 'text-emerald-400/80'
                          : 'text-zinc-400'
                      }`}>
                        {log.message}
                      </span>
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
