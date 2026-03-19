import React, { useState, useEffect } from 'react';
import {
  Play, Square, UploadCloud,
  TerminalSquare, Sparkles, ChevronUp
} from 'lucide-react';
import usePipelineStore from '../store/pipelineStore';
import useSessionStore from '../store/sessionStore';
import { selectContentDir } from '../lib/tauri';

export const PipelinePanel: React.FC = () => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const {
    status,
    currentStage,
    stages,
    logs,
    contentDir,
    sessionName,
    startedAt,
    setContentDir,
    setSessionName,
    startPipeline,
    stopPipeline,
  } = usePipelineStore();

  const { createSession } = useSessionStore();

  const isRunning = status === 'running' || status === 'stopping';
  const completedStages = stages.filter((s) => s.status === 'complete').length;
  const percent = Math.round((completedStages / stages.length) * 100);
  const radius = 28;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percent / 100) * circumference;

  // Elapsed timer
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
    return `${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
  };

  // Last few log lines
  const recentLogs = logs.slice(-4);

  async function handleBrowse() {
    const dir = await selectContentDir();
    if (dir) {
      setContentDir(dir);
    }
  }

  async function handleStart() {
    if (!contentDir) return;
    const name = sessionName.trim() || `session_${Date.now()}`;
    if (sessionName.trim()) {
      createSession(name, contentDir);
    }
    await startPipeline(contentDir, name);
  }

  async function handleStop() {
    await stopPipeline();
  }

  function handleAutoName() {
    const date = new Date();
    const ts = date.toISOString().slice(0, 10).replace(/-/g, '');
    const idx = String(Math.floor(Math.random() * 999)).padStart(3, '0');
    setSessionName(`scan_${ts}_${idx}`);
  }

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
          {isRunning && (
            <span className="ml-1 px-1.5 py-0.5 bg-emerald-500/10 text-emerald-400 text-[9px] font-bold rounded-full animate-pulse">
              LIVE
            </span>
          )}
        </div>
        <button onClick={() => setIsCollapsed(true)} className="text-zinc-500 hover:text-zinc-300 transition-colors">
          <ChevronUp size={16} />
        </button>
      </div>

      <div className="p-5 space-y-6">

        {/* Input Configuration */}
        <div className="space-y-3">
          <label className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Input Directory</label>
          <div className="relative group cursor-pointer" onClick={isRunning ? undefined : handleBrowse}>
            <div className="absolute inset-0 bg-zinc-800/50 rounded-xl group-hover:bg-zinc-800 transition-colors" />
            <div className={`relative border border-dashed rounded-xl p-4 flex flex-col items-center justify-center gap-2 text-center transition-colors ${
              contentDir
                ? 'border-emerald-500/30 bg-emerald-500/5'
                : 'border-zinc-700 group-hover:border-emerald-500/50'
            } ${isRunning ? 'opacity-50 cursor-not-allowed' : ''}`}>
              <UploadCloud size={20} className={contentDir ? 'text-emerald-400' : 'text-zinc-400 group-hover:text-emerald-400 transition-colors'} />
              {contentDir ? (
                <p className="text-xs text-emerald-300 truncate max-w-full px-2" title={contentDir}>
                  {contentDir.split(/[/\\]/).slice(-2).join('/')}
                </p>
              ) : (
                <p className="text-xs text-zinc-400 group-hover:text-zinc-300">
                  Click to <span className="text-emerald-400">browse</span> for content directory
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Session Name */}
        <div className="space-y-3">
          <label className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Session Name</label>
          <div className="flex gap-2">
            <input
              type="text"
              value={sessionName}
              onChange={(e) => setSessionName(e.target.value)}
              placeholder="e.g. bust_scan_01"
              disabled={isRunning}
              className="flex-1 bg-zinc-900/50 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/50 transition-all disabled:opacity-50"
            />
            <button
              onClick={handleAutoName}
              disabled={isRunning}
              className="p-2 bg-zinc-900/50 border border-zinc-800 rounded-lg text-zinc-400 hover:text-emerald-400 hover:border-emerald-500/50 transition-colors disabled:opacity-50"
              title="Auto-generate session name"
            >
              <Sparkles size={16} />
            </button>
          </div>
        </div>

        {/* Controls & Progress */}
        <div className="p-4 bg-zinc-900/40 border border-zinc-800 rounded-xl flex items-center justify-between">
          <div className="flex gap-2">
            {!isRunning ? (
              <button
                onClick={handleStart}
                disabled={!contentDir}
                className={`p-3 rounded-lg flex items-center justify-center transition-all ${
                  contentDir
                    ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20 hover:bg-emerald-500/20'
                    : 'bg-zinc-800 text-zinc-600 border border-zinc-700 cursor-not-allowed'
                }`}
              >
                <Play size={18} fill="currentColor" />
              </button>
            ) : (
              <button
                onClick={handleStop}
                className="p-3 rounded-lg bg-red-500/10 text-red-500 border border-red-500/20 hover:bg-red-500/20 transition-all"
              >
                <Square size={18} fill="currentColor" />
              </button>
            )}
          </div>

          <div className="flex items-center gap-3">
            <div className="text-right">
              <div className="text-xs text-zinc-400">
                {isRunning
                  ? `Stage ${currentStage}/14`
                  : status === 'complete'
                  ? 'Complete'
                  : status === 'error'
                  ? 'Error'
                  : 'Ready'}
              </div>
              <div className={`text-sm font-medium ${
                isRunning ? 'text-emerald-400 animate-pulse' : status === 'error' ? 'text-red-400' : 'text-zinc-500'
              }`}>
                {isRunning
                  ? formatElapsed(elapsed)
                  : status === 'complete'
                  ? 'Done'
                  : status === 'error'
                  ? 'Failed'
                  : 'Idle'}
              </div>
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
                  strokeLinecap="round"
                  className={`transition-all duration-500 ease-out ${
                    status === 'complete' ? 'text-blue-500' : status === 'error' ? 'text-red-500' : 'text-emerald-500'
                  }`}
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center text-[11px] font-bold text-zinc-200">
                {percent}%
              </div>
            </div>
          </div>
        </div>

        {/* Mini Log */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-zinc-500">
            <span className="font-semibold uppercase tracking-wider">Recent Logs</span>
            <span>{isRunning && startedAt ? formatElapsed(elapsed) : '--:--'} elapsed</span>
          </div>
          <div className="bg-[#050505] border border-zinc-800/80 rounded-lg p-3 font-mono text-[11px] text-zinc-400 space-y-1 h-20 overflow-hidden relative">
            {recentLogs.length === 0 ? (
              <p className="text-zinc-600 italic">Waiting for pipeline output...</p>
            ) : (
              recentLogs.map((log, i) => (
                <p
                  key={i}
                  className={
                    log.level === 'error'
                      ? 'text-red-400'
                      : log.level === 'warn'
                      ? 'text-amber-400'
                      : i === recentLogs.length - 1
                      ? 'text-emerald-400'
                      : 'text-zinc-400'
                  }
                >
                  <span className={
                    log.level === 'error'
                      ? 'text-red-500/50'
                      : log.level === 'warn'
                      ? 'text-amber-500/50'
                      : 'text-zinc-600'
                  }>
                    [{log.level.toUpperCase()}]
                  </span>{' '}
                  {log.message.length > 60 ? log.message.slice(0, 60) + '...' : log.message}
                </p>
              ))
            )}
            <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-[#050505] to-transparent pointer-events-none" />
          </div>
        </div>

      </div>
    </div>
  );
};
