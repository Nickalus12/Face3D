import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { Terminal, Search, Lock, Unlock, Copy, Trash2, ClipboardCopy } from 'lucide-react';
import usePipelineStore from '../store/pipelineStore';

type LogFilter = 'all' | 'info' | 'warn' | 'error' | 'stderr';

// ── Highlight search matches in text ──────────────────────────

const HighlightedText: React.FC<{ text: string; search: string }> = ({ text, search }) => {
  if (!search.trim()) return <>{text}</>;

  const regex = new RegExp(`(${search.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
  const parts = text.split(regex);

  return (
    <>
      {parts.map((part, i) =>
        regex.test(part) ? (
          <mark key={i} className="bg-amber-500/30 text-amber-200 rounded-sm px-0.5">
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </>
  );
};

export const Console: React.FC = () => {
  const [autoScroll, setAutoScroll] = useState(true);
  const [height, setHeight] = useState(300);
  const [filter, setFilter] = useState<LogFilter>('all');
  const [search, setSearch] = useState('');
  const [copiedAll, setCopiedAll] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const { logs, clearLogs, status } = usePipelineStore();
  const isComplete = status === 'complete';

  // Filter logs
  const filteredLogs = logs.filter((log) => {
    if (filter !== 'all' && log.level !== filter) return false;
    if (search && !log.message.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  // Auto-scroll on new logs
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [filteredLogs.length, autoScroll]);

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const handleCopyAll = useCallback(() => {
    const allText = filteredLogs
      .map((l) => `[${l.timestamp.slice(0, 8)}] [${l.level.toUpperCase()}] ${l.message}`)
      .join('\n');
    navigator.clipboard.writeText(allText);
    setCopiedAll(true);
    setTimeout(() => setCopiedAll(false), 2000);
  }, [filteredLogs]);

  // Log rate sparkline (lines/sec over last 30 seconds)
  const sparklineData = useMemo(() => {
    if (logs.length < 2) return [];
    const buckets: number[] = new Array(30).fill(0);
    const recentLogs = logs.slice(-300);
    const bucketSize = Math.max(1, Math.ceil(recentLogs.length / 30));
    for (let i = 0; i < 30; i++) {
      const start = i * bucketSize;
      const end = Math.min(start + bucketSize, recentLogs.length);
      buckets[i] = end - start;
    }
    const max = Math.max(...buckets, 1);
    return buckets.map((v) => v / max);
  }, [logs.length]);

  const FILTERS: { label: string; value: LogFilter; color: string }[] = [
    { label: 'All', value: 'all', color: '' },
    { label: 'Info', value: 'info', color: '' },
    { label: 'Warn', value: 'warn', color: 'text-amber-400' },
    { label: 'Error', value: 'error', color: 'text-red-400' },
    { label: 'Stderr', value: 'stderr', color: 'text-orange-400' },
  ];

  return (
    <div
      style={{ height: `${height}px` }}
      className="fixed bottom-0 left-0 right-0 bg-[#050505] border-t border-zinc-800/60 flex flex-col z-40 shadow-[0_-10px_40px_rgba(0,0,0,0.5)]"
    >
      {/* Drag Handle with grip dots */}
      <div
        className="h-2 w-full bg-zinc-900/80 hover:bg-indigo-500/20 cursor-row-resize transition-colors flex items-center justify-center group"
        onMouseDown={(e) => {
          const startY = e.clientY;
          const startHeight = height;
          const onMouseMove = (moveEvent: MouseEvent) => {
            const newHeight = startHeight + (startY - moveEvent.clientY);
            setHeight(Math.max(100, Math.min(newHeight, 800)));
          };
          const onMouseUp = () => {
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
          };
          document.addEventListener('mousemove', onMouseMove);
          document.addEventListener('mouseup', onMouseUp);
        }}
      >
        <div className="grip-dots">
          <span />
        </div>
      </div>

      {/* Toolbar */}
      <div className="h-10 border-b border-zinc-800/50 bg-[#0a0a0b] flex items-center justify-between px-4">
        <div className="flex items-center gap-3">
          <Terminal size={14} className="text-emerald-500" />
          <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
            System Console
          </span>
          <span className="text-[10px] text-zinc-600">
            {filteredLogs.length}/{logs.length}
          </span>
        </div>

        <div className="flex items-center gap-3">
          {/* Filter tabs */}
          <div className="flex gap-1">
            {FILTERS.map((f) => (
              <button
                key={f.value}
                onClick={() => setFilter(f.value)}
                className={`px-2 py-0.5 text-[10px] rounded-md transition-all duration-150 ${
                  filter === f.value
                    ? 'bg-zinc-800 text-zinc-200 shadow-sm'
                    : `text-zinc-600 hover:text-zinc-400 hover:bg-white/[0.03] ${f.color}`
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>

          {/* Search */}
          <div className="relative">
            <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-zinc-500" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search logs..."
              className="bg-zinc-900/80 border border-zinc-800 rounded-md pl-8 pr-3 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:outline-none focus:ring-1 focus:ring-indigo-500/50 focus:border-indigo-500/30 w-40 transition-all"
            />
          </div>

          {/* Auto-scroll toggle */}
          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={`p-1.5 rounded-md border transition-all duration-150 flex items-center gap-1.5 text-xs active:scale-90 ${
              autoScroll
                ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                : 'bg-zinc-900 border-zinc-800 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/80'
            }`}
          >
            {autoScroll ? <Lock size={12} /> : <Unlock size={12} />}
          </button>

          {/* Copy all — with feedback */}
          <button
            onClick={handleCopyAll}
            className={`p-1.5 rounded-md border transition-all duration-150 active:scale-90 ${
              copiedAll
                ? 'border-emerald-500/30 text-emerald-400 bg-emerald-500/10'
                : 'border-zinc-800 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/80'
            }`}
            title="Copy all filtered logs"
          >
            <ClipboardCopy size={12} />
          </button>

          {/* Clear */}
          <button
            onClick={clearLogs}
            className="p-1.5 rounded-md border border-zinc-800 text-zinc-500 hover:text-red-400 hover:bg-red-500/5 hover:border-red-500/20 active:scale-90 transition-all duration-150"
            title="Clear logs"
          >
            <Trash2 size={12} />
          </button>
        </div>
      </div>

      {/* Logs Area */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto font-mono text-xs py-2 scrollbar-hide"
      >
        {filteredLogs.length === 0 ? (
          <div className="px-4 py-8 text-center text-zinc-600 italic">
            {logs.length === 0
              ? 'No logs yet. Start the pipeline to see output.'
              : 'No logs match the current filter.'}
          </div>
        ) : (
          filteredLogs.map((log, index) => {
            // Stage transition detection
            const isStageTransition = log.message.match(
              /(?:===\s*Stage|Running\s+stage|Stage\s+\d+\s+completed)/i,
            );

            const isStageNameLine = log.message.match(/^Stage\s+\d+[:\s]/i);
            const isError = log.level === 'error';

            // Format timestamp to HH:MM:SS
            const shortTimestamp = log.timestamp.slice(0, 8);

            if (isStageTransition) {
              return (
                <div key={index} className="w-full my-2">
                  <div className="h-px w-full bg-gradient-to-r from-transparent via-emerald-500/30 to-transparent" />
                  <div className="bg-emerald-500/[0.06] py-1.5 px-4 flex items-center">
                    <span className="text-emerald-400 font-bold uppercase tracking-widest text-[10px]">
                      {log.message}
                    </span>
                  </div>
                  <div className="h-px w-full bg-gradient-to-r from-transparent via-emerald-500/30 to-transparent" />
                </div>
              );
            }

            return (
              <div
                key={index}
                className={`group flex items-start px-4 leading-6 hover:bg-white/[0.02] transition-colors ${
                  isError ? 'border-l-2 border-red-500/40 bg-red-500/[0.02]' : ''
                }`}
              >
                {/* Line number */}
                <div className="w-10 text-zinc-700 select-none text-right pr-4 border-r border-zinc-800/30 mr-4 shrink-0 font-mono tabular-nums">
                  {index + 1}
                </div>
                {/* Timestamp — HH:MM:SS format */}
                <div className="w-16 text-zinc-600 shrink-0 select-none opacity-60 text-[10px]">
                  {shortTimestamp}
                </div>
                {/* Level badge */}
                <div
                  className={`w-16 shrink-0 font-bold ${
                    log.level === 'error'
                      ? 'text-red-400'
                      : log.level === 'warn'
                      ? 'text-amber-400'
                      : log.level === 'stderr'
                      ? 'text-orange-400'
                      : 'text-zinc-500'
                  }`}
                >
                  [{log.level.toUpperCase()}]
                </div>
                {/* Message — with search highlighting */}
                <div
                  className={`flex-1 break-all pr-4 ${
                    isError
                      ? 'text-red-300'
                      : isStageNameLine
                      ? 'text-emerald-300 font-bold'
                      : 'text-zinc-300'
                  }`}
                >
                  <HighlightedText text={log.message} search={search} />
                </div>
                {/* Copy button */}
                <button
                  onClick={() => handleCopy(log.message)}
                  className="opacity-0 group-hover:opacity-100 p-1 text-zinc-600 hover:text-zinc-200 active:scale-90 transition-all duration-150"
                >
                  <Copy size={12} />
                </button>
              </div>
            );
          })
        )}

        {/* Pipeline Complete Banner */}
        {isComplete && logs.length > 0 && (
          <div className="mx-4 my-4 relative overflow-hidden rounded-xl border border-emerald-500/20 bg-emerald-500/[0.05] p-4 animate-scaleIn">
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
              {[...Array(12)].map((_, i) => (
                <div
                  key={i}
                  className="absolute w-1.5 h-1.5 rounded-full"
                  style={{
                    left: `${8 + i * 8}%`,
                    top: `${20 + (i % 3) * 25}%`,
                    background: ['#10b981', '#6366f1', '#f59e0b', '#3b82f6', '#a78bfa', '#34d399'][i % 6],
                    opacity: 0.6,
                    animation: `confetti-dot 2s ease-out ${i * 0.1}s infinite`,
                  }}
                />
              ))}
            </div>
            <div className="relative text-center">
              <div className="text-emerald-400 font-bold text-sm tracking-wide">Pipeline Complete</div>
              <div className="text-zinc-500 text-[10px] mt-1">All 14 stages finished successfully</div>
            </div>
          </div>
        )}
      </div>

      {/* Sparkline log rate bar at bottom */}
      {sparklineData.length > 0 && (
        <div className="h-5 border-t border-zinc-800/30 bg-[#0a0a0b] flex items-end px-4 gap-px">
          <span className="text-[8px] text-zinc-700 mr-2 self-center shrink-0">LOG RATE</span>
          {sparklineData.map((v, i) => (
            <div
              key={i}
              className="flex-1 bg-indigo-500/30 rounded-t-sm sparkline-bar"
              style={{
                height: `${Math.max(1, v * 12)}px`,
                animationDelay: `${i * 10}ms`,
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
};
