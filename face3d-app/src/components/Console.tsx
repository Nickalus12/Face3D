import React, { useState, useRef, useEffect } from 'react';
import { Terminal, Search, Lock, Unlock, Copy, GripHorizontal, Trash2 } from 'lucide-react';
import usePipelineStore from '../store/pipelineStore';

type LogFilter = 'all' | 'info' | 'warn' | 'error' | 'stderr';

export const Console: React.FC = () => {
  const [autoScroll, setAutoScroll] = useState(true);
  const [height, setHeight] = useState(300);
  const [filter, setFilter] = useState<LogFilter>('all');
  const [search, setSearch] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  const { logs, clearLogs } = usePipelineStore();

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

  const handleCopyAll = () => {
    const allText = filteredLogs
      .map((l) => `[${l.timestamp}] [${l.level.toUpperCase()}] ${l.message}`)
      .join('\n');
    navigator.clipboard.writeText(allText);
  };

  const FILTERS: { label: string; value: LogFilter }[] = [
    { label: 'All', value: 'all' },
    { label: 'Info', value: 'info' },
    { label: 'Warn', value: 'warn' },
    { label: 'Error', value: 'error' },
    { label: 'Stderr', value: 'stderr' },
  ];

  return (
    <div
      style={{ height: `${height}px` }}
      className="fixed bottom-0 left-0 right-0 bg-[#050505] border-t border-zinc-800 flex flex-col z-40 shadow-[0_-10px_40px_rgba(0,0,0,0.5)]"
    >
      {/* Drag Handle */}
      <div
        className="h-1.5 w-full bg-zinc-900 hover:bg-emerald-500/50 cursor-row-resize transition-colors flex items-center justify-center group"
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
        <GripHorizontal
          size={12}
          className="text-zinc-600 group-hover:text-zinc-300 opacity-0 group-hover:opacity-100 transition-opacity"
        />
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
                className={`px-2 py-0.5 text-[10px] rounded-md transition-colors ${
                  filter === f.value
                    ? 'bg-zinc-800 text-zinc-200'
                    : 'text-zinc-600 hover:text-zinc-400'
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
              placeholder="Filter logs..."
              className="bg-zinc-900 border border-zinc-800 rounded-md pl-8 pr-3 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-emerald-500/50 w-40"
            />
          </div>

          {/* Auto-scroll toggle */}
          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={`p-1.5 rounded-md border transition-colors flex items-center gap-1.5 text-xs ${
              autoScroll
                ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                : 'bg-zinc-900 border-zinc-800 text-zinc-500 hover:text-zinc-300'
            }`}
          >
            {autoScroll ? <Lock size={12} /> : <Unlock size={12} />}
          </button>

          {/* Copy all */}
          <button
            onClick={handleCopyAll}
            className="p-1.5 rounded-md border border-zinc-800 text-zinc-500 hover:text-zinc-300 transition-colors"
            title="Copy all logs"
          >
            <Copy size={12} />
          </button>

          {/* Clear */}
          <button
            onClick={clearLogs}
            className="p-1.5 rounded-md border border-zinc-800 text-zinc-500 hover:text-red-400 transition-colors"
            title="Clear logs"
          >
            <Trash2 size={12} />
          </button>
        </div>
      </div>

      {/* Logs Area */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto font-mono text-xs py-2 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:bg-zinc-800 [&::-webkit-scrollbar-track]:bg-transparent"
      >
        {filteredLogs.length === 0 ? (
          <div className="px-4 py-8 text-center text-zinc-600 italic">
            {logs.length === 0
              ? 'No logs yet. Start the pipeline to see output.'
              : 'No logs match the current filter.'}
          </div>
        ) : (
          filteredLogs.map((log, index) => {
            // Stage transition detection for visual dividers
            const isStageTransition =
              log.message.match(/(?:===\s*Stage|Running\s+stage|Stage\s+\d+\s+completed)/i);

            if (isStageTransition) {
              return (
                <div
                  key={index}
                  className="w-full my-2 bg-emerald-500/10 border-y border-emerald-500/20 py-1.5 px-4 flex items-center"
                >
                  <span className="text-emerald-400 font-bold uppercase tracking-widest text-[10px]">
                    {log.message}
                  </span>
                </div>
              );
            }

            return (
              <div
                key={index}
                className="group flex items-start px-4 py-0.5 hover:bg-zinc-900/50 transition-colors"
              >
                <div className="w-10 text-zinc-700 select-none text-right pr-4 border-r border-zinc-800/50 mr-4 shrink-0">
                  {index + 1}
                </div>
                <div className="w-20 text-zinc-500 shrink-0 select-none">{log.timestamp}</div>
                <div
                  className={`w-16 shrink-0 font-bold ${
                    log.level === 'error'
                      ? 'text-red-400'
                      : log.level === 'warn'
                      ? 'text-amber-400'
                      : log.level === 'stderr'
                      ? 'text-orange-400'
                      : 'text-zinc-400'
                  }`}
                >
                  [{log.level.toUpperCase()}]
                </div>
                <div
                  className={`flex-1 break-all pr-4 ${
                    log.level === 'error' ? 'text-red-300' : 'text-zinc-300'
                  }`}
                >
                  {log.message}
                </div>
                <button
                  onClick={() => handleCopy(log.message)}
                  className="opacity-0 group-hover:opacity-100 p-1 text-zinc-500 hover:text-zinc-200 transition-all"
                >
                  <Copy size={12} />
                </button>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
