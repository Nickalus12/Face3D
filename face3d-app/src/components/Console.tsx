import React, { useState, useRef, useEffect } from 'react';
import { Terminal, Search, Lock, Unlock, Copy, GripHorizontal } from 'lucide-react';

const mockLogs = [
  { id: 1, level: 'INFO', time: '14:22:01', msg: 'Initializing 3D Gaussian Splatting engine v2.4.1...' },
  { id: 2, level: 'INFO', time: '14:22:02', msg: 'Loading configuration from config.yaml' },
  { id: 3, transition: true, msg: 'Stage 1: Feature Extraction' },
  { id: 4, level: 'INFO', time: '14:22:05', msg: 'Found 142 valid images in input directory.' },
  { id: 5, level: 'WARN', time: '14:22:06', msg: 'Image IMG_0042.jpg has missing EXIF data. Estimating focal length.' },
  { id: 6, level: 'INFO', time: '14:22:15', msg: 'Extracted 1.4M SIFT features across all images.' },
  { id: 7, transition: true, msg: 'Stage 2: Feature Matching' },
  { id: 8, level: 'ERROR', time: '14:22:18', msg: 'CUDA out of memory during matching. Falling back to chunked processing.' },
  { id: 9, level: 'INFO', time: '14:22:20', msg: 'Processing chunk 1/4...' },
  { id: 10, level: 'INFO', time: '14:22:25', msg: 'Processing chunk 2/4...' },
];

export const Console: React.FC = () => {
  const [autoScroll, setAutoScroll] = useState(true);
  const [height, setHeight] = useState(300);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [mockLogs, autoScroll]);

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
  };

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
        <GripHorizontal size={12} className="text-zinc-600 group-hover:text-zinc-300 opacity-0 group-hover:opacity-100 transition-opacity" />
      </div>

      {/* Toolbar */}
      <div className="h-10 border-b border-zinc-800/50 bg-[#0a0a0b] flex items-center justify-between px-4">
        <div className="flex items-center gap-3">
          <Terminal size={14} className="text-emerald-500" />
          <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400">System Console</span>
        </div>

        <div className="flex items-center gap-4">
          <div className="relative">
            <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-zinc-500" />
            <input 
              type="text" 
              placeholder="Filter logs..." 
              className="bg-zinc-900 border border-zinc-800 rounded-md pl-8 pr-3 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-emerald-500/50"
            />
          </div>
          
          <button 
            onClick={() => setAutoScroll(!autoScroll)}
            className={`p-1.5 rounded-md border transition-colors flex items-center gap-1.5 text-xs ${
              autoScroll 
                ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' 
                : 'bg-zinc-900 border-zinc-800 text-zinc-500 hover:text-zinc-300'
            }`}
          >
            {autoScroll ? <Lock size={12} /> : <Unlock size={12} />}
            Auto-scroll
          </button>
        </div>
      </div>

      {/* Logs Area */}
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto font-mono text-xs py-2 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:bg-zinc-800 [&::-webkit-scrollbar-track]:bg-transparent"
      >
        {mockLogs.map((log, index) => {
          if (log.transition) {
            return (
              <div key={log.id} className="w-full my-2 bg-emerald-500/10 border-y border-emerald-500/20 py-1.5 px-4 flex items-center">
                <span className="text-emerald-400 font-bold uppercase tracking-widest text-[10px]">{log.msg}</span>
              </div>
            );
          }

          return (
            <div key={log.id} className="group flex items-start px-4 py-0.5 hover:bg-zinc-900/50 transition-colors">
              <div className="w-10 text-zinc-700 select-none text-right pr-4 border-r border-zinc-800/50 mr-4 shrink-0">
                {index + 1}
              </div>
              <div className="w-20 text-zinc-500 shrink-0 select-none">{log.time}</div>
              <div className={`w-16 shrink-0 font-bold ${
                log.level === 'INFO' ? 'text-zinc-400' :
                log.level === 'WARN' ? 'text-amber-400' :
                'text-red-400'
              }`}>
                [{log.level}]
              </div>
              <div className={`flex-1 break-all pr-4 ${
                log.level === 'ERROR' ? 'text-red-300' : 'text-zinc-300'
              }`}>
                {log.msg}
              </div>
              <button 
                onClick={() => handleCopy(log.msg || '')}
                className="opacity-0 group-hover:opacity-100 p-1 text-zinc-500 hover:text-zinc-200 transition-all"
              >
                <Copy size={12} />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
};
