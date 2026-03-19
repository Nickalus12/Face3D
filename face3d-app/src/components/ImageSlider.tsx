import React, { useRef, useState, useCallback } from 'react';

interface ImageSliderProps {
  leftUrl: string;
  rightUrl: string;
  leftLabel?: string;
  rightLabel?: string;
  className?: string;
}

const ImageSlider: React.FC<ImageSliderProps> = ({
  leftUrl,
  rightUrl,
  leftLabel = 'A',
  rightLabel = 'B',
  className = '',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [sliderPos, setSliderPos] = useState(50);
  const [isDragging, setIsDragging] = useState(false);

  const updatePosition = useCallback((clientX: number) => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const x = clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPos(pct);
  }, []);

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
      setIsDragging(true);
      updatePosition(e.clientX);
    },
    [updatePosition],
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!isDragging) return;
      updatePosition(e.clientX);
    },
    [isDragging, updatePosition],
  );

  const handlePointerUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  return (
    <div
      ref={containerRef}
      className={`relative select-none overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900 ${className}`}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      style={{ touchAction: 'none', cursor: 'ew-resize' }}
    >
      {/* Right image (full, underneath) */}
      <img
        src={rightUrl}
        alt={rightLabel}
        className="w-full h-full object-cover pointer-events-none"
        draggable={false}
      />

      {/* Left image (clipped) */}
      <img
        src={leftUrl}
        alt={leftLabel}
        className="absolute inset-0 w-full h-full object-cover pointer-events-none"
        draggable={false}
        style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
      />

      {/* Labels */}
      <div className="absolute top-3 left-3 bg-black/60 backdrop-blur-sm px-2.5 py-1 rounded-md text-[11px] font-semibold text-indigo-400 border border-indigo-500/30 pointer-events-none">
        {leftLabel}
      </div>
      <div className="absolute top-3 right-3 bg-black/60 backdrop-blur-sm px-2.5 py-1 rounded-md text-[11px] font-semibold text-amber-400 border border-amber-500/30 pointer-events-none">
        {rightLabel}
      </div>

      {/* Frosted glass divider line */}
      <div
        className="absolute top-0 bottom-0 w-[2px] pointer-events-none z-10"
        style={{
          left: `calc(${sliderPos}% - 1px)`,
          background: 'rgba(255,255,255,0.3)',
          backdropFilter: 'blur(4px)',
          boxShadow: '0 0 8px rgba(255,255,255,0.15)',
        }}
      >
        {/* Grip handle */}
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-10 rounded-lg flex items-center justify-center"
          style={{
            background: 'rgba(255,255,255,0.12)',
            backdropFilter: 'blur(12px)',
            border: '1px solid rgba(255,255,255,0.2)',
            boxShadow: '0 2px 12px rgba(0,0,0,0.4)',
          }}
        >
          <span className="text-white/80 text-xs font-light tracking-tighter select-none">
            {'<|>'}
          </span>
        </div>
      </div>
    </div>
  );
};

export default ImageSlider;
