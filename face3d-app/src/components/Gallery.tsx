import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Image as ImageIcon,
  X,
  ChevronLeft,
  ChevronRight,
  Download,
  Eye,
  Layers,
  Film,
  ZoomIn,
  ZoomOut,
  RotateCcw,
} from 'lucide-react';
import { convertFileSrc } from '@tauri-apps/api/core';
import useSessionStore from '../store/sessionStore';
import {
  listRenders,
  listPreviews,
  listInputFrames,
  type PreviewFile,
} from '../lib/tauri';
import { ReportButton } from './ReportButton';

// ── Types ───────────────────────────────────────────────────────

type GalleryTab = 'renders' | 'previews' | 'frames';

interface LazyImageProps {
  src: string;
  alt: string;
  className?: string;
  onClick?: () => void;
  badge?: string;
  badgeColor?: string;
}

// ── Helpers ─────────────────────────────────────────────────────

function isTauri(): boolean {
  return typeof window !== 'undefined' && !!(window as any).__TAURI_INTERNALS__;
}

function toImageSrc(filePath: string): string {
  if (isTauri()) {
    return convertFileSrc(filePath);
  }
  // In browser dev mode, show a placeholder
  return '';
}

function filenameFromPath(path: string): string {
  const parts = path.replace(/\\/g, '/').split('/');
  return parts[parts.length - 1] || path;
}

// ── Lazy Image with Intersection Observer ───────────────────────

const LazyImage: React.FC<LazyImageProps> = ({ src, alt, className, onClick, badge, badgeColor }) => {
  const imgRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    const el = imgRef.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.unobserve(el);
        }
      },
      { rootMargin: '200px' }
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={imgRef}
      className={`group relative overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900/50 cursor-pointer hover:border-indigo-500/40 transition-all duration-200 ${className || ''}`}
      onClick={onClick}
    >
      {isVisible && src ? (
        <>
          <img
            src={src}
            alt={alt}
            className={`w-full h-full object-cover transition-opacity duration-300 ${
              isLoaded ? 'opacity-100' : 'opacity-0'
            }`}
            onLoad={() => setIsLoaded(true)}
            onError={() => setHasError(true)}
            loading="lazy"
          />
          {!isLoaded && !hasError && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-5 h-5 border-2 border-zinc-700 border-t-indigo-500 rounded-full animate-spin" />
            </div>
          )}
          {hasError && (
            <div className="absolute inset-0 flex items-center justify-center text-zinc-600">
              <ImageIcon size={24} />
            </div>
          )}
        </>
      ) : (
        <div className="w-full h-full min-h-[120px] flex items-center justify-center">
          <div className="w-4 h-4 bg-zinc-800 rounded animate-pulse" />
        </div>
      )}

      {/* Hover overlay */}
      <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-end justify-between p-2">
        <span className="text-[10px] text-zinc-300 font-mono truncate max-w-[80%]">{alt}</span>
        <Eye size={14} className="text-zinc-400 shrink-0" />
      </div>

      {/* Badge */}
      {badge && (
        <div
          className={`absolute top-2 left-2 px-1.5 py-0.5 rounded text-[9px] font-semibold uppercase tracking-wider border ${
            badgeColor || 'bg-zinc-800/80 text-zinc-300 border-zinc-700'
          }`}
        >
          {badge}
        </div>
      )}
    </div>
  );
};

// ── Lightbox ────────────────────────────────────────────────────

interface LightboxProps {
  images: { src: string; label: string }[];
  initialIndex: number;
  onClose: () => void;
}

const Lightbox: React.FC<LightboxProps> = ({ images, initialIndex, onClose }) => {
  const [index, setIndex] = useState(initialIndex);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  const current = images[index];

  const goNext = useCallback(() => {
    setIndex((i) => (i + 1) % images.length);
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [images.length]);

  const goPrev = useCallback(() => {
    setIndex((i) => (i - 1 + images.length) % images.length);
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [images.length]);

  const resetView = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  // Keyboard controls
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'Escape':
          onClose();
          break;
        case 'ArrowRight':
          goNext();
          break;
        case 'ArrowLeft':
          goPrev();
          break;
        case '0':
          resetView();
          break;
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose, goNext, goPrev, resetView]);

  // Scroll to zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    setZoom((z) => {
      const newZoom = z - e.deltaY * 0.001;
      return Math.min(Math.max(0.5, newZoom), 8);
    });
  }, []);

  // Pan when zoomed
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (zoom > 1) {
        setIsDragging(true);
        setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
      }
    },
    [zoom, pan]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (isDragging) {
        setPan({
          x: e.clientX - dragStart.x,
          y: e.clientY - dragStart.y,
        });
      }
    },
    [isDragging, dragStart]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Download current image
  const handleDownload = useCallback(() => {
    const link = document.createElement('a');
    link.href = current.src;
    link.download = current.label;
    link.click();
  }, [current]);

  return (
    <div
      className="fixed inset-0 z-[100] bg-black/95 backdrop-blur-sm flex flex-col"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      {/* Top bar */}
      <div className="flex items-center justify-between px-6 py-4 shrink-0">
        <div className="flex items-center gap-4">
          <span className="text-sm font-mono text-zinc-400">
            {index + 1} / {images.length}
          </span>
          <span className="text-sm text-zinc-500 truncate max-w-[400px]">{current.label}</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setZoom((z) => Math.min(z + 0.5, 8))}
            className="p-2 text-zinc-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
            title="Zoom in"
          >
            <ZoomIn size={18} />
          </button>
          <button
            onClick={() => setZoom((z) => Math.max(z - 0.5, 0.5))}
            className="p-2 text-zinc-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
            title="Zoom out"
          >
            <ZoomOut size={18} />
          </button>
          <button
            onClick={resetView}
            className="p-2 text-zinc-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
            title="Reset view"
          >
            <RotateCcw size={18} />
          </button>
          <span className="text-xs font-mono text-zinc-500 w-12 text-center">
            {Math.round(zoom * 100)}%
          </span>
          <div className="w-px h-5 bg-zinc-800 mx-1" />
          <button
            onClick={handleDownload}
            className="p-2 text-zinc-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
            title="Download"
          >
            <Download size={18} />
          </button>
          <button
            onClick={onClose}
            className="p-2 text-zinc-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
            title="Close (Esc)"
          >
            <X size={18} />
          </button>
        </div>
      </div>

      {/* Image area */}
      <div
        ref={containerRef}
        className={`flex-1 flex items-center justify-center overflow-hidden select-none ${
          zoom > 1 ? 'cursor-grab' : 'cursor-default'
        } ${isDragging ? 'cursor-grabbing' : ''}`}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <img
          src={current.src}
          alt={current.label}
          className="max-w-full max-h-full object-contain transition-transform duration-100"
          style={{
            transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
          }}
          draggable={false}
        />
      </div>

      {/* Navigation arrows */}
      {images.length > 1 && (
        <>
          <button
            onClick={(e) => {
              e.stopPropagation();
              goPrev();
            }}
            className="absolute left-4 top-1/2 -translate-y-1/2 p-3 bg-black/50 hover:bg-black/80 border border-zinc-700 rounded-full text-zinc-400 hover:text-white transition-all"
          >
            <ChevronLeft size={24} />
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              goNext();
            }}
            className="absolute right-4 top-1/2 -translate-y-1/2 p-3 bg-black/50 hover:bg-black/80 border border-zinc-700 rounded-full text-zinc-400 hover:text-white transition-all"
          >
            <ChevronRight size={24} />
          </button>
        </>
      )}

      {/* Bottom thumbnail strip */}
      {images.length > 1 && images.length <= 60 && (
        <div className="shrink-0 px-6 py-3 flex gap-1.5 overflow-x-auto justify-center [&::-webkit-scrollbar]:h-1 [&::-webkit-scrollbar-thumb]:bg-zinc-800">
          {images.map((img, i) => (
            <button
              key={i}
              onClick={() => {
                setIndex(i);
                setZoom(1);
                setPan({ x: 0, y: 0 });
              }}
              className={`w-12 h-12 rounded-md overflow-hidden border-2 shrink-0 transition-all ${
                i === index
                  ? 'border-indigo-500 opacity-100 scale-110'
                  : 'border-transparent opacity-40 hover:opacity-70'
              }`}
            >
              <img
                src={img.src}
                alt=""
                className="w-full h-full object-cover"
                loading="lazy"
              />
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

// ── Preview Badge Color Map ─────────────────────────────────────

const PREVIEW_BADGE_STYLES: Record<string, string> = {
  depth: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
  landmark: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
  mask: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
  comparison: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
  other: 'bg-zinc-800/80 text-zinc-300 border-zinc-700',
};

// ── Gallery Component ───────────────────────────────────────────

export const Gallery: React.FC = () => {
  const { currentSession } = useSessionStore();
  const [activeTab, setActiveTab] = useState<GalleryTab>('renders');
  const [renders, setRenders] = useState<string[]>([]);
  const [previews, setPreviews] = useState<PreviewFile[]>([]);
  const [frames, setFrames] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [lightbox, setLightbox] = useState<{
    images: { src: string; label: string }[];
    index: number;
  } | null>(null);

  // Load data when session or tab changes
  useEffect(() => {
    if (!currentSession) {
      setRenders([]);
      setPreviews([]);
      setFrames([]);
      return;
    }

    setIsLoading(true);
    const sessionId = currentSession.id;

    switch (activeTab) {
      case 'renders':
        listRenders(sessionId).then((r) => {
          setRenders(r);
          setIsLoading(false);
        });
        break;
      case 'previews':
        listPreviews(sessionId).then((p) => {
          setPreviews(p);
          setIsLoading(false);
        });
        break;
      case 'frames':
        listInputFrames(sessionId).then((f) => {
          setFrames(f);
          setIsLoading(false);
        });
        break;
    }
  }, [currentSession?.id, activeTab]);

  // Build lightbox images for current tab
  const openLightbox = useCallback(
    (index: number) => {
      let images: { src: string; label: string }[] = [];

      switch (activeTab) {
        case 'renders':
          images = renders.map((r) => ({
            src: toImageSrc(r),
            label: filenameFromPath(r),
          }));
          break;
        case 'previews':
          images = previews.map((p) => ({
            src: toImageSrc(p.path),
            label: p.name,
          }));
          break;
        case 'frames':
          images = frames.map((f) => ({
            src: toImageSrc(f),
            label: filenameFromPath(f),
          }));
          break;
      }

      if (images.length > 0) {
        setLightbox({ images, index });
      }
    },
    [activeTab, renders, previews, frames]
  );

  const tabs: { id: GalleryTab; label: string; icon: typeof Layers; count: number }[] = [
    { id: 'renders', label: 'Turntable Renders', icon: Layers, count: renders.length },
    { id: 'previews', label: 'Previews', icon: Eye, count: previews.length },
    { id: 'frames', label: 'Input Frames', icon: Film, count: frames.length },
  ];

  // Empty state — no session selected
  if (!currentSession) {
    return (
      <div className="flex-1 bg-[#0a0a0b] flex flex-col items-center justify-center text-zinc-500">
        <ImageIcon size={48} className="mb-4 opacity-20" />
        <p className="text-sm">No session selected.</p>
        <p className="text-xs mt-1 text-zinc-600">Select a session from the sidebar to view images.</p>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-[#0a0a0b] flex flex-col overflow-hidden">
      {/* Tab Bar */}
      <div className="shrink-0 border-b border-zinc-800/50 px-6 pt-4 pb-3 flex items-center justify-between">
        <div className="flex gap-2">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-lg border border-b-0 transition-all ${
                  isActive
                    ? 'bg-[#111113] text-zinc-200 border-zinc-700/50'
                    : 'bg-transparent text-zinc-500 border-transparent hover:text-zinc-300 hover:bg-white/[0.02]'
                }`}
              >
                <Icon size={14} />
                {tab.label}
                {tab.count > 0 && (
                  <span
                    className={`ml-1 px-1.5 py-0.5 rounded-full text-[10px] font-bold ${
                      isActive
                        ? 'bg-indigo-500/20 text-indigo-300'
                        : 'bg-zinc-800 text-zinc-500'
                    }`}
                  >
                    {tab.count}
                  </span>
                )}
              </button>
            );
          })}
        </div>
        <div className="pb-2.5">
          <ReportButton sessionId={currentSession.id} />
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:bg-zinc-800 [&::-webkit-scrollbar-track]:bg-transparent">
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <div className="w-6 h-6 border-2 border-zinc-700 border-t-indigo-500 rounded-full animate-spin" />
          </div>
        ) : activeTab === 'renders' ? (
          <RenderGrid renders={renders} onSelect={openLightbox} />
        ) : activeTab === 'previews' ? (
          <PreviewGrid previews={previews} onSelect={openLightbox} />
        ) : (
          <FrameGrid frames={frames} onSelect={openLightbox} />
        )}
      </div>

      {/* Lightbox */}
      {lightbox && (
        <Lightbox
          images={lightbox.images}
          initialIndex={lightbox.index}
          onClose={() => setLightbox(null)}
        />
      )}
    </div>
  );
};

// ── Render Grid ─────────────────────────────────────────────────

const RenderGrid: React.FC<{ renders: string[]; onSelect: (i: number) => void }> = ({
  renders,
  onSelect,
}) => {
  if (renders.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-zinc-500">
        <Layers size={40} className="mb-3 opacity-20" />
        <p className="text-sm">No renders yet.</p>
        <p className="text-xs mt-1 text-zinc-600">Run the pipeline to generate turntable renders.</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
      {renders.map((path, i) => (
        <LazyImage
          key={path}
          src={toImageSrc(path)}
          alt={filenameFromPath(path)}
          className="aspect-square"
          onClick={() => onSelect(i)}
        />
      ))}
    </div>
  );
};

// ── Preview Grid ────────────────────────────────────────────────

const PreviewGrid: React.FC<{ previews: PreviewFile[]; onSelect: (i: number) => void }> = ({
  previews,
  onSelect,
}) => {
  if (previews.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-zinc-500">
        <Eye size={40} className="mb-3 opacity-20" />
        <p className="text-sm">No preview images yet.</p>
        <p className="text-xs mt-1 text-zinc-600">Previews are generated during pipeline stages.</p>
      </div>
    );
  }

  // Group by type
  const grouped: Record<string, { preview: PreviewFile; index: number }[]> = {};
  previews.forEach((p, i) => {
    const type = p.preview_type;
    if (!grouped[type]) grouped[type] = [];
    grouped[type].push({ preview: p, index: i });
  });

  const typeOrder = ['depth', 'landmark', 'mask', 'comparison', 'other'];

  return (
    <div className="space-y-8">
      {typeOrder.map((type) => {
        const items = grouped[type];
        if (!items || items.length === 0) return null;

        const typeLabel = type.charAt(0).toUpperCase() + type.slice(1);

        return (
          <div key={type}>
            <h3 className="text-xs font-semibold uppercase tracking-widest text-zinc-500 mb-3 flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                type === 'depth' ? 'bg-blue-400' :
                type === 'landmark' ? 'bg-emerald-400' :
                type === 'mask' ? 'bg-purple-400' :
                type === 'comparison' ? 'bg-amber-400' :
                'bg-zinc-500'
              }`} />
              {typeLabel}
              <span className="text-zinc-600">{items.length}</span>
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
              {items.map(({ preview, index }) => (
                <LazyImage
                  key={preview.path}
                  src={toImageSrc(preview.path)}
                  alt={preview.name}
                  className="aspect-[4/3]"
                  onClick={() => onSelect(index)}
                  badge={typeLabel}
                  badgeColor={PREVIEW_BADGE_STYLES[type]}
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
};

// ── Frame Grid ──────────────────────────────────────────────────

const FrameGrid: React.FC<{ frames: string[]; onSelect: (i: number) => void }> = ({
  frames,
  onSelect,
}) => {
  if (frames.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-zinc-500">
        <Film size={40} className="mb-3 opacity-20" />
        <p className="text-sm">No input frames found.</p>
        <p className="text-xs mt-1 text-zinc-600">Frames are extracted in Stage 1 of the pipeline.</p>
      </div>
    );
  }

  return (
    <div>
      <p className="text-xs text-zinc-500 mb-4">
        {frames.length} frames extracted from source video
      </p>
      <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 xl:grid-cols-8 gap-2">
        {frames.map((path, i) => (
          <LazyImage
            key={path}
            src={toImageSrc(path)}
            alt={filenameFromPath(path)}
            className="aspect-[16/9]"
            onClick={() => onSelect(i)}
          />
        ))}
      </div>
    </div>
  );
};
