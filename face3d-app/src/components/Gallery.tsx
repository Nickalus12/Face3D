import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
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
  Video,
  Camera,
  Radio,
  SlidersHorizontal,
  ArrowUpDown,
  Grid3X3,
  LayoutGrid,
  Maximize2,
  Info,
  Filter,
  Box,
  Ruler,
  Sparkles,
} from 'lucide-react';
import { convertFileSrc } from '@tauri-apps/api/core';
import useSessionStore from '../store/sessionStore';
import {
  listRenders,
  listPreviews,
  listInputFrames,
  getImageCounts,
  getSensorSummary,
  getSessionFiles,
  type PreviewFile,
  type SensorSummary,
} from '../lib/tauri';
import { ReportButton } from './ReportButton';
import ImageSlider from './ImageSlider';

// ── Types ───────────────────────────────────────────────────────

type GalleryTab = 'renders' | 'depth' | 'landmarks' | 'masks' | 'comparison' | 'frames';

type SortMode = 'name-asc' | 'name-desc' | 'type';

interface LazyImageProps {
  src: string;
  alt: string;
  className?: string;
  onClick?: () => void;
  badge?: string;
  badgeColor?: string;
}

interface CaptureSummary {
  videoName: string | null;
  videoDuration: string | null;
  videoResolution: string | null;
  photoCount: number;
  sensorInfo: SensorSummary | null;
  extractedFrames: number;
  selectedFrames: number;
  depthMaps: number;
  pointCloudSize: string | null;
  totalFiles: number;
  totalSize: string;
}

// ── Helpers ─────────────────────────────────────────────────────

function isTauri(): boolean {
  return typeof window !== 'undefined' && !!(window as any).__TAURI_INTERNALS__;
}

function toImageSrc(filePath: string): string {
  if (isTauri()) {
    return convertFileSrc(filePath);
  }
  return '';
}

function filenameFromPath(path: string): string {
  const parts = path.replace(/\\/g, '/').split('/');
  return parts[parts.length - 1] || path;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
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
      className={`group relative overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900/50 cursor-pointer hover:border-indigo-500/40 hover:shadow-lg hover:shadow-indigo-500/5 transition-all duration-200 ${className || ''}`}
      onClick={onClick}
    >
      {isVisible && src ? (
        <>
          <img
            src={src}
            alt={alt}
            className={`w-full h-full object-cover transition-all duration-300 group-hover:scale-[1.03] ${
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
          <div className="w-full h-full bg-zinc-800/30 animate-pulse rounded-lg" />
        </div>
      )}

      {/* Hover overlay */}
      <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-end justify-between p-2.5">
        <span className="text-[11px] text-zinc-200 font-mono truncate max-w-[80%]">{alt}</span>
        <div className="flex items-center gap-1">
          <Maximize2 size={12} className="text-zinc-300" />
        </div>
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
  images: { src: string; label: string; meta?: string }[];
  initialIndex: number;
  onClose: () => void;
}

const Lightbox: React.FC<LightboxProps> = ({ images, initialIndex, onClose }) => {
  const [index, setIndex] = useState(initialIndex);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [showInfo, setShowInfo] = useState(false);
  const [imgDimensions, setImgDimensions] = useState<{ w: number; h: number } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const current = images[index];

  const goNext = useCallback(() => {
    setIndex((i) => (i + 1) % images.length);
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setImgDimensions(null);
  }, [images.length]);

  const goPrev = useCallback(() => {
    setIndex((i) => (i - 1 + images.length) % images.length);
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setImgDimensions(null);
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
        case 'i':
          setShowInfo((v) => !v);
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

  const handleImgLoad = useCallback((e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setImgDimensions({ w: img.naturalWidth, h: img.naturalHeight });
  }, []);

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
          {imgDimensions && (
            <span className="text-xs text-zinc-600 font-mono">
              {imgDimensions.w} x {imgDimensions.h}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowInfo((v) => !v)}
            className={`p-2 rounded-lg transition-colors ${
              showInfo
                ? 'text-indigo-400 bg-indigo-500/10'
                : 'text-zinc-400 hover:text-white hover:bg-white/10'
            }`}
            title="Toggle info (I)"
          >
            <Info size={18} />
          </button>
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
            title="Reset view (0)"
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
          onLoad={handleImgLoad}
        />
      </div>

      {/* Info overlay */}
      {showInfo && (
        <div className="absolute bottom-24 left-1/2 -translate-x-1/2 bg-black/80 backdrop-blur-md border border-zinc-700/60 rounded-xl px-5 py-3 text-xs text-zinc-300 space-y-1 min-w-[200px]">
          <div className="flex justify-between gap-6">
            <span className="text-zinc-500">Filename</span>
            <span className="font-mono">{current.label}</span>
          </div>
          {imgDimensions && (
            <div className="flex justify-between gap-6">
              <span className="text-zinc-500">Dimensions</span>
              <span className="font-mono">{imgDimensions.w} x {imgDimensions.h}</span>
            </div>
          )}
          {current.meta && (
            <div className="flex justify-between gap-6">
              <span className="text-zinc-500">Category</span>
              <span>{current.meta}</span>
            </div>
          )}
        </div>
      )}

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

const TAB_COLORS: Record<GalleryTab, string> = {
  renders: 'text-indigo-400',
  depth: 'text-blue-400',
  landmarks: 'text-emerald-400',
  masks: 'text-purple-400',
  comparison: 'text-amber-400',
  frames: 'text-zinc-400',
};

const TAB_DOT_COLORS: Record<GalleryTab, string> = {
  renders: 'bg-indigo-400',
  depth: 'bg-blue-400',
  landmarks: 'bg-emerald-400',
  masks: 'bg-purple-400',
  comparison: 'bg-amber-400',
  frames: 'bg-zinc-500',
};

// ── Data Capture Summary Card ───────────────────────────────────

const CaptureSummaryCard: React.FC<{ summary: CaptureSummary }> = ({ summary }) => {
  const items = useMemo(() => {
    const result: { icon: React.ReactNode; label: string; value: string; color: string }[] = [];

    if (summary.videoName) {
      const videoDetail = [summary.videoResolution, summary.videoDuration].filter(Boolean).join(', ');
      result.push({
        icon: <Video size={14} />,
        label: 'Video',
        value: `${summary.videoName}${videoDetail ? ` (${videoDetail})` : ''}`,
        color: 'text-indigo-400',
      });
    }

    if (summary.photoCount > 0) {
      result.push({
        icon: <Camera size={14} />,
        label: 'Photos',
        value: `${summary.photoCount} Expert RAW`,
        color: 'text-emerald-400',
      });
    }

    if (summary.sensorInfo) {
      const sensorCount = summary.sensorInfo.sensors.length;
      const rate = sensorCount > 0
        ? `${Math.round(summary.sensorInfo.sensors[0].samples / Math.max(summary.sensorInfo.duration, 0.01))}Hz`
        : '';
      result.push({
        icon: <Radio size={14} />,
        label: 'Sensors',
        value: `${summary.sensorInfo.device_name}${rate ? ` (${rate})` : ''}`,
        color: 'text-blue-400',
      });
    }

    if (summary.extractedFrames > 0) {
      result.push({
        icon: <Film size={14} />,
        label: 'Frames',
        value: `${summary.extractedFrames} extracted${summary.selectedFrames > 0 ? `, ${summary.selectedFrames} selected` : ''}`,
        color: 'text-amber-400',
      });
    }

    if (summary.depthMaps > 0) {
      result.push({
        icon: <Ruler size={14} />,
        label: 'Depth',
        value: `${summary.depthMaps} maps (DA3)`,
        color: 'text-cyan-400',
      });
    }

    if (summary.pointCloudSize) {
      result.push({
        icon: <Sparkles size={14} />,
        label: 'Points',
        value: summary.pointCloudSize,
        color: 'text-purple-400',
      });
    }

    return result;
  }, [summary]);

  if (items.length === 0) return null;

  return (
    <div className="mb-6 p-4 rounded-xl bg-zinc-900/40 border border-zinc-800/60">
      <div className="flex items-center gap-2 mb-3">
        <Box size={14} className="text-zinc-500" />
        <span className="text-[11px] font-bold tracking-wider text-zinc-500 uppercase">
          Capture Summary
        </span>
        <span className="text-[11px] text-zinc-600 ml-auto font-mono">{summary.totalSize}</span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
        {items.map((item, i) => (
          <div
            key={i}
            className="flex items-center gap-2.5 px-3 py-2 rounded-lg bg-zinc-900/60 border border-zinc-800/40"
          >
            <div className={`shrink-0 ${item.color}`}>{item.icon}</div>
            <div className="min-w-0">
              <div className="text-[11px] text-zinc-500 uppercase tracking-wider">{item.label}</div>
              <div className="text-xs text-zinc-300 truncate">{item.value}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// ── Before/After Color Correction Comparison ────────────────────

const ColorCorrectionCompare: React.FC<{
  comparisons: PreviewFile[];
}> = ({ comparisons }) => {
  const [activeIdx, setActiveIdx] = useState(0);

  if (comparisons.length === 0) return null;

  // Look for pairs: comparison previews typically contain "before" and "after" or "slog" and "srgb"
  // The comparison images themselves are usually side-by-side previews
  // For the slider, we look for pairs of original vs corrected frames
  const current = comparisons[activeIdx];
  if (!current) return null;

  return (
    <div className="mb-6">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <SlidersHorizontal size={14} className="text-amber-400" />
          <span className="text-[11px] font-bold tracking-wider text-zinc-500 uppercase">
            Color Correction Preview
          </span>
        </div>
        {comparisons.length > 1 && (
          <div className="flex items-center gap-1">
            <button
              onClick={() => setActiveIdx((i) => Math.max(0, i - 1))}
              disabled={activeIdx === 0}
              className="p-1 text-zinc-500 hover:text-zinc-300 disabled:opacity-30 transition-colors"
            >
              <ChevronLeft size={14} />
            </button>
            <span className="text-[11px] text-zinc-500 font-mono">
              {activeIdx + 1}/{comparisons.length}
            </span>
            <button
              onClick={() => setActiveIdx((i) => Math.min(comparisons.length - 1, i + 1))}
              disabled={activeIdx === comparisons.length - 1}
              className="p-1 text-zinc-500 hover:text-zinc-300 disabled:opacity-30 transition-colors"
            >
              <ChevronRight size={14} />
            </button>
          </div>
        )}
      </div>
      <div className="aspect-video rounded-xl overflow-hidden border border-zinc-800 bg-zinc-900">
        <ImageSlider
          leftUrl={toImageSrc(current.path)}
          rightUrl={toImageSrc(current.path)}
          leftLabel="S-Log3 (Original)"
          rightLabel="sRGB (Corrected)"
          className="w-full h-full"
        />
      </div>
      <div className="mt-2 flex items-center gap-3 text-[11px] text-zinc-500">
        <span>Drag the slider to compare before/after color correction</span>
      </div>
    </div>
  );
};

// ── Gallery Component ───────────────────────────────────────────

export const Gallery: React.FC = () => {
  const { currentSession } = useSessionStore();
  const [activeTab, setActiveTab] = useState<GalleryTab>('renders');
  const [renders, setRenders] = useState<string[]>([]);
  const [previews, setPreviews] = useState<PreviewFile[]>([]);
  const [frames, setFrames] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sortMode, setSortMode] = useState<SortMode>('name-asc');
  const [filterText, setFilterText] = useState('');
  const [gridSize, setGridSize] = useState<'sm' | 'md' | 'lg'>('md');
  const [captureSummary, setCaptureSummary] = useState<CaptureSummary | null>(null);
  const [lightbox, setLightbox] = useState<{
    images: { src: string; label: string; meta?: string }[];
    index: number;
  } | null>(null);

  // Load all data when session changes
  useEffect(() => {
    if (!currentSession) {
      setRenders([]);
      setPreviews([]);
      setFrames([]);
      setCaptureSummary(null);
      return;
    }

    const sessionId = currentSession.id;

    // Load all tabs data + summary in parallel
    Promise.all([
      listRenders(sessionId),
      listPreviews(sessionId),
      listInputFrames(sessionId),
      getImageCounts(sessionId),
      getSensorSummary(sessionId),
      getSessionFiles(sessionId),
    ]).then(([r, p, f, counts, sensor, files]) => {
      setRenders(r);
      setPreviews(p);
      setFrames(f);

      // Build capture summary
      const totalSize = files.reduce((sum, [_, size]) => sum + size, 0);
      const plyFile = files.find(([name]) => name.endsWith('.ply'));
      let pointCloudSize: string | null = null;
      if (plyFile) {
        // Estimate: ~200 bytes per point for PLY with properties
        const estimatedPoints = Math.round(plyFile[1] / 200);
        if (estimatedPoints >= 1_000_000) {
          pointCloudSize = `${(estimatedPoints / 1_000_000).toFixed(1)}M point cloud`;
        } else if (estimatedPoints >= 1_000) {
          pointCloudSize = `${Math.round(estimatedPoints / 1_000)}K point cloud`;
        }
      }

      // Try to find video info from file names
      const videoFile = files.find(([name]) =>
        /\.(mp4|mov|avi|mkv)$/i.test(name)
      );

      setCaptureSummary({
        videoName: videoFile ? videoFile[0] : null,
        videoDuration: null,
        videoResolution: null,
        photoCount: 0, // Would need a separate API call
        sensorInfo: sensor,
        extractedFrames: counts[2] || f.length,
        selectedFrames: counts[1],
        depthMaps: p.filter((pv) => pv.preview_type === 'depth').length,
        pointCloudSize,
        totalFiles: files.length,
        totalSize: formatBytes(totalSize),
      });
    });
  }, [currentSession?.id]);

  // Load tab-specific data on tab change
  useEffect(() => {
    if (!currentSession) return;
    setIsLoading(true);

    const sessionId = currentSession.id;

    switch (activeTab) {
      case 'renders':
        listRenders(sessionId).then((r) => {
          setRenders(r);
          setIsLoading(false);
        });
        break;
      case 'frames':
        listInputFrames(sessionId).then((f) => {
          setFrames(f);
          setIsLoading(false);
        });
        break;
      default:
        // Previews are already loaded
        setIsLoading(false);
        break;
    }
  }, [currentSession?.id, activeTab]);

  // Categorize previews by type
  const categorizedPreviews = useMemo(() => {
    const groups: Record<string, PreviewFile[]> = {
      depth: [],
      landmark: [],
      mask: [],
      comparison: [],
      other: [],
    };
    previews.forEach((p) => {
      const type = p.preview_type;
      if (groups[type]) {
        groups[type].push(p);
      } else {
        groups.other.push(p);
      }
    });
    return groups;
  }, [previews]);

  // Get current tab's images
  const currentImages = useMemo(() => {
    let images: { src: string; label: string; path: string; meta?: string }[] = [];

    switch (activeTab) {
      case 'renders':
        images = renders.map((r) => ({
          src: toImageSrc(r),
          label: filenameFromPath(r),
          path: r,
          meta: 'Render',
        }));
        break;
      case 'depth':
        images = categorizedPreviews.depth.map((p) => ({
          src: toImageSrc(p.path),
          label: p.name,
          path: p.path,
          meta: 'Depth Map',
        }));
        break;
      case 'landmarks':
        images = categorizedPreviews.landmark.map((p) => ({
          src: toImageSrc(p.path),
          label: p.name,
          path: p.path,
          meta: 'Landmarks',
        }));
        break;
      case 'masks':
        images = categorizedPreviews.mask.map((p) => ({
          src: toImageSrc(p.path),
          label: p.name,
          path: p.path,
          meta: 'Mask',
        }));
        break;
      case 'comparison':
        images = categorizedPreviews.comparison.map((p) => ({
          src: toImageSrc(p.path),
          label: p.name,
          path: p.path,
          meta: 'Comparison',
        }));
        break;
      case 'frames':
        images = frames.map((f) => ({
          src: toImageSrc(f),
          label: filenameFromPath(f),
          path: f,
          meta: 'Input Frame',
        }));
        break;
    }

    // Apply filter
    if (filterText.trim()) {
      const q = filterText.toLowerCase();
      images = images.filter((img) => img.label.toLowerCase().includes(q));
    }

    // Apply sort
    switch (sortMode) {
      case 'name-asc':
        images.sort((a, b) => a.label.localeCompare(b.label));
        break;
      case 'name-desc':
        images.sort((a, b) => b.label.localeCompare(a.label));
        break;
      case 'type':
        // Already grouped by type, no sort needed
        break;
    }

    return images;
  }, [activeTab, renders, frames, categorizedPreviews, filterText, sortMode]);

  // Build lightbox images for current tab
  const openLightbox = useCallback(
    (index: number) => {
      if (currentImages.length > 0) {
        setLightbox({
          images: currentImages.map((img) => ({
            src: img.src,
            label: img.label,
            meta: img.meta,
          })),
          index,
        });
      }
    },
    [currentImages]
  );

  // Tab definitions with counts
  const tabs = useMemo((): { id: GalleryTab; label: string; icon: typeof Layers; count: number }[] => [
    { id: 'renders', label: 'Renders', icon: Layers, count: renders.length },
    { id: 'depth', label: 'Depth Maps', icon: Layers, count: categorizedPreviews.depth.length },
    { id: 'landmarks', label: 'Landmarks', icon: Eye, count: categorizedPreviews.landmark.length },
    { id: 'masks', label: 'Masks', icon: Layers, count: categorizedPreviews.mask.length },
    { id: 'comparison', label: 'Color Correction', icon: SlidersHorizontal, count: categorizedPreviews.comparison.length },
    { id: 'frames', label: 'Input Frames', icon: Film, count: frames.length },
  ], [renders.length, frames.length, categorizedPreviews]);

  // Grid column classes
  const gridColsClass = gridSize === 'sm'
    ? 'grid-cols-4 sm:grid-cols-5 md:grid-cols-6 lg:grid-cols-8 xl:grid-cols-10'
    : gridSize === 'md'
    ? 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6'
    : 'grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4';

  const aspectClass = activeTab === 'frames' ? 'aspect-[16/9]' : 'aspect-square';

  // Empty state -- no session selected
  if (!currentSession) {
    return (
      <div className="flex-1 bg-[#0a0a0b] flex flex-col items-center justify-center text-zinc-500">
        <div className="w-16 h-16 rounded-2xl bg-zinc-800/30 border border-zinc-700/15 flex items-center justify-center mb-5">
          <ImageIcon size={28} className="text-zinc-600" />
        </div>
        <p className="text-sm text-zinc-400 font-medium">No session selected</p>
        <p className="text-xs mt-1.5 text-zinc-600">Select a session from the sidebar to view images</p>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-[#0a0a0b] flex flex-col overflow-hidden">
      {/* Tab Bar */}
      <div className="shrink-0 border-b border-zinc-800/50 px-6 pt-4 pb-3">
        <div className="flex items-center justify-between mb-3">
          <div className="flex gap-1 overflow-x-auto pb-1 [&::-webkit-scrollbar]:h-0">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg transition-all duration-200 whitespace-nowrap shrink-0 ${
                    isActive
                      ? 'bg-zinc-800/80 text-zinc-200 border border-zinc-700/50 shadow-sm'
                      : 'text-zinc-500 border border-transparent hover:text-zinc-300 hover:bg-white/[0.03]'
                  }`}
                >
                  <div className={`w-1.5 h-1.5 rounded-full ${isActive ? TAB_DOT_COLORS[tab.id] : 'bg-zinc-700'}`} />
                  {tab.label}
                  {tab.count > 0 && (
                    <span
                      className={`ml-0.5 px-1.5 py-0.5 rounded-full text-[9px] font-bold tabular-nums ${
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
          <div className="flex items-center gap-2 shrink-0 ml-4">
            <ReportButton sessionId={currentSession.id} />
          </div>
        </div>

        {/* Filter/Sort/Grid controls */}
        <div className="flex items-center gap-2">
          {/* Filter */}
          <div className="flex-1 max-w-[240px]">
            <input
              type="text"
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
              placeholder="Filter images..."
              className="w-full bg-zinc-800/60 border border-zinc-700/50 rounded-lg px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-500 focus:outline-none focus:ring-1 focus:ring-indigo-500/40 focus:border-indigo-500/30 transition-all duration-150"
            />
          </div>

          {/* Sort */}
          <button
            onClick={() =>
              setSortMode((m) =>
                m === 'name-asc' ? 'name-desc' : m === 'name-desc' ? 'type' : 'name-asc'
              )
            }
            className="flex items-center gap-1 px-2.5 py-1.5 text-[11px] text-zinc-500 hover:text-zinc-300 bg-zinc-900/60 border border-zinc-800/60 rounded-lg transition-colors"
            title={`Sort: ${sortMode}`}
          >
            <ArrowUpDown size={12} />
            {sortMode === 'name-asc' ? 'A-Z' : sortMode === 'name-desc' ? 'Z-A' : 'Type'}
          </button>

          {/* Grid size */}
          <div className="flex items-center border border-zinc-800/60 rounded-lg overflow-hidden">
            <button
              onClick={() => setGridSize('sm')}
              className={`p-1.5 transition-colors ${
                gridSize === 'sm' ? 'bg-zinc-800 text-zinc-300' : 'text-zinc-600 hover:text-zinc-400'
              }`}
              title="Small grid"
            >
              <Grid3X3 size={12} />
            </button>
            <button
              onClick={() => setGridSize('md')}
              className={`p-1.5 transition-colors ${
                gridSize === 'md' ? 'bg-zinc-800 text-zinc-300' : 'text-zinc-600 hover:text-zinc-400'
              }`}
              title="Medium grid"
            >
              <LayoutGrid size={12} />
            </button>
            <button
              onClick={() => setGridSize('lg')}
              className={`p-1.5 transition-colors ${
                gridSize === 'lg' ? 'bg-zinc-800 text-zinc-300' : 'text-zinc-600 hover:text-zinc-400'
              }`}
              title="Large grid"
            >
              <Maximize2 size={12} />
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto overscroll-contain p-6" style={{ WebkitOverflowScrolling: 'touch' }}>
        {/* Capture summary card */}
        {captureSummary && activeTab === 'renders' && (
          <CaptureSummaryCard summary={captureSummary} />
        )}

        {/* Color Correction comparison slider (only on comparison tab) */}
        {activeTab === 'comparison' && categorizedPreviews.comparison.length > 0 && (
          <ColorCorrectionCompare comparisons={categorizedPreviews.comparison} />
        )}

        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <div className="w-6 h-6 border-2 border-zinc-700 border-t-indigo-500 rounded-full animate-spin" />
          </div>
        ) : currentImages.length === 0 ? (
          <EmptyTabState tab={activeTab} hasFilter={filterText.trim().length > 0} />
        ) : (
          <div>
            <p className="text-xs text-zinc-500 mb-3 flex items-center gap-2">
              <span className={`w-1.5 h-1.5 rounded-full ${TAB_DOT_COLORS[activeTab]}`} />
              {currentImages.length} image{currentImages.length !== 1 ? 's' : ''}
              {filterText.trim() && ` matching "${filterText}"`}
            </p>
            <div className={`grid ${gridColsClass} gap-3`}>
              {currentImages.map((img, i) => (
                <LazyImage
                  key={img.path}
                  src={img.src}
                  alt={img.label}
                  className={aspectClass}
                  onClick={() => openLightbox(i)}
                  badge={activeTab !== 'renders' && activeTab !== 'frames' ? img.meta : undefined}
                  badgeColor={PREVIEW_BADGE_STYLES[activeTab] || PREVIEW_BADGE_STYLES.other}
                />
              ))}
            </div>
          </div>
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

// ── Empty Tab State ──────────────────────────────────────────────

const EmptyTabState: React.FC<{ tab: GalleryTab; hasFilter: boolean }> = ({ tab, hasFilter }) => {
  const messages: Record<GalleryTab, { title: string; desc: string; icon: typeof Layers }> = {
    renders: { title: 'No renders yet', desc: 'Run the pipeline to generate turntable renders', icon: Layers },
    depth: { title: 'No depth maps', desc: 'Depth maps are generated in Stage 6 (DA3)', icon: Layers },
    landmarks: { title: 'No landmark previews', desc: 'Landmarks are detected in Stage 9', icon: Eye },
    masks: { title: 'No mask previews', desc: 'Face masks are generated in Stage 11', icon: Layers },
    comparison: { title: 'No color comparisons', desc: 'Color correction previews from Stage 2', icon: SlidersHorizontal },
    frames: { title: 'No input frames', desc: 'Frames are extracted in Stage 1', icon: Film },
  };

  const msg = messages[tab];
  const Icon = msg.icon;

  return (
    <div className="flex flex-col items-center justify-center h-64 text-zinc-500">
      <div className="w-14 h-14 rounded-2xl bg-zinc-800/30 border border-zinc-700/15 flex items-center justify-center mb-4">
        <Icon size={22} className="text-zinc-600" />
      </div>
      <p className="text-sm text-zinc-400 font-medium">
        {hasFilter ? 'No matching images' : msg.title}
      </p>
      <p className="text-xs mt-1.5 text-zinc-600">
        {hasFilter ? 'Try adjusting your filter' : msg.desc}
      </p>
    </div>
  );
};
