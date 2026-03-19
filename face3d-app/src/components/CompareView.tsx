import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  ArrowLeftRight,
  TrendingUp,
  TrendingDown,
  Minus,
  Eye,
  Layers as LayersIcon,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import useSessionStore from '../store/sessionStore';
import {
  getSessionMetrics,
  listRenders,
  getSessionFiles,
} from '../lib/tauri';
import ImageSlider from './ImageSlider';

// ── Types ────────────────────────────────────────────────────────

interface MetricEntry {
  [key: string]: unknown;
  iter?: number;
  iteration?: number;
  step?: number;
  loss?: number;
  psnr?: number;
  ssim?: number;
  n_gs?: number;
  gaussians?: number;
  vram?: number;
  vram_peak?: number;
  training_time?: number;
  elapsed?: number;
}

interface SessionData {
  metrics: MetricEntry[];
  renders: string[];
  files: { name: string; size: number }[];
}

// ── Helpers ──────────────────────────────────────────────────────

function getIter(m: MetricEntry): number {
  return m.iter ?? m.iteration ?? m.step ?? 0;
}

function lastVal(metrics: MetricEntry[], key: keyof MetricEntry): number | null {
  for (let i = metrics.length - 1; i >= 0; i--) {
    const v = metrics[i][key];
    if (typeof v === 'number' && isFinite(v)) return v;
  }
  return null;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

function formatNumber(n: number | null): string {
  if (n === null) return '--';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toFixed(2);
}

function fileSize(files: { name: string; size: number }[], pattern: string): number | null {
  const f = files.find((f) => f.name.toLowerCase().includes(pattern.toLowerCase()));
  return f ? f.size : null;
}

// ── Component ────────────────────────────────────────────────────

const CompareView: React.FC = () => {
  const { sessions } = useSessionStore();

  const [sessionAId, setSessionAId] = useState<string>('');
  const [sessionBId, setSessionBId] = useState<string>('');
  const [dataA, setDataA] = useState<SessionData | null>(null);
  const [dataB, setDataB] = useState<SessionData | null>(null);
  const [loading, setLoading] = useState(false);

  // Visual comparison state
  const [compareMode, setCompareMode] = useState<'slider' | 'overlay'>('slider');
  const [overlayOpacity, setOverlayOpacity] = useState(50);
  const [frameIndex, setFrameIndex] = useState(0);

  // Auto-select first two sessions
  useEffect(() => {
    if (sessions.length >= 2 && !sessionAId && !sessionBId) {
      setSessionAId(sessions[0].id);
      setSessionBId(sessions[1].id);
    } else if (sessions.length === 1 && !sessionAId) {
      setSessionAId(sessions[0].id);
    }
  }, [sessions, sessionAId, sessionBId]);

  // Load data for a session
  const loadSessionData = useCallback(async (sid: string): Promise<SessionData | null> => {
    if (!sid) return null;
    try {
      const [metricsStr, renders, rawFiles] = await Promise.all([
        getSessionMetrics(sid),
        listRenders(sid),
        getSessionFiles(sid),
      ]);

      let metrics: MetricEntry[] = [];
      if (metricsStr) {
        try {
          metrics = JSON.parse(metricsStr);
        } catch {
          metrics = [];
        }
      }

      const files = rawFiles.map(([name, size]) => ({ name, size }));
      return { metrics, renders, files };
    } catch {
      return null;
    }
  }, []);

  // Load data when selections change
  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    Promise.all([
      sessionAId ? loadSessionData(sessionAId) : Promise.resolve(null),
      sessionBId ? loadSessionData(sessionBId) : Promise.resolve(null),
    ]).then(([a, b]) => {
      if (!cancelled) {
        setDataA(a);
        setDataB(b);
        setFrameIndex(0);
        setLoading(false);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [sessionAId, sessionBId, loadSessionData]);

  // Swap sessions
  const handleSwap = () => {
    setSessionAId(sessionBId);
    setSessionBId(sessionAId);
  };

  // ── Metrics comparison data ──────────────────────────────────

  const metricRows = useMemo(() => {
    if (!dataA && !dataB) return [];

    const mA = dataA?.metrics ?? [];
    const mB = dataB?.metrics ?? [];

    const rows: {
      label: string;
      valueA: number | null;
      valueB: number | null;
      unit: string;
      higherIsBetter: boolean;
    }[] = [
      {
        label: 'PSNR',
        valueA: lastVal(mA, 'psnr'),
        valueB: lastVal(mB, 'psnr'),
        unit: 'dB',
        higherIsBetter: true,
      },
      {
        label: 'Training Time',
        valueA: lastVal(mA, 'training_time') ?? lastVal(mA, 'elapsed'),
        valueB: lastVal(mB, 'training_time') ?? lastVal(mB, 'elapsed'),
        unit: 's',
        higherIsBetter: false,
      },
      {
        label: 'Gaussian Count',
        valueA: lastVal(mA, 'n_gs') ?? lastVal(mA, 'gaussians'),
        valueB: lastVal(mB, 'n_gs') ?? lastVal(mB, 'gaussians'),
        unit: '',
        higherIsBetter: false,
      },
      {
        label: 'VRAM Peak',
        valueA: lastVal(mA, 'vram_peak') ?? lastVal(mA, 'vram'),
        valueB: lastVal(mB, 'vram_peak') ?? lastVal(mB, 'vram'),
        unit: 'MB',
        higherIsBetter: false,
      },
      {
        label: 'PLY Size',
        valueA: fileSize(dataA?.files ?? [], '.ply'),
        valueB: fileSize(dataB?.files ?? [], '.ply'),
        unit: '',
        higherIsBetter: false,
      },
      {
        label: 'Mesh Size',
        valueA: fileSize(dataA?.files ?? [], 'mesh'),
        valueB: fileSize(dataB?.files ?? [], 'mesh'),
        unit: '',
        higherIsBetter: false,
      },
    ];

    return rows;
  }, [dataA, dataB]);

  // ── Chart overlay data ───────────────────────────────────────

  const psnrChartData = useMemo(() => {
    const mA = dataA?.metrics ?? [];
    const mB = dataB?.metrics ?? [];

    const map = new Map<number, { iter: number; psnrA?: number; psnrB?: number }>();

    for (const m of mA) {
      if (m.psnr != null) {
        const iter = getIter(m);
        const existing = map.get(iter) ?? { iter };
        existing.psnrA = m.psnr;
        map.set(iter, existing);
      }
    }

    for (const m of mB) {
      if (m.psnr != null) {
        const iter = getIter(m);
        const existing = map.get(iter) ?? { iter };
        existing.psnrB = m.psnr;
        map.set(iter, existing);
      }
    }

    return Array.from(map.values()).sort((a, b) => a.iter - b.iter);
  }, [dataA, dataB]);

  const lossChartData = useMemo(() => {
    const mA = dataA?.metrics ?? [];
    const mB = dataB?.metrics ?? [];

    const map = new Map<number, { iter: number; lossA?: number; lossB?: number }>();

    for (const m of mA) {
      if (m.loss != null) {
        const iter = getIter(m);
        const existing = map.get(iter) ?? { iter };
        existing.lossA = m.loss;
        map.set(iter, existing);
      }
    }

    for (const m of mB) {
      if (m.loss != null) {
        const iter = getIter(m);
        const existing = map.get(iter) ?? { iter };
        existing.lossB = m.loss;
        map.set(iter, existing);
      }
    }

    return Array.from(map.values()).sort((a, b) => a.iter - b.iter);
  }, [dataA, dataB]);

  // ── Parameter diff ───────────────────────────────────────────

  const paramDiffs = useMemo(() => {
    if (!dataA?.metrics.length || !dataB?.metrics.length) return [];

    const firstA = dataA.metrics[0];
    const firstB = dataB.metrics[0];

    const allKeys = new Set([...Object.keys(firstA), ...Object.keys(firstB)]);
    const skipKeys = new Set(['iter', 'iteration', 'step', 'loss', 'psnr', 'ssim', 'n_gs', 'gaussians', 'vram', 'vram_peak', 'training_time', 'elapsed']);

    const diffs: { key: string; valA: string; valB: string; isDifferent: boolean }[] = [];

    for (const key of allKeys) {
      if (skipKeys.has(key)) continue;
      const valA = String(firstA[key] ?? '--');
      const valB = String(firstB[key] ?? '--');
      diffs.push({ key, valA, valB, isDifferent: valA !== valB });
    }

    // Sort: differences first
    diffs.sort((a, b) => (a.isDifferent === b.isDifferent ? 0 : a.isDifferent ? -1 : 1));
    return diffs;
  }, [dataA, dataB]);

  // ── Renders for visual comparison ────────────────────────────

  const maxFrames = Math.max(dataA?.renders.length ?? 0, dataB?.renders.length ?? 0);
  const renderA = dataA?.renders[frameIndex];
  const renderB = dataB?.renders[frameIndex];

  // Convert local file path to Tauri asset URL
  const assetUrl = (path: string | undefined) => {
    if (!path) return '';
    // Convert Windows path to asset protocol URL
    return `asset://localhost/${path.replace(/\\/g, '/')}`;
  };

  // ── Render ───────────────────────────────────────────────────

  if (sessions.length < 2) {
    return (
      <div className="flex-1 bg-[#0a0a0b] flex flex-col items-center justify-center text-zinc-500">
        <LayersIcon size={48} className="mb-4 opacity-20" />
        <p className="text-sm">Need at least two sessions to compare.</p>
        <p className="text-xs mt-1 text-zinc-600">Run the pipeline multiple times with different parameters.</p>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-[#0a0a0b] overflow-y-auto [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:bg-zinc-800 [&::-webkit-scrollbar-track]:bg-transparent">
      {/* ── Session Picker ──────────────────────────────────────── */}
      <div className="sticky top-0 z-20 bg-[#0a0a0b]/95 backdrop-blur-md border-b border-zinc-800/60 px-6 py-4">
        <div className="flex items-center gap-4 max-w-5xl mx-auto">
          {/* Session A */}
          <div className="flex-1">
            <label className="block text-[11px] font-bold tracking-wider text-indigo-400 uppercase mb-1.5">
              Session A
            </label>
            <select
              value={sessionAId}
              onChange={(e) => setSessionAId(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-700/60 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/20 transition-colors"
            >
              <option value="">Select session...</option>
              {sessions.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>

          {/* Swap Button */}
          <button
            onClick={handleSwap}
            className="mt-5 p-2.5 rounded-xl bg-zinc-800/60 border border-zinc-700/40 text-zinc-400 hover:text-white hover:bg-zinc-700/60 hover:border-zinc-600 active:scale-95 transition-all"
            title="Swap sessions"
          >
            <ArrowLeftRight size={16} />
          </button>

          {/* Session B */}
          <div className="flex-1">
            <label className="block text-[11px] font-bold tracking-wider text-amber-400 uppercase mb-1.5">
              Session B
            </label>
            <select
              value={sessionBId}
              onChange={(e) => setSessionBId(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-700/60 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20 transition-colors"
            >
              <option value="">Select session...</option>
              {sessions.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {loading && (
        <div className="flex items-center justify-center py-12 text-zinc-500 text-sm">
          Loading session data...
        </div>
      )}

      {!loading && (sessionAId || sessionBId) && (
        <div className="max-w-5xl mx-auto px-6 py-6 space-y-8">
          {/* ── Metrics Comparison ──────────────────────────────── */}
          <section>
            <h3 className="text-[11px] font-bold tracking-wider text-zinc-500 uppercase mb-4">
              Metrics Comparison
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {metricRows.map((row) => {
                const diff =
                  row.valueA != null && row.valueB != null && row.valueA !== 0
                    ? ((row.valueB - row.valueA) / Math.abs(row.valueA)) * 100
                    : null;
                const bIsWinner =
                  diff !== null
                    ? row.higherIsBetter
                      ? diff > 0
                      : diff < 0
                    : false;
                const aIsWinner =
                  diff !== null
                    ? row.higherIsBetter
                      ? diff < 0
                      : diff > 0
                    : false;

                return (
                  <div
                    key={row.label}
                    className="bg-zinc-900/60 border border-zinc-800/60 rounded-xl p-4 flex flex-col gap-3"
                  >
                    <div className="text-[11px] font-bold tracking-wider text-zinc-500 uppercase">
                      {row.label}
                    </div>
                    <div className="flex items-end justify-between gap-2">
                      {/* Value A */}
                      <div
                        className={`flex flex-col items-start ${
                          aIsWinner ? 'ring-1 ring-emerald-500/20 bg-emerald-500/5 rounded-lg p-2 -m-2' : ''
                        }`}
                      >
                        <span className="text-[9px] text-indigo-400 font-semibold uppercase">A</span>
                        <span className="text-lg font-medium text-zinc-200">
                          {row.label.includes('Size') && row.valueA != null
                            ? formatBytes(row.valueA)
                            : formatNumber(row.valueA)}
                        </span>
                        {row.unit && row.valueA != null && !row.label.includes('Size') && (
                          <span className="text-[11px] text-zinc-600">{row.unit}</span>
                        )}
                      </div>

                      {/* Diff indicator */}
                      <div className="flex flex-col items-center pb-1">
                        {diff !== null ? (
                          <>
                            {Math.abs(diff) < 0.5 ? (
                              <Minus size={14} className="text-zinc-500" />
                            ) : bIsWinner ? (
                              <TrendingUp size={14} className="text-emerald-400" />
                            ) : (
                              <TrendingDown size={14} className="text-red-400" />
                            )}
                            <span
                              className={`text-[11px] font-mono ${
                                Math.abs(diff) < 0.5
                                  ? 'text-zinc-500'
                                  : bIsWinner
                                  ? 'text-emerald-400'
                                  : 'text-red-400'
                              }`}
                            >
                              {diff > 0 ? '+' : ''}
                              {diff.toFixed(1)}%
                            </span>
                          </>
                        ) : (
                          <span className="text-zinc-600 text-xs">--</span>
                        )}
                      </div>

                      {/* Value B */}
                      <div
                        className={`flex flex-col items-end ${
                          bIsWinner ? 'ring-1 ring-emerald-500/20 bg-emerald-500/5 rounded-lg p-2 -m-2' : ''
                        }`}
                      >
                        <span className="text-[9px] text-amber-400 font-semibold uppercase">B</span>
                        <span className="text-lg font-medium text-zinc-200">
                          {row.label.includes('Size') && row.valueB != null
                            ? formatBytes(row.valueB)
                            : formatNumber(row.valueB)}
                        </span>
                        {row.unit && row.valueB != null && !row.label.includes('Size') && (
                          <span className="text-[11px] text-zinc-600">{row.unit}</span>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          {/* ── Visual Comparison ───────────────────────────────── */}
          {maxFrames > 0 && (
            <section>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-[11px] font-bold tracking-wider text-zinc-500 uppercase">
                  Visual Comparison
                </h3>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setCompareMode('slider')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                      compareMode === 'slider'
                        ? 'bg-indigo-500/15 text-indigo-400 border border-indigo-500/30'
                        : 'text-zinc-500 hover:text-zinc-300 border border-transparent'
                    }`}
                  >
                    <Eye size={12} className="inline mr-1.5" />
                    Slider
                  </button>
                  <button
                    onClick={() => setCompareMode('overlay')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                      compareMode === 'overlay'
                        ? 'bg-amber-500/15 text-amber-400 border border-amber-500/30'
                        : 'text-zinc-500 hover:text-zinc-300 border border-transparent'
                    }`}
                  >
                    <LayersIcon size={12} className="inline mr-1.5" />
                    Overlay
                  </button>
                </div>
              </div>

              {/* Comparison area */}
              <div className="aspect-video rounded-xl overflow-hidden border border-zinc-800 bg-zinc-900">
                {renderA && renderB ? (
                  compareMode === 'slider' ? (
                    <ImageSlider
                      leftUrl={assetUrl(renderA)}
                      rightUrl={assetUrl(renderB)}
                      leftLabel="A"
                      rightLabel="B"
                      className="w-full h-full"
                    />
                  ) : (
                    /* Overlay mode */
                    <div className="relative w-full h-full">
                      <img
                        src={assetUrl(renderA)}
                        alt="Session A"
                        className="absolute inset-0 w-full h-full object-cover"
                      />
                      <img
                        src={assetUrl(renderB)}
                        alt="Session B"
                        className="absolute inset-0 w-full h-full object-cover"
                        style={{ opacity: overlayOpacity / 100 }}
                      />
                      {/* Labels */}
                      <div className="absolute top-3 left-3 bg-black/60 backdrop-blur-sm px-2.5 py-1 rounded-md text-xs font-semibold text-indigo-400 border border-indigo-500/30">
                        A (base)
                      </div>
                      <div className="absolute top-3 right-3 bg-black/60 backdrop-blur-sm px-2.5 py-1 rounded-md text-xs font-semibold text-amber-400 border border-amber-500/30">
                        B ({overlayOpacity}%)
                      </div>
                    </div>
                  )
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-zinc-500 text-sm">
                    {!renderA && !renderB
                      ? 'No renders available for either session'
                      : !renderA
                      ? 'No renders available for Session A'
                      : 'No renders available for Session B'}
                  </div>
                )}
              </div>

              {/* Controls bar */}
              <div className="mt-3 flex items-center gap-4">
                {/* Frame selector */}
                {maxFrames > 1 && (
                  <div className="flex-1">
                    <label className="text-[11px] text-zinc-500 font-medium block mb-1">
                      Frame {frameIndex + 1} / {maxFrames}
                    </label>
                    <input
                      type="range"
                      min={0}
                      max={maxFrames - 1}
                      value={frameIndex}
                      onChange={(e) => setFrameIndex(Number(e.target.value))}
                      className="w-full h-1.5 bg-zinc-800 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-indigo-500 [&::-webkit-slider-thumb]:shadow-lg"
                    />
                  </div>
                )}

                {/* Opacity slider for overlay mode */}
                {compareMode === 'overlay' && (
                  <div className="w-48">
                    <label className="text-[11px] text-zinc-500 font-medium block mb-1">
                      B Opacity: {overlayOpacity}%
                    </label>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      value={overlayOpacity}
                      onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                      className="w-full h-1.5 bg-zinc-800 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-amber-500 [&::-webkit-slider-thumb]:shadow-lg"
                    />
                  </div>
                )}
              </div>
            </section>
          )}

          {/* ── Parameter Diff ──────────────────────────────────── */}
          {paramDiffs.length > 0 && (
            <section>
              <h3 className="text-[11px] font-bold tracking-wider text-zinc-500 uppercase mb-4">
                Parameter Differences
              </h3>
              <div className="bg-zinc-900/60 border border-zinc-800/60 rounded-xl overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-zinc-800/60">
                      <th className="text-left px-4 py-2.5 text-[11px] font-bold tracking-wider text-zinc-500 uppercase">
                        Parameter
                      </th>
                      <th className="text-right px-4 py-2.5 text-[11px] font-bold tracking-wider text-indigo-400 uppercase">
                        Session A
                      </th>
                      <th className="text-right px-4 py-2.5 text-[11px] font-bold tracking-wider text-amber-400 uppercase">
                        Session B
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {paramDiffs.map((d) => (
                      <tr
                        key={d.key}
                        className={`border-b border-zinc-800/30 ${
                          d.isDifferent ? 'bg-amber-500/5' : ''
                        }`}
                      >
                        <td className={`px-4 py-2 font-mono text-xs ${d.isDifferent ? 'text-amber-300' : 'text-zinc-400'}`}>
                          {d.key}
                        </td>
                        <td className="px-4 py-2 text-right font-mono text-xs text-zinc-300">
                          {d.valA}
                        </td>
                        <td className="px-4 py-2 text-right font-mono text-xs text-zinc-300">
                          {d.valB}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* ── Chart Overlays ──────────────────────────────────── */}
          {(psnrChartData.length > 0 || lossChartData.length > 0) && (
            <section>
              <h3 className="text-[11px] font-bold tracking-wider text-zinc-500 uppercase mb-4">
                Training Curves
              </h3>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* PSNR Chart */}
                {psnrChartData.length > 0 && (
                  <div className="bg-zinc-900/60 border border-zinc-800/60 rounded-xl p-4">
                    <div className="text-[11px] font-bold tracking-wider text-zinc-500 uppercase mb-3">
                      PSNR Over Training
                    </div>
                    <ResponsiveContainer width="100%" height={240}>
                      <LineChart data={psnrChartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                        <XAxis
                          dataKey="iter"
                          stroke="#52525b"
                          tick={{ fontSize: 10, fill: '#71717a' }}
                          label={{ value: 'Iteration', position: 'insideBottom', offset: -5, fill: '#52525b', fontSize: 10 }}
                        />
                        <YAxis
                          stroke="#52525b"
                          tick={{ fontSize: 10, fill: '#71717a' }}
                          label={{ value: 'PSNR (dB)', angle: -90, position: 'insideLeft', fill: '#52525b', fontSize: 10 }}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#18181b',
                            border: '1px solid #3f3f46',
                            borderRadius: '8px',
                            fontSize: '11px',
                          }}
                          labelStyle={{ color: '#a1a1aa' }}
                        />
                        <Legend
                          wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }}
                        />
                        <Line
                          type="monotone"
                          dataKey="psnrA"
                          name={`A: ${sessionAId}`}
                          stroke="#818cf8"
                          strokeWidth={2}
                          dot={false}
                          connectNulls
                        />
                        <Line
                          type="monotone"
                          dataKey="psnrB"
                          name={`B: ${sessionBId}`}
                          stroke="#f59e0b"
                          strokeWidth={2}
                          dot={false}
                          connectNulls
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Loss Chart */}
                {lossChartData.length > 0 && (
                  <div className="bg-zinc-900/60 border border-zinc-800/60 rounded-xl p-4">
                    <div className="text-[11px] font-bold tracking-wider text-zinc-500 uppercase mb-3">
                      Loss Over Training
                    </div>
                    <ResponsiveContainer width="100%" height={240}>
                      <LineChart data={lossChartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                        <XAxis
                          dataKey="iter"
                          stroke="#52525b"
                          tick={{ fontSize: 10, fill: '#71717a' }}
                          label={{ value: 'Iteration', position: 'insideBottom', offset: -5, fill: '#52525b', fontSize: 10 }}
                        />
                        <YAxis
                          stroke="#52525b"
                          tick={{ fontSize: 10, fill: '#71717a' }}
                          label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: '#52525b', fontSize: 10 }}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#18181b',
                            border: '1px solid #3f3f46',
                            borderRadius: '8px',
                            fontSize: '11px',
                          }}
                          labelStyle={{ color: '#a1a1aa' }}
                        />
                        <Legend
                          wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }}
                        />
                        <Line
                          type="monotone"
                          dataKey="lossA"
                          name={`A: ${sessionAId}`}
                          stroke="#818cf8"
                          strokeWidth={2}
                          dot={false}
                          connectNulls
                        />
                        <Line
                          type="monotone"
                          dataKey="lossB"
                          name={`B: ${sessionBId}`}
                          stroke="#f59e0b"
                          strokeWidth={2}
                          dot={false}
                          connectNulls
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            </section>
          )}

          {/* Bottom spacer */}
          <div className="h-8" />
        </div>
      )}
    </div>
  );
};

export default CompareView;
