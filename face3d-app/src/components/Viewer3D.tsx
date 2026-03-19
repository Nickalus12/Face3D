import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Grid, Sparkles, GizmoHelper, GizmoViewport } from '@react-three/drei';
import {
  Layers,
  SlidersHorizontal,
  Maximize,
  Loader2,
  FileQuestion,
  RotateCcw,
  Eye,
  Box,
  User,
  MoveHorizontal,
  ArrowUp,
  RotateCw,
  Palette,
  AlertTriangle,
  RefreshCw,
} from 'lucide-react';
import * as THREE from 'three';
import { readFile } from '@tauri-apps/plugin-fs';
import useSessionStore from '../store/sessionStore';
import { getModelPath } from '../lib/tauri';
import { loadPlyFromBytes, type PlyData } from '../lib/plyLoader';

// ── Types ──────────────────────────────────────────────────────

type ViewMode = 'points' | 'mesh';
type BgMode = 'dark' | 'light' | 'gradient' | 'transparent';

interface ModelInfo {
  path: string;
  fileName: string;
  plyData: PlyData | null;
  error: string | null;
}

interface ViewPreset {
  id: string;
  label: string;
  icon: typeof User;
  position: [number, number, number];
}

const VIEW_PRESETS: ViewPreset[] = [
  { id: 'front', label: 'Front', icon: User, position: [0, 1, 3] },
  { id: 'side', label: 'Side', icon: MoveHorizontal, position: [3, 1, 0] },
  { id: 'top', label: 'Top', icon: ArrowUp, position: [0, 4, 0.1] },
  { id: 'threequarter', label: '3/4', icon: RotateCw, position: [2, 1.5, 2] },
];

const BG_OPTIONS: { id: BgMode; label: string }[] = [
  { id: 'dark', label: 'Dark' },
  { id: 'light', label: 'Light' },
  { id: 'gradient', label: 'Gradient' },
  { id: 'transparent', label: 'None' },
];

// Stored camera positions per session
const cameraStateCache = new Map<string, { position: THREE.Vector3; target: THREE.Vector3 }>();

// ── Point Cloud Component ──────────────────────────────────────

function PointCloud({
  plyData,
  pointSize,
  ghostMode,
}: {
  plyData: PlyData;
  pointSize: number;
  ghostMode: boolean;
}) {
  const pointsRef = useRef<THREE.Points>(null);

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(plyData.positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(plyData.colors, 3));
    geo.computeBoundingSphere();
    return geo;
  }, [plyData]);

  return (
    <points ref={pointsRef} geometry={geometry}>
      <pointsMaterial
        size={pointSize}
        vertexColors
        transparent
        opacity={ghostMode ? 0.3 : 0.9}
        sizeAttenuation
        depthWrite={!ghostMode}
      />
    </points>
  );
}

// ── Auto-fit camera to the model ──────────────────────────────

function AutoFit({ plyData, sessionId }: { plyData: PlyData; sessionId?: string }) {
  const { camera } = useThree();
  const fitted = useRef(false);

  useEffect(() => {
    if (fitted.current) return;
    fitted.current = true;

    // Check if we have a cached camera position for this session
    if (sessionId && cameraStateCache.has(sessionId)) {
      const cached = cameraStateCache.get(sessionId)!;
      camera.position.copy(cached.position);
      camera.lookAt(cached.target);
      if (camera instanceof THREE.PerspectiveCamera) {
        camera.updateProjectionMatrix();
      }
      return;
    }

    // Compute bounding box
    const positions = plyData.positions;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (let i = 0; i < plyData.count; i++) {
      const x = positions[i * 3];
      const y = positions[i * 3 + 1];
      const z = positions[i * 3 + 2];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;
    }

    const center = new THREE.Vector3(
      (minX + maxX) / 2,
      (minY + maxY) / 2,
      (minZ + maxZ) / 2,
    );

    const size = new THREE.Vector3(maxX - minX, maxY - minY, maxZ - minZ);
    const maxDim = Math.max(size.x, size.y, size.z);

    const distance = maxDim * 1.8;
    camera.position.set(center.x, center.y, center.z + distance);
    camera.lookAt(center);

    if (camera instanceof THREE.PerspectiveCamera) {
      camera.near = distance * 0.01;
      camera.far = distance * 100;
      camera.updateProjectionMatrix();
    }
  }, [plyData, camera, sessionId]);

  return null;
}

// ── Camera State Saver ────────────────────────────────────────

function CameraSaver({ sessionId }: { sessionId?: string }) {
  const { camera } = useThree();
  const controlsRef = useRef<any>(null);

  useFrame(() => {
    if (sessionId) {
      cameraStateCache.set(sessionId, {
        position: camera.position.clone(),
        target: new THREE.Vector3(0, 0, 0), // simplified
      });
    }
  });

  return null;
}

// ── Camera Distance Tracker ────────────────────────────────────

function CameraTracker({ onUpdate }: { onUpdate: (pos: THREE.Vector3) => void }) {
  const { camera } = useThree();

  useFrame(() => {
    onUpdate(camera.position.clone());
  });

  return null;
}

// ── Double-click focus ────────────────────────────────────────

function DoubleClickFocus({ controlsRef }: { controlsRef: React.RefObject<any> }) {
  const { camera, raycaster, scene } = useThree();

  useEffect(() => {
    const canvas = document.querySelector('canvas');
    if (!canvas) return;

    const handleDblClick = (event: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(scene.children, true);

      if (intersects.length > 0) {
        const point = intersects[0].point;
        if (controlsRef.current) {
          controlsRef.current.target.copy(point);
          controlsRef.current.update();
        }
      }
    };

    canvas.addEventListener('dblclick', handleDblClick);
    return () => canvas.removeEventListener('dblclick', handleDblClick);
  }, [camera, raycaster, scene, controlsRef]);

  return null;
}

// ── Synthetic Placeholder ──────────────────────────────────────

function SyntheticCloud({ pointSize, ghostMode }: { pointSize: number; ghostMode: boolean }) {
  const pointsRef = useRef<THREE.Points>(null);

  useFrame(({ clock }) => {
    if (pointsRef.current) {
      pointsRef.current.position.y = Math.sin(clock.getElapsedTime() * 0.5) * 0.02;
    }
  });

  const { positions, colors } = useMemo(() => {
    const count = 80000;
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    const color = new THREE.Color();

    for (let i = 0; i < count; i++) {
      const u = Math.random();
      const v = Math.random();
      const theta = u * 2.0 * Math.PI;
      const phi = Math.acos(2.0 * v - 1.0);
      const rx = 0.7, ry = 1.0, rz = 0.8;
      let x = rx * Math.sin(phi) * Math.cos(theta);
      let y = ry * Math.sin(phi) * Math.sin(theta);
      let z = rz * Math.cos(phi);
      if (z > 0.2) z *= 0.6;
      const noise = (Math.random() - 0.5) * 0.05;
      pos[i * 3] = x + noise;
      pos[i * 3 + 1] = y + noise + 1.0;
      pos[i * 3 + 2] = z + noise;

      const hue = 0.05 + Math.random() * 0.04;
      const saturation = 0.4 + z * 0.2;
      const lightness = 0.4 + y * 0.2 + Math.random() * 0.1;
      color.setHSL(hue, saturation, lightness);
      if (z > 0.4 && y > 0.8 && y < 1.2) color.setHSL(0.02, 0.6, 0.6);
      col[i * 3] = color.r;
      col[i * 3 + 1] = color.g;
      col[i * 3 + 2] = color.b;
    }
    return { positions: pos, colors: col };
  }, []);

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial
        size={pointSize}
        vertexColors
        transparent
        opacity={ghostMode ? 0.3 : 0.9}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

// ── Format helpers ─────────────────────────────────────────────

function formatNumber(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return n.toString();
}

function formatBytes(bytes: number): string {
  if (bytes >= 1_073_741_824) return (bytes / 1_073_741_824).toFixed(1) + ' GB';
  if (bytes >= 1_048_576) return (bytes / 1_048_576).toFixed(1) + ' MB';
  if (bytes >= 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return bytes + ' B';
}

// ── BG class helper ────────────────────────────────────────────

function getBgClass(bg: BgMode): string {
  switch (bg) {
    case 'light': return 'bg-zinc-300';
    case 'gradient': return 'bg-gradient-to-b from-[#0a0a1a] via-[#0d1017] to-[#050505]';
    case 'transparent': return 'bg-transparent';
    default: return 'bg-[#0a0a0b]';
  }
}

// ── Main Viewer Component ──────────────────────────────────────

export default function Viewer3D() {
  const [pointSize, setPointSize] = useState(0.005);
  const [ghostMode, setGhostMode] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('points');
  const [bgMode, setBgMode] = useState<BgMode>('gradient');
  const [showBgPicker, setShowBgPicker] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [cameraPos, setCameraPos] = useState(new THREE.Vector3(0, 1.2, 3.5));
  const [loadError, setLoadError] = useState<string | null>(null);

  const currentSession = useSessionStore((s) => s.currentSession);
  const controlsRef = useRef<any>(null);

  // Load model when session changes
  useEffect(() => {
    let cancelled = false;

    async function loadModel() {
      if (!currentSession) {
        setModelInfo(null);
        setLoadError(null);
        return;
      }

      if (!currentSession.has_gaussians && !currentSession.has_mesh) {
        setModelInfo(null);
        setLoadError(null);
        return;
      }

      setIsLoading(true);
      setLoadError(null);

      try {
        const modelPath = await getModelPath(currentSession.id);
        if (cancelled || !modelPath) {
          if (!cancelled) {
            setIsLoading(false);
            setLoadError('No model file found');
          }
          return;
        }

        const fileName = modelPath.split(/[/\\]/).pop() ?? 'unknown';

        const bytes = await readFile(modelPath);
        if (cancelled) return;

        if (fileName.endsWith('.ply')) {
          const plyData = loadPlyFromBytes(bytes);
          if (cancelled) return;

          setModelInfo({
            path: modelPath,
            fileName,
            plyData,
            error: null,
          });

          if (fileName === 'gaussians.ply') {
            setViewMode('points');
          }
        } else {
          setModelInfo({
            path: modelPath,
            fileName,
            plyData: null,
            error: null,
          });
        }
      } catch (err) {
        if (!cancelled) {
          const msg = err instanceof Error ? err.message : String(err);
          setLoadError(msg);
          setModelInfo(null);
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    loadModel();
    return () => {
      cancelled = true;
    };
  }, [currentSession?.id, currentSession?.has_gaussians, currentSession?.has_mesh]);

  const handleResetCamera = useCallback(() => {
    if (controlsRef.current) {
      controlsRef.current.reset();
    }
  }, []);

  const handleViewPreset = useCallback((preset: ViewPreset) => {
    if (controlsRef.current) {
      const controls = controlsRef.current;
      // Animate to preset position
      controls.object.position.set(...preset.position);
      controls.target.set(0, 1, 0);
      controls.update();
    }
  }, []);

  const handleCameraUpdate = useCallback((pos: THREE.Vector3) => {
    setCameraPos(pos);
  }, []);

  const handleFullscreen = useCallback(() => {
    const el = document.querySelector('.viewer3d-container');
    if (!el) return;
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      el.requestFullscreen();
    }
  }, []);

  const handleRetryLoad = useCallback(() => {
    // Force a re-load by toggling session dependency
    setLoadError(null);
    setIsLoading(true);
    // Re-trigger the effect
    if (currentSession) {
      getModelPath(currentSession.id).then(async (modelPath) => {
        if (!modelPath) {
          setIsLoading(false);
          setLoadError('No model file found');
          return;
        }
        try {
          const fileName = modelPath.split(/[/\\]/).pop() ?? 'unknown';
          const bytes = await readFile(modelPath);
          if (fileName.endsWith('.ply')) {
            const plyData = loadPlyFromBytes(bytes);
            setModelInfo({ path: modelPath, fileName, plyData, error: null });
          }
        } catch (err) {
          setLoadError(err instanceof Error ? err.message : String(err));
        } finally {
          setIsLoading(false);
        }
      });
    }
  }, [currentSession]);

  const cameraDistance = cameraPos.length();
  const hasModel = modelInfo?.plyData != null;
  const noSession = !currentSession;
  const noModel =
    !noSession &&
    !isLoading &&
    !hasModel &&
    !loadError &&
    !currentSession?.has_gaussians &&
    !currentSession?.has_mesh;

  return (
    <div className={`viewer3d-container w-full h-full relative ${getBgClass(bgMode)}`}>
      {/* Floating Toolbar */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-1 p-1 bg-[#111113]/80 backdrop-blur-2xl border border-white/10 rounded-xl shadow-2xl">
        {/* View mode toggle */}
        <button
          className={`p-2 rounded-lg transition-all duration-150 ${
            viewMode === 'points'
              ? 'bg-indigo-500/20 text-indigo-400'
              : 'text-zinc-400 hover:text-white hover:bg-white/5'
          }`}
          onClick={() => setViewMode('points')}
          title="Points View"
        >
          <Eye size={16} />
        </button>
        <button
          className={`p-2 rounded-lg transition-all duration-150 ${
            viewMode === 'mesh'
              ? 'bg-indigo-500/20 text-indigo-400'
              : 'text-zinc-400 hover:text-white hover:bg-white/5'
          }`}
          onClick={() => setViewMode('mesh')}
          title="Mesh View"
        >
          <Box size={16} />
        </button>

        <div className="w-[1px] h-4 bg-white/10 mx-2" />

        <button
          className={`p-2 rounded-lg transition-all duration-150 ${
            ghostMode
              ? 'bg-indigo-500/20 text-indigo-400'
              : 'text-zinc-400 hover:text-white hover:bg-white/5'
          }`}
          onClick={() => setGhostMode(!ghostMode)}
          title="Ghost Mode"
        >
          <Layers size={16} />
        </button>

        <div className="w-[1px] h-4 bg-white/10 mx-2" />

        {/* Point size slider */}
        <div
          className="flex items-center gap-3 px-2 text-zinc-400 group"
          title="Point Size"
        >
          <SlidersHorizontal
            size={14}
            className="group-hover:text-white transition-colors"
          />
          <input
            type="range"
            min="0.001"
            max="0.02"
            step="0.0005"
            value={pointSize}
            onChange={(e) => setPointSize(parseFloat(e.target.value))}
            className="w-20 h-1 bg-zinc-800 rounded-full appearance-none cursor-pointer accent-indigo-500 opacity-70 group-hover:opacity-100 transition-opacity"
          />
        </div>

        <div className="w-[1px] h-4 bg-white/10 mx-2" />

        {/* Background picker */}
        <div className="relative">
          <button
            className={`p-2 rounded-lg transition-all duration-150 ${
              showBgPicker
                ? 'bg-indigo-500/20 text-indigo-400'
                : 'text-zinc-400 hover:text-white hover:bg-white/5'
            }`}
            onClick={() => setShowBgPicker(!showBgPicker)}
            title="Background"
          >
            <Palette size={16} />
          </button>
          {showBgPicker && (
            <div className="absolute top-full mt-2 right-0 bg-[#111113]/95 backdrop-blur-xl border border-white/10 rounded-lg p-1.5 shadow-2xl min-w-[120px] animate-scaleIn">
              {BG_OPTIONS.map((opt) => (
                <button
                  key={opt.id}
                  onClick={() => { setBgMode(opt.id); setShowBgPicker(false); }}
                  className={`w-full text-left px-3 py-1.5 rounded-md text-xs transition-all duration-150 ${
                    bgMode === opt.id
                      ? 'bg-indigo-500/20 text-indigo-300'
                      : 'text-zinc-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          )}
        </div>

        <button
          className="p-2 rounded-lg text-zinc-400 hover:text-white hover:bg-white/5 transition-all duration-150"
          onClick={handleResetCamera}
          title="Reset Camera"
        >
          <RotateCcw size={16} />
        </button>
        <button
          className="p-2 rounded-lg text-zinc-400 hover:text-white hover:bg-white/5 transition-all duration-150"
          onClick={handleFullscreen}
          title="Fullscreen View"
        >
          <Maximize size={16} />
        </button>
      </div>

      {/* HUD Overlays */}
      <div className="absolute top-4 right-4 z-10 flex flex-col gap-2 pointer-events-none">
        {hasModel && modelInfo?.plyData && (
          <>
            <div className="bg-[#0a0a0b]/60 backdrop-blur-xl px-3 py-2 rounded-lg border border-white/5 shadow-2xl flex items-center justify-between gap-6 w-44">
              <span className="text-[10px] font-bold text-zinc-500 tracking-widest">
                POINTS
              </span>
              <span className="text-xs font-mono text-zinc-200">
                {formatNumber(modelInfo.plyData.count)}
              </span>
            </div>
            <div className="bg-[#0a0a0b]/60 backdrop-blur-xl px-3 py-2 rounded-lg border border-white/5 shadow-2xl flex items-center justify-between gap-6 w-44">
              <span className="text-[10px] font-bold text-zinc-500 tracking-widest">
                FILE
              </span>
              <span className="text-xs font-mono text-zinc-200">
                {formatBytes(modelInfo.plyData.fileSize)}
              </span>
            </div>
          </>
        )}
        <div className="bg-[#0a0a0b]/60 backdrop-blur-xl px-3 py-2 rounded-lg border border-white/5 shadow-2xl flex flex-col gap-1 w-44">
          <span className="text-[10px] font-bold text-zinc-500 tracking-widest">
            CAMERA DIST
          </span>
          <span className="text-[10px] font-mono text-zinc-300">
            {cameraDistance.toFixed(2)}
          </span>
        </div>
        {modelInfo?.fileName && (
          <div className="bg-[#0a0a0b]/60 backdrop-blur-xl px-3 py-2 rounded-lg border border-white/5 shadow-2xl flex items-center justify-between gap-4 w-44">
            <span className="text-[10px] font-bold text-zinc-500 tracking-widest">
              MODEL
            </span>
            <span className="text-[10px] font-mono text-zinc-300 truncate">
              {modelInfo.fileName}
            </span>
          </div>
        )}
      </div>

      {/* Bottom Toolbar — View Presets */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-1 p-1 bg-[#111113]/80 backdrop-blur-2xl border border-white/10 rounded-xl shadow-2xl">
        {VIEW_PRESETS.map((preset) => {
          const Icon = preset.icon;
          return (
            <button
              key={preset.id}
              onClick={() => handleViewPreset(preset)}
              className="px-3 py-1.5 rounded-lg text-[11px] font-medium text-zinc-400 hover:text-white hover:bg-white/5 transition-all duration-150 flex items-center gap-1.5 active:scale-95"
              title={preset.label}
            >
              <Icon size={13} />
              {preset.label}
            </button>
          );
        })}
        <div className="w-[1px] h-4 bg-white/10 mx-1" />
        <button
          onClick={handleResetCamera}
          className="px-3 py-1.5 rounded-lg text-[11px] font-medium text-zinc-400 hover:text-white hover:bg-white/5 transition-all duration-150 flex items-center gap-1.5 active:scale-95"
          title="Reset View"
        >
          <RotateCcw size={13} />
          Reset
        </button>
        <button
          onClick={handleFullscreen}
          className="px-3 py-1.5 rounded-lg text-[11px] font-medium text-zinc-400 hover:text-white hover:bg-white/5 transition-all duration-150 flex items-center gap-1.5 active:scale-95"
          title="Fullscreen"
        >
          <Maximize size={13} />
        </button>
      </div>

      {/* Branding */}
      <div className="absolute bottom-14 left-4 z-10 pointer-events-none">
        <div className="text-2xl font-black text-white/5 tracking-tighter uppercase select-none">
          Face3D Engine
        </div>
        <div className="text-[10px] font-bold text-indigo-400/50 uppercase tracking-[0.2em] mt-1 ml-1">
          {hasModel ? 'Gaussian Splat Viewer' : 'Waiting for Model'}
        </div>
      </div>

      {/* Loading overlay with skeleton */}
      {isLoading && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="flex flex-col items-center gap-4">
            <Loader2 size={40} className="text-indigo-400 animate-spin" />
            <div className="text-sm text-zinc-300 font-medium">
              Loading model...
            </div>
            <div className="text-xs text-zinc-500">
              Parsing PLY binary data
            </div>
            {/* Skeleton loader bars */}
            <div className="w-48 space-y-2">
              <div className="h-2 skeleton rounded-full" />
              <div className="h-2 skeleton rounded-full w-3/4" />
              <div className="h-2 skeleton rounded-full w-1/2" />
            </div>
          </div>
        </div>
      )}

      {/* Error overlay with retry */}
      {loadError && !isLoading && (
        <div className="absolute inset-0 z-20 flex items-center justify-center pointer-events-none">
          <div className="flex flex-col items-center gap-3 p-6 bg-red-950/40 backdrop-blur-xl border border-red-500/20 rounded-2xl max-w-sm pointer-events-auto">
            <AlertTriangle size={32} className="text-red-400" />
            <div className="text-sm text-red-300 font-medium text-center">
              Failed to load model
            </div>
            <div className="text-xs text-red-400/70 text-center">
              {loadError}
            </div>
            <button
              onClick={handleRetryLoad}
              className="mt-2 flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium bg-red-500/10 text-red-300 border border-red-500/20 hover:bg-red-500/20 transition-all duration-150 active:scale-95"
            >
              <RefreshCw size={13} /> Retry
            </button>
          </div>
        </div>
      )}

      {/* No model state — solid dark background, clean message */}
      {noModel && (
        <div className="absolute inset-0 z-30 flex items-center justify-center bg-[#0a0a0b]">
          <div className="flex flex-col items-center gap-5 max-w-sm text-center px-6">
            <div className="w-20 h-20 rounded-2xl bg-zinc-800/40 border border-zinc-700/20 flex items-center justify-center">
              <Box size={32} className="text-zinc-600" />
            </div>
            <div>
              <h3 className="text-lg font-medium text-zinc-300 mb-2">
                No 3D Model Yet
              </h3>
              <p className="text-sm text-zinc-500 leading-relaxed">
                Run the reconstruction pipeline to generate a Gaussian Splat model from your capture data.
              </p>
            </div>
            <div className="flex items-center gap-4 text-xs text-zinc-600">
              <span className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-600" />
                .ply point cloud
              </span>
              <span className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-blue-600" />
                .obj mesh
              </span>
            </div>
          </div>
        </div>
      )}

      {/* R3F Canvas */}
      <Canvas
        camera={{ position: [0, 1.2, 3.5], fov: 45 }}
        gl={{
          antialias: true,
          toneMapping: THREE.ACESFilmicToneMapping,
          alpha: bgMode === 'transparent',
        }}
      >
        <ambientLight intensity={0.3} />
        <directionalLight position={[5, 10, 5]} intensity={1.2} color="#ffffff" />
        <directionalLight position={[-5, 5, -5]} intensity={0.4} color="#6366f1" />

        {/* Cinematic Particles */}
        <Sparkles
          count={200}
          scale={8}
          size={1}
          speed={0.15}
          opacity={0.1}
          color="#818cf8"
        />

        {/* Floor Grid */}
        <Grid
          infiniteGrid
          fadeDistance={15}
          sectionColor="#3f3f46"
          cellColor="#27272a"
          position={[0, 0, 0]}
          sectionSize={1}
          cellSize={0.2}
          sectionThickness={1}
          cellThickness={0.5}
        />

        {/* Axis Gizmo */}
        <GizmoHelper alignment="bottom-left" margin={[60, 60]}>
          <GizmoViewport
            axisColors={['#ef4444', '#22c55e', '#3b82f6']}
            labelColor="#fff"
          />
        </GizmoHelper>

        {/* Model or placeholder */}
        {hasModel && modelInfo?.plyData ? (
          <>
            <PointCloud
              plyData={modelInfo.plyData}
              pointSize={pointSize}
              ghostMode={ghostMode}
            />
            <AutoFit plyData={modelInfo.plyData} sessionId={currentSession?.id} />
          </>
        ) : (
          <SyntheticCloud pointSize={0.015} ghostMode={ghostMode} />
        )}

        <CameraTracker onUpdate={handleCameraUpdate} />
        <CameraSaver sessionId={currentSession?.id} />
        <DoubleClickFocus controlsRef={controlsRef} />

        <OrbitControls
          ref={controlsRef}
          makeDefault
          dampingFactor={0.06}
          enableDamping
          enablePan
          target={hasModel ? undefined : [0, 1, 0]}
        />
      </Canvas>
    </div>
  );
}
