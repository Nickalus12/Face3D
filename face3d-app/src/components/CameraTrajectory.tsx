import { useEffect, useState, useMemo, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { Compass, RotateCcw, Target } from 'lucide-react';
import useSessionStore from '../store/sessionStore';
import { getSensorData } from '../lib/tauri';

/**
 * Convert an orientation quaternion (wxyz) to a position on a unit sphere.
 * We extract the camera "forward" direction from each quaternion and treat
 * it as a position, giving a visual representation of the camera path.
 */
function quatToPosition(qw: number, qx: number, qy: number, qz: number): [number, number, number] {
  // Build rotation matrix elements needed for the forward (Z) column
  const x2 = qx + qx, y2 = qy + qy;
  const xx = qx * x2, xz = qx * (qz + qz);
  const yy = qy * y2, yz = qy * (qz + qz);
  const wx = qw * x2, wy = qw * y2;

  // Forward direction (camera -Z in world space) = third column of rotation matrix
  const fx = xz + wy;
  const fy = yz - wx;
  const fz = 1 - (xx + yy);

  // Place on a sphere of radius 2
  const len = Math.sqrt(fx * fx + fy * fy + fz * fz) || 1;
  return [
    (fx / len) * 2,
    (fy / len) * 2,
    (fz / len) * 2,
  ];
}

/**
 * Color interpolation from blue -> green -> yellow -> red based on t (0..1).
 */
function timeColor(t: number): THREE.Color {
  if (t < 0.33) {
    const s = t / 0.33;
    return new THREE.Color().setRGB(0, s, 1 - s);
  } else if (t < 0.66) {
    const s = (t - 0.33) / 0.33;
    return new THREE.Color().setRGB(s, 1, 0);
  } else {
    const s = (t - 0.66) / 0.34;
    return new THREE.Color().setRGB(1, 1 - s, 0);
  }
}

interface OrientationPoint {
  time: number;
  qw: number;
  qx: number;
  qy: number;
  qz: number;
  position: [number, number, number];
}

function TrajectoryLine({ points }: { points: OrientationPoint[] }) {
  const lineRef = useRef<THREE.Line>(null);

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(points.length * 3);
    const colors = new Float32Array(points.length * 3);

    for (let i = 0; i < points.length; i++) {
      const p = points[i].position;
      positions[i * 3] = p[0];
      positions[i * 3 + 1] = p[1];
      positions[i * 3 + 2] = p[2];

      const t = points.length > 1 ? i / (points.length - 1) : 0;
      const color = timeColor(t);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }

    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    return geo;
  }, [points]);

  const lineObject = useMemo(() => {
    const mat = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 });
    return new THREE.Line(geometry, mat);
  }, [geometry]);

  return (
    <primitive object={lineObject} ref={lineRef as any} />
  );
}

function StartEndMarkers({ points }: { points: OrientationPoint[] }) {
  if (points.length < 2) return null;

  const start = points[0].position;
  const end = points[points.length - 1].position;

  return (
    <>
      {/* Start marker - blue */}
      <mesh position={start}>
        <sphereGeometry args={[0.08, 16, 16]} />
        <meshStandardMaterial color="#3b82f6" emissive="#3b82f6" emissiveIntensity={0.5} />
      </mesh>
      {/* End marker - red */}
      <mesh position={end}>
        <sphereGeometry args={[0.08, 16, 16]} />
        <meshStandardMaterial color="#ef4444" emissive="#ef4444" emissiveIntensity={0.5} />
      </mesh>
    </>
  );
}

function OriginSphere() {
  return (
    <mesh position={[0, 0, 0]}>
      <sphereGeometry args={[0.12, 32, 32]} />
      <meshStandardMaterial
        color="#6366f1"
        emissive="#6366f1"
        emissiveIntensity={0.3}
        transparent
        opacity={0.3}
      />
    </mesh>
  );
}

function AxisHelper() {
  return (
    <group>
      <arrowHelper args={[new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), 0.5, 0xff4444, 0.08, 0.04]} />
      <arrowHelper args={[new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 0), 0.5, 0x44ff44, 0.08, 0.04]} />
      <arrowHelper args={[new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), 0.5, 0x4444ff, 0.08, 0.04]} />
    </group>
  );
}

function Scene({ points }: { points: OrientationPoint[] }) {
  return (
    <>
      <ambientLight intensity={0.6} />
      <pointLight position={[5, 5, 5]} intensity={1} />
      <OriginSphere />
      <AxisHelper />
      <TrajectoryLine points={points} />
      <StartEndMarkers points={points} />
      <OrbitControls
        enableDamping
        dampingFactor={0.1}
        rotateSpeed={0.5}
        minDistance={1}
        maxDistance={8}
      />
    </>
  );
}

export default function CameraTrajectory() {
  const { currentSession } = useSessionStore();
  const [orientationData, setOrientationData] = useState<OrientationPoint[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [trajectoryStats, setTrajectoryStats] = useState<{
    totalRotation: number;
    coverageAngle: number;
    quality: string;
  } | null>(null);

  const sessionId = currentSession?.id;

  useEffect(() => {
    if (!sessionId) {
      setOrientationData([]);
      setTrajectoryStats(null);
      return;
    }

    let cancelled = false;
    setIsLoading(true);

    (async () => {
      const raw = await getSensorData(sessionId, 'orientation');
      if (cancelled) return;

      if (!raw || raw.length === 0) {
        setOrientationData([]);
        setTrajectoryStats(null);
        setIsLoading(false);
        return;
      }

      // raw is [[t, qw, qx, qy, qz], ...]
      const points: OrientationPoint[] = raw.map((r) => {
        const [time, qw, qx, qy, qz] = r;
        return {
          time,
          qw,
          qx,
          qy,
          qz,
          position: quatToPosition(qw, qx, qy, qz),
        };
      });

      setOrientationData(points);

      // Compute trajectory stats
      if (points.length >= 2) {
        let totalAngle = 0;
        for (let i = 1; i < points.length; i++) {
          // Relative quaternion: q_rel = q_prev^-1 * q_curr
          const p = points[i - 1];
          const c = points[i];
          // Inverse of previous (conjugate for unit quaternion)
          const pInvW = p.qw, pInvX = -p.qx, pInvY = -p.qy, pInvZ = -p.qz;
          // Multiply: q_rel = pInv * c
          const rW = pInvW * c.qw - pInvX * c.qx - pInvY * c.qy - pInvZ * c.qz;
          // Angle from relative quaternion
          const angle = 2 * Math.acos(Math.min(1, Math.abs(rW)));
          totalAngle += angle;
        }
        const totalDeg = (totalAngle * 180) / Math.PI;

        // Net rotation (first to last)
        const f = points[0];
        const l = points[points.length - 1];
        const fInvW = f.qw, fInvX = -f.qx, fInvY = -f.qy, fInvZ = -f.qz;
        const netW = fInvW * l.qw - fInvX * l.qx - fInvY * l.qy - fInvZ * l.qz;
        const netAngle = (2 * Math.acos(Math.min(1, Math.abs(netW))) * 180) / Math.PI;

        let quality = 'Limited';
        if (netAngle >= 270) quality = 'Excellent';
        else if (netAngle >= 180) quality = 'Good';
        else if (netAngle >= 90) quality = 'Fair';

        setTrajectoryStats({
          totalRotation: Math.round(totalDeg),
          coverageAngle: Math.round(netAngle),
          quality,
        });
      }

      setIsLoading(false);
    })();

    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  const hasData = orientationData.length > 0;

  return (
    <div className="bg-zinc-900/60 border border-zinc-800/50 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800/40">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-md bg-purple-500/15 text-purple-400">
            <Compass size={14} />
          </div>
          <div>
            <div className="text-xs font-medium text-zinc-200">Camera Trajectory</div>
            <div className="text-[10px] text-zinc-500">3D orientation path from sensor data</div>
          </div>
        </div>

        {trajectoryStats && (
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-[11px]">
              <RotateCcw size={12} className="text-zinc-500" />
              <span className="text-zinc-300 font-mono">{trajectoryStats.totalRotation}</span>
              <span className="text-zinc-500">deg total</span>
            </div>
            <div className="flex items-center gap-1.5 text-[11px]">
              <Target size={12} className="text-zinc-500" />
              <span className="text-zinc-300 font-mono">{trajectoryStats.coverageAngle}</span>
              <span className="text-zinc-500">deg net</span>
            </div>
            <div
              className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
                trajectoryStats.quality === 'Excellent'
                  ? 'bg-emerald-500/15 text-emerald-400'
                  : trajectoryStats.quality === 'Good'
                  ? 'bg-blue-500/15 text-blue-400'
                  : trajectoryStats.quality === 'Fair'
                  ? 'bg-amber-500/15 text-amber-400'
                  : 'bg-red-500/15 text-red-400'
              }`}
            >
              {trajectoryStats.quality}
            </div>
          </div>
        )}
      </div>

      {/* 3D Viewport */}
      <div className="h-[300px] bg-[#08080a]">
        {isLoading ? (
          <div className="h-full flex items-center justify-center">
            <div className="w-5 h-5 border-2 border-zinc-600 border-t-zinc-300 rounded-full animate-spin" />
          </div>
        ) : hasData ? (
          <Canvas camera={{ position: [3, 2, 3], fov: 50 }}>
            <Scene points={orientationData} />
          </Canvas>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-zinc-600">
            <Compass size={32} strokeWidth={1} className="text-zinc-700 mb-2" />
            <p className="text-xs">No orientation data available</p>
          </div>
        )}
      </div>

      {/* Legend */}
      {hasData && (
        <div className="flex items-center justify-between px-4 py-2 border-t border-zinc-800/40">
          <div className="flex items-center gap-4 text-[10px] text-zinc-500">
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-blue-500" />
              <span>Start</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-8 h-0.5 bg-gradient-to-r from-blue-500 via-green-500 to-red-500 rounded-full" />
              <span>Time progression</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-red-500" />
              <span>End</span>
            </div>
          </div>
          <div className="text-[10px] text-zinc-600">
            {orientationData.length} samples | Drag to rotate
          </div>
        </div>
      )}
    </div>
  );
}
