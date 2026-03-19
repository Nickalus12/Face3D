import { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float } from '@react-three/drei';
import * as THREE from 'three';

// ── Face geometry generation ─────────────────────────────────────

/** Generate points roughly shaped like a face using parametric math */
function generateFacePoints(count: number): Float32Array {
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const i3 = i * 3;
    // Use a combination of ellipsoids for head shape
    const t = Math.random() * Math.PI * 2;
    const p = Math.random() * Math.PI;
    const r = 0.8 + Math.random() * 0.2;

    // Base ellipsoid (head)
    let x = Math.sin(p) * Math.cos(t) * 0.7 * r;
    let y = Math.cos(p) * 1.0 * r;
    let z = Math.sin(p) * Math.sin(t) * 0.65 * r;

    // Flatten the back
    if (z < -0.2) z *= 0.5;

    // Chin taper
    if (y < -0.4) {
      const chinFactor = 1 - Math.abs(y + 0.4) * 0.8;
      x *= Math.max(0.3, chinFactor);
      z *= Math.max(0.4, chinFactor);
    }

    // Nose bump
    if (Math.random() < 0.08 && Math.abs(x) < 0.15 && y > -0.2 && y < 0.2) {
      z += 0.2 + Math.random() * 0.15;
    }

    // Eye sockets (subtle indentations are implied by density)
    positions[i3] = x;
    positions[i3 + 1] = y;
    positions[i3 + 2] = z;
  }
  return positions;
}

/** Generate random scattered positions */
function generateScatteredPoints(count: number): Float32Array {
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const i3 = i * 3;
    positions[i3] = (Math.random() - 0.5) * 4;
    positions[i3 + 1] = (Math.random() - 0.5) * 4;
    positions[i3 + 2] = (Math.random() - 0.5) * 4;
  }
  return positions;
}

/** Generate wireframe edges from face points (connect nearest neighbors) */
function generateWireframeEdges(facePositions: Float32Array, edgeCount: number): Float32Array {
  const pointCount = facePositions.length / 3;
  const edges = new Float32Array(edgeCount * 6); // 2 vertices per edge, 3 components each
  let edgeIdx = 0;

  for (let i = 0; i < pointCount && edgeIdx < edgeCount; i++) {
    const ix = facePositions[i * 3];
    const iy = facePositions[i * 3 + 1];
    const iz = facePositions[i * 3 + 2];

    // Connect to a few nearby points
    let bestDist = Infinity;
    let bestJ = -1;
    const skip = Math.max(1, Math.floor(pointCount / 200));
    for (let j = i + 1; j < pointCount; j += skip) {
      const dx = facePositions[j * 3] - ix;
      const dy = facePositions[j * 3 + 1] - iy;
      const dz = facePositions[j * 3 + 2] - iz;
      const d = dx * dx + dy * dy + dz * dz;
      if (d < bestDist && d < 0.15) {
        bestDist = d;
        bestJ = j;
      }
    }

    if (bestJ >= 0 && edgeIdx < edgeCount) {
      const e6 = edgeIdx * 6;
      edges[e6] = ix;
      edges[e6 + 1] = iy;
      edges[e6 + 2] = iz;
      edges[e6 + 3] = facePositions[bestJ * 3];
      edges[e6 + 4] = facePositions[bestJ * 3 + 1];
      edges[e6 + 5] = facePositions[bestJ * 3 + 2];
      edgeIdx++;
    }
  }

  return edges.slice(0, edgeIdx * 6);
}

// ── Phase constants ──────────────────────────────────────────────

type Phase = 0 | 1 | 2 | 3 | 4;

function stageToPhase(completedStages: number): Phase {
  if (completedStages <= 3) return 0;  // Particle cloud
  if (completedStages <= 6) return 1;  // Point cloud structure
  if (completedStages <= 9) return 2;  // Wireframe mesh
  if (completedStages <= 11) return 3; // FLAME mesh
  return 4;                             // Gaussian splat
}

// ── Particle system ──────────────────────────────────────────────

const PARTICLE_COUNT = 2000;
const WIREFRAME_EDGE_COUNT = 600;

interface FaceParticlesProps {
  completedStages: number;
}

function FaceParticles({ completedStages }: FaceParticlesProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const wireRef = useRef<THREE.LineSegments>(null);
  const surfaceRef = useRef<THREE.Mesh>(null);
  const groupRef = useRef<THREE.Group>(null);

  const phase = stageToPhase(completedStages);
  const phaseRef = useRef<Phase>(phase);
  const transitionRef = useRef(1); // 0..1 transition progress
  const celebrationRef = useRef(0);

  // Detect phase change
  useEffect(() => {
    if (phase !== phaseRef.current) {
      phaseRef.current = phase;
      transitionRef.current = 0;
      if (phase === 4) celebrationRef.current = 1;
    }
  }, [phase]);

  // Pre-compute geometry
  const { scatteredPos, facePos, wireframeEdges, colors, dummy, landmarkIndices } = useMemo(() => {
    const scattered = generateScatteredPoints(PARTICLE_COUNT);
    const face = generateFacePoints(PARTICLE_COUNT);
    const edges = generateWireframeEdges(face, WIREFRAME_EDGE_COUNT);

    // Depth-based colors for point cloud phase
    const cols = new Float32Array(PARTICLE_COUNT * 3);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const z = face[i * 3 + 2];
      const t = (z + 1) / 2; // normalize z to 0..1
      // Near (warm) to far (cool)
      cols[i * 3] = 0.3 + t * 0.6;     // R
      cols[i * 3 + 1] = 0.2 + (1 - t) * 0.5; // G
      cols[i * 3 + 2] = 0.5 + (1 - t) * 0.5; // B
    }

    // Pick ~50 points as "landmarks" for phase 2
    const landmarks: number[] = [];
    for (let i = 0; i < PARTICLE_COUNT; i += Math.floor(PARTICLE_COUNT / 50)) {
      landmarks.push(i);
    }

    return {
      scatteredPos: scattered,
      facePos: face,
      wireframeEdges: edges,
      colors: cols,
      dummy: new THREE.Object3D(),
      landmarkIndices: landmarks,
    };
  }, []);

  // Current interpolated positions
  const currentPos = useRef(new Float32Array(PARTICLE_COUNT * 3));

  // Initialize positions
  useEffect(() => {
    currentPos.current.set(scatteredPos);
  }, [scatteredPos]);

  // Color attribute for instanced mesh
  const colorAttr = useMemo(() => {
    const arr = new Float32Array(PARTICLE_COUNT * 3);
    arr.fill(0.5); // zinc gray default
    return arr;
  }, []);

  // Wireframe geometry
  const wireGeom = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(wireframeEdges, 3));
    return geom;
  }, [wireframeEdges]);

  useFrame((state, delta) => {
    if (!meshRef.current) return;

    const mesh = meshRef.current;
    const t = state.clock.elapsedTime;

    // Advance transition
    if (transitionRef.current < 1) {
      transitionRef.current = Math.min(1, transitionRef.current + delta * 2); // ~500ms
    }

    // Advance celebration decay
    if (celebrationRef.current > 0) {
      celebrationRef.current = Math.max(0, celebrationRef.current - delta * 0.5);
    }

    const transition = transitionRef.current;
    const currentPhase = phaseRef.current;

    // Determine target positions and sizes per particle
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const i3 = i * 3;
      let tx: number, ty: number, tz: number;
      let scale = 0.008;
      let cr = 0.45, cg = 0.45, cb = 0.5; // zinc default

      if (currentPhase === 0) {
        // Scattered particles drifting toward vague clusters
        const clusterProgress = Math.min(1, completedStages / 4);
        tx = scatteredPos[i3] * (1 - clusterProgress * 0.5) + facePos[i3] * clusterProgress * 0.3;
        ty = scatteredPos[i3 + 1] * (1 - clusterProgress * 0.5) + facePos[i3 + 1] * clusterProgress * 0.3;
        tz = scatteredPos[i3 + 2] * (1 - clusterProgress * 0.5) + facePos[i3 + 2] * clusterProgress * 0.3;
        // Gentle drift
        tx += Math.sin(t * 0.3 + i * 0.1) * 0.02;
        ty += Math.cos(t * 0.2 + i * 0.15) * 0.02;
        scale = 0.006 + Math.sin(t + i) * 0.002;
        // Subtle indigo glow
        cr = 0.35; cg = 0.35; cb = 0.5;
      } else if (currentPhase === 1) {
        // Snap into face point cloud
        tx = facePos[i3];
        ty = facePos[i3 + 1];
        tz = facePos[i3 + 2];
        scale = 0.007;
        // Depth coloring
        cr = colors[i3]; cg = colors[i3 + 1]; cb = colors[i3 + 2];
      } else if (currentPhase === 2) {
        // Face structure with landmarks highlighted
        tx = facePos[i3];
        ty = facePos[i3 + 1];
        tz = facePos[i3 + 2];
        const isLandmark = landmarkIndices.includes(i);
        if (isLandmark) {
          scale = 0.012 + Math.sin(t * 3 + i) * 0.003;
          cr = 0.2; cg = 0.9; cb = 0.4; // green landmarks
        } else {
          scale = 0.005;
          cr = 0.4; cg = 0.4; cb = 0.45;
        }
      } else if (currentPhase === 3) {
        // FLAME mesh — particles form smooth surface
        tx = facePos[i3];
        ty = facePos[i3 + 1];
        tz = facePos[i3 + 2];
        scale = 0.009;
        // Warm skin-like tone
        cr = 0.75; cg = 0.55; cb = 0.45;
      } else {
        // Gaussian splats — soft colored particles
        tx = facePos[i3] + (Math.random() - 0.5) * 0.01;
        ty = facePos[i3 + 1] + (Math.random() - 0.5) * 0.01;
        tz = facePos[i3 + 2];
        scale = 0.01 + Math.sin(t * 0.5 + i * 0.3) * 0.004;
        // Varied warm colors
        const hue = (i / PARTICLE_COUNT) * 0.15 + 0.0;
        cr = 0.7 + Math.sin(hue * Math.PI * 2) * 0.2;
        cg = 0.5 + Math.cos(hue * Math.PI * 2) * 0.15;
        cb = 0.4 + Math.sin(hue * Math.PI * 4) * 0.1;

        // Celebration sparkle
        if (celebrationRef.current > 0) {
          const sparkle = Math.sin(t * 8 + i * 2) * celebrationRef.current;
          scale += sparkle * 0.008;
          cr = Math.min(1, cr + sparkle * 0.3);
          cg = Math.min(1, cg + sparkle * 0.3);
          cb = Math.min(1, cb + sparkle * 0.3);
        }
      }

      // Smooth lerp from current to target
      const lerp = Math.min(1, transition);
      currentPos.current[i3] += (tx - currentPos.current[i3]) * lerp;
      currentPos.current[i3 + 1] += (ty - currentPos.current[i3 + 1]) * lerp;
      currentPos.current[i3 + 2] += (tz - currentPos.current[i3 + 2]) * lerp;

      // Update instance matrix
      dummy.position.set(currentPos.current[i3], currentPos.current[i3 + 1], currentPos.current[i3 + 2]);
      dummy.scale.setScalar(scale);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

      // Update color
      colorAttr[i3] = cr;
      colorAttr[i3 + 1] = cg;
      colorAttr[i3 + 2] = cb;
    }

    mesh.instanceMatrix.needsUpdate = true;

    // Update instanced color
    const colorAttribute = mesh.geometry.getAttribute('instanceColor');
    if (colorAttribute) {
      (colorAttribute as THREE.BufferAttribute).set(colorAttr);
      colorAttribute.needsUpdate = true;
    }

    // Wireframe visibility — phases 2-3
    if (wireRef.current) {
      const wireVisible = currentPhase === 2 || currentPhase === 3;
      wireRef.current.visible = wireVisible;
      if (wireVisible) {
        const mat = wireRef.current.material as THREE.LineBasicMaterial;
        const pulse = 0.15 + Math.sin(t * 2) * 0.05;
        mat.opacity = currentPhase === 2 ? pulse + 0.15 : pulse * 0.5;
      }
    }

    // Surface mesh visibility — phase 3 only
    if (surfaceRef.current) {
      surfaceRef.current.visible = currentPhase === 3;
      if (currentPhase === 3) {
        const mat = surfaceRef.current.material as THREE.MeshStandardMaterial;
        mat.opacity = transition * 0.15;
      }
    }

    // Slow orbit
    if (groupRef.current) {
      groupRef.current.rotation.y = t * 0.15;
    }
  });

  return (
    <group ref={groupRef}>
      {/* Instanced particles */}
      <instancedMesh
        ref={meshRef}
        args={[undefined, undefined, PARTICLE_COUNT]}
        frustumCulled={false}
      >
        <sphereGeometry args={[1, 6, 6]}>
          <instancedBufferAttribute
            attach="attributes-instanceColor"
            args={[colorAttr, 3]}
          />
        </sphereGeometry>
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>

      {/* Wireframe overlay */}
      <lineSegments ref={wireRef} geometry={wireGeom} visible={false}>
        <lineBasicMaterial
          color="#6366f1"
          transparent
          opacity={0.2}
          depthWrite={false}
        />
      </lineSegments>

      {/* Subtle surface mesh for FLAME phase */}
      <mesh ref={surfaceRef} visible={false}>
        <sphereGeometry args={[0.85, 32, 24]} />
        <meshStandardMaterial
          color="#d4a574"
          transparent
          opacity={0}
          roughness={0.8}
          metalness={0.1}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

// ── Ambient glow ring ────────────────────────────────────────────

function GlowRing({ phase }: { phase: Phase }) {
  const ref = useRef<THREE.Mesh>(null);

  const color = useMemo(() => {
    switch (phase) {
      case 0: return '#3f3f46'; // zinc
      case 1: return '#6366f1'; // indigo
      case 2: return '#22c55e'; // green
      case 3: return '#f97316'; // orange
      case 4: return '#8b5cf6'; // purple celebration
    }
  }, [phase]);

  useFrame((state) => {
    if (!ref.current) return;
    const t = state.clock.elapsedTime;
    const mat = ref.current.material as THREE.MeshBasicMaterial;
    mat.opacity = 0.03 + Math.sin(t * 1.5) * 0.015;
    ref.current.scale.setScalar(1.8 + Math.sin(t * 0.8) * 0.1);
  });

  return (
    <mesh ref={ref} position={[0, 0, -0.5]} rotation={[0, 0, 0]}>
      <ringGeometry args={[1.2, 1.6, 64]} />
      <meshBasicMaterial
        color={color}
        transparent
        opacity={0.03}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  );
}

// ── Main component ───────────────────────────────────────────────

interface PipelineVisualizerProps {
  completedStages: number;
  className?: string;
  /** Compact mode for small containers (e.g., WelcomeScreen hero) */
  compact?: boolean;
}

export default function PipelineVisualizer({
  completedStages,
  className = '',
  compact = false,
}: PipelineVisualizerProps) {
  const phase = stageToPhase(completedStages);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const phaseLabel = useMemo(() => {
    switch (phase) {
      case 0: return 'Capturing';
      case 1: return 'Structuring';
      case 2: return 'Meshing';
      case 3: return 'Fitting';
      case 4: return 'Splatting';
    }
  }, [phase]);

  return (
    <div className={`relative ${className}`}>
      <div
        className={`w-full transition-opacity duration-700 ${mounted ? 'opacity-100' : 'opacity-0'}`}
        style={{ aspectRatio: compact ? '1' : '16/10' }}
      >
        <Canvas
          camera={{ position: [0, 0, 3], fov: compact ? 40 : 35 }}
          dpr={[1, 1.5]}
          gl={{ antialias: true, alpha: true, powerPreference: 'low-power' }}
          style={{ background: 'transparent' }}
        >
          <ambientLight intensity={0.4} />
          <pointLight position={[2, 2, 3]} intensity={0.6} color="#818cf8" />
          <pointLight position={[-2, -1, 2]} intensity={0.3} color="#6366f1" />

          <Float speed={1.5} rotationIntensity={0.1} floatIntensity={0.3}>
            <FaceParticles completedStages={completedStages} />
          </Float>

          <GlowRing phase={phase} />
        </Canvas>
      </div>

      {/* Phase label overlay */}
      {!compact && (
        <div className="absolute bottom-3 left-0 right-0 flex justify-center pointer-events-none">
          <div className="px-3 py-1 rounded-full bg-zinc-900/70 border border-zinc-800/50 backdrop-blur-sm">
            <span className="text-[11px] font-mono text-zinc-500 uppercase tracking-widest">
              {phaseLabel}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
