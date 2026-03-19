import { useState, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Environment, Sparkles } from '@react-three/drei';
import { BoxSelect, Maximize, SlidersHorizontal, Layers, Camera, Palette, Video } from 'lucide-react';
import * as THREE from 'three';

// Synthetic representation of a 3D Face Point Cloud
function FacePointCloud({ pointSize, wireframe }: { pointSize: number, wireframe: boolean }) {
  const pointsRef = useRef<THREE.Points>(null);
  
  useFrame(({ clock }) => {
    if (pointsRef.current) {
       // Gentle breathing/floating animation
       pointsRef.current.position.y = Math.sin(clock.getElapsedTime() * 0.5) * 0.02;
    }
  });

  // Generate an ellipsoid point cloud approximating a head
  const count = 80000;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const color = new THREE.Color();
  
  for(let i = 0; i < count; i++) {
     const u = Math.random();
     const v = Math.random();
     const theta = u * 2.0 * Math.PI;
     const phi = Math.acos(2.0 * v - 1.0);
     
     // Ellipsoid radii (head proportion)
     const rx = 0.7;
     const ry = 1.0;
     const rz = 0.8;

     // Base positions
     let x = rx * Math.sin(phi) * Math.cos(theta);
     let y = ry * Math.sin(phi) * Math.sin(theta);
     let z = rz * Math.cos(phi);

     // Flatten the front slightly to simulate a face plane
     if (z > 0.2) z *= 0.6;

     // Add noise
     const noise = (Math.random() - 0.5) * 0.05;
     positions[i*3] = x + noise;
     positions[i*3+1] = y + noise + 1.0; // Raise above grid
     positions[i*3+2] = z + noise;

     // Skin-tone-ish colors based on height and depth
     const hue = 0.05 + (Math.random() * 0.04);
     const saturation = 0.4 + (z * 0.2); // More saturated forward
     const lightness = 0.4 + (y * 0.2) + (Math.random() * 0.1);
     
     color.setHSL(hue, saturation, lightness);
     
     // Highlight "features" arbitrarily based on math
     if (z > 0.4 && y > 0.8 && y < 1.2) { // "Nose" area
       color.setHSL(0.02, 0.6, 0.6);
     }

     colors[i*3] = color.r;
     colors[i*3+1] = color.g;
     colors[i*3+2] = color.b;
  }

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
        opacity={wireframe ? 0.3 : 0.9} 
        sizeAttenuation 
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

export default function Viewer3D() {
  const [pointSize, setPointSize] = useState(0.015);
  const [wireframe, setWireframe] = useState(false);

  return (
    <div className="w-full h-full relative bg-gradient-to-b from-[#0a0a0f] via-[#0d1017] to-[#050505]">
      {/* Floating Toolbar (Top Center) */}
      <div className="absolute top-5 left-1/2 -translate-x-1/2 z-10 flex items-center gap-1 p-1 bg-[#111113]/80 backdrop-blur-2xl border border-white/10 rounded-xl shadow-2xl">
        <button className="p-2 rounded-lg text-zinc-400 hover:text-white hover:bg-white/5 transition-all" title="Select Area">
          <BoxSelect size={16} />
        </button>
        <button 
          className={`p-2 rounded-lg transition-all ${wireframe ? 'bg-indigo-500/20 text-indigo-400' : 'text-zinc-400 hover:text-white hover:bg-white/5'}`} 
          onClick={() => setWireframe(!wireframe)} 
          title="Ghost Mode"
        >
          <Layers size={16} />
        </button>
        
        <div className="w-[1px] h-4 bg-white/10 mx-2" />
        
        <div className="flex items-center gap-3 px-2 text-zinc-400 group" title="Splat Size">
          <SlidersHorizontal size={14} className="group-hover:text-white transition-colors" />
          <input 
            type="range" 
            min="0.005" 
            max="0.05" 
            step="0.001" 
            value={pointSize} 
            onChange={(e) => setPointSize(parseFloat(e.target.value))} 
            className="w-20 h-1 bg-zinc-800 rounded-full appearance-none cursor-pointer accent-indigo-500 opacity-70 group-hover:opacity-100 transition-opacity" 
          />
        </div>
        
        <div className="w-[1px] h-4 bg-white/10 mx-2" />
        
        <button className="p-2 rounded-lg text-zinc-400 hover:text-white hover:bg-white/5 transition-all" title="Theme Colors">
          <Palette size={16} />
        </button>
        <button className="p-2 rounded-lg text-zinc-400 hover:text-white hover:bg-white/5 transition-all" title="Render Video">
          <Video size={16} />
        </button>
        <button className="p-2 rounded-lg text-zinc-400 hover:text-white hover:bg-white/5 transition-all" title="Reset Camera">
          <Camera size={16} />
        </button>
        <button className="p-2 rounded-lg text-zinc-400 hover:text-white hover:bg-white/5 transition-all" title="Fullscreen View">
          <Maximize size={16} />
        </button>
      </div>

      {/* Cinematic HUD Overlays */}
      <div className="absolute top-5 right-5 z-10 flex flex-col gap-2 pointer-events-none">
        <div className="bg-[#0a0a0b]/60 backdrop-blur-xl px-3 py-2 rounded-lg border border-white/5 shadow-2xl flex items-center justify-between gap-6 w-40">
          <span className="text-[10px] font-bold text-zinc-500 tracking-widest">FPS</span>
          <span className="text-xs font-mono text-emerald-400">144.2</span>
        </div>
        <div className="bg-[#0a0a0b]/60 backdrop-blur-xl px-3 py-2 rounded-lg border border-white/5 shadow-2xl flex items-center justify-between gap-6 w-40">
          <span className="text-[10px] font-bold text-zinc-500 tracking-widest">SPLATS</span>
          <span className="text-xs font-mono text-zinc-200">80,000</span>
        </div>
        <div className="bg-[#0a0a0b]/60 backdrop-blur-xl px-3 py-2 rounded-lg border border-white/5 shadow-2xl flex flex-col gap-1 w-40">
          <span className="text-[10px] font-bold text-zinc-500 tracking-widest">CAMERA POS</span>
          <span className="text-[10px] font-mono text-zinc-300">X: 0.00 Y: 1.50 Z: 3.50</span>
        </div>
      </div>

      <div className="absolute bottom-6 left-6 z-10 pointer-events-none">
        <div className="text-2xl font-black text-white/5 tracking-tighter uppercase select-none">
          Face3D Engine
        </div>
        <div className="text-[10px] font-bold text-indigo-400/50 uppercase tracking-[0.2em] mt-1 ml-1">
          Orthographic / Perspective
        </div>
      </div>

      {/* R3F Canvas */}
      <Canvas 
        camera={{ position: [0, 1.2, 3.5], fov: 45 }} 
        gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, alpha: true }}
      >
        <ambientLight intensity={0.2} />
        <directionalLight position={[5, 10, 5]} intensity={1.5} color="#ffffff" />
        <directionalLight position={[-5, 5, -5]} intensity={0.5} color="#6366f1" />
        
        {/* Cinematic Particles */}
        <Sparkles count={300} scale={8} size={1} speed={0.2} opacity={0.15} color="#818cf8" />
        
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
        
        <FacePointCloud pointSize={pointSize} wireframe={wireframe} />
        
        <OrbitControls 
          makeDefault 
          dampingFactor={0.04} 
          enablePan={true}
          maxPolarAngle={Math.PI / 2 + 0.1} 
        />
        <Environment preset="city" />
      </Canvas>
    </div>
  );
}
