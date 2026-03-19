import React, { useState } from 'react';
import { Image as ImageIcon, X, Expand, SplitSquareHorizontal } from 'lucide-react';

export const Gallery: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<number | null>(null);
  const [sliderPos, setSliderPos] = useState(50);

  // Mocks
  const renders = Array.from({ length: 9 }, (_, i) => ({
    id: i,
    url: `https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=600&auto=format&fit=crop&sig=${i}`, // Placeholder base
    gt_url: `https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=600&auto=format&fit=crop&blur=10&sig=${i}`, // Placeholder blurred GT
    psnr: (30 + Math.random() * 5).toFixed(2),
    ssim: (0.85 + Math.random() * 0.1).toFixed(3)
  }));

  if (renders.length === 0) {
    return (
      <div className="flex-1 bg-[#0a0a0b] flex flex-col items-center justify-center text-zinc-500">
        <ImageIcon size={48} className="mb-4 opacity-20" />
        <p className="text-sm">No renders available yet.</p>
        <p className="text-xs mt-1">Start training to see outputs here.</p>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-[#0a0a0b] p-8 overflow-y-auto [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:bg-zinc-800 [&::-webkit-scrollbar-track]:bg-transparent">
      
      {/* Masonry Grid */}
      <div className="columns-2 md:columns-3 lg:columns-4 gap-4 space-y-4">
        {renders.map((render) => (
          <div 
            key={render.id} 
            className="group relative rounded-xl overflow-hidden border border-zinc-800 bg-zinc-900 cursor-pointer hover:border-emerald-500/50 transition-colors break-inside-avoid"
            onClick={() => setSelectedImage(render.id)}
          >
            <img src={render.url} alt={`Render ${render.id}`} className="w-full object-cover" />
            
            {/* Overlay Info */}
            <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-end p-4">
              <div className="flex items-center gap-3 text-xs font-mono text-zinc-200">
                <span className="bg-zinc-900/80 px-2 py-1 rounded border border-zinc-700">PSNR {render.psnr}</span>
                <span className="bg-zinc-900/80 px-2 py-1 rounded border border-zinc-700">SSIM {render.ssim}</span>
              </div>
            </div>
            
            <div className="absolute top-3 right-3 p-1.5 bg-black/50 backdrop-blur rounded-lg text-white opacity-0 group-hover:opacity-100 transition-opacity">
              <Expand size={14} />
            </div>
          </div>
        ))}
      </div>

      {/* Lightbox / Comparison Slider */}
      {selectedImage !== null && (
        <div className="fixed inset-0 z-50 bg-black/95 backdrop-blur-sm flex items-center justify-center">
          
          <div className="absolute top-6 left-6 flex gap-4 text-white">
            <div className="bg-zinc-900/80 px-4 py-2 rounded-lg border border-zinc-700 font-mono text-sm shadow-xl flex items-center gap-2">
              <SplitSquareHorizontal size={16} className="text-emerald-400" />
              Interactive Comparison
            </div>
          </div>

          <button 
            className="absolute top-6 right-6 p-2 bg-zinc-900/50 hover:bg-zinc-800 rounded-full text-zinc-400 hover:text-white transition-colors border border-zinc-800"
            onClick={() => setSelectedImage(null)}
          >
            <X size={24} />
          </button>

          {/* Slider Container */}
          <div className="relative w-full max-w-5xl aspect-video rounded-lg overflow-hidden border border-zinc-800 shadow-2xl bg-zinc-900 select-none">
            
            {/* Bottom Image (Ground Truth) */}
            <img 
              src={renders[selectedImage].gt_url} 
              alt="Ground Truth" 
              className="absolute inset-0 w-full h-full object-cover pointer-events-none"
            />
            
            {/* Top Image (Rendered) with Clip Path */}
            <img 
              src={renders[selectedImage].url} 
              alt="Rendered" 
              className="absolute inset-0 w-full h-full object-cover pointer-events-none"
              style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
            />

            {/* Labels */}
            <div className="absolute bottom-4 left-4 bg-black/60 backdrop-blur px-3 py-1 rounded text-xs font-medium text-emerald-400 border border-emerald-500/30">
              Rendered
            </div>
            <div className="absolute bottom-4 right-4 bg-black/60 backdrop-blur px-3 py-1 rounded text-xs font-medium text-zinc-400 border border-zinc-700">
              Ground Truth
            </div>

            {/* Divider Line */}
            <div 
              className="absolute top-0 bottom-0 w-1 bg-white cursor-ew-resize flex items-center justify-center shadow-[0_0_10px_rgba(0,0,0,0.5)] z-20 pointer-events-none"
              style={{ left: `calc(${sliderPos}% - 2px)` }}
            >
              <div className="w-6 h-8 bg-white rounded flex items-center justify-center shadow-lg">
                <div className="flex gap-0.5">
                  <div className="w-0.5 h-4 bg-zinc-400 rounded-full" />
                  <div className="w-0.5 h-4 bg-zinc-400 rounded-full" />
                </div>
              </div>
            </div>

            {/* Invisible Range Input for Native Dragging */}
            <input 
              type="range" 
              min="0" max="100" 
              value={sliderPos}
              onChange={(e) => setSliderPos(Number(e.target.value))}
              className="absolute inset-0 w-full h-full opacity-0 cursor-ew-resize z-30 m-0 p-0"
            />
          </div>

          <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex gap-8 font-mono text-sm text-zinc-400 bg-zinc-900/80 px-6 py-3 rounded-xl border border-zinc-800">
            <div>PSNR: <span className="text-white font-semibold">{renders[selectedImage].psnr}</span></div>
            <div>SSIM: <span className="text-white font-semibold">{renders[selectedImage].ssim}</span></div>
          </div>
        </div>
      )}
    </div>
  );
};
