import { Upload, Clock, ArrowRight, Video, Cpu, Box, Sparkles, Layers, CheckCircle2 } from 'lucide-react';
import useSessionStore from '../store/sessionStore';


interface WelcomeScreenProps {
  onNewScan: () => void;
  onSelectSession: (session: any) => void;
}

export default function WelcomeScreen({ onNewScan, onSelectSession }: WelcomeScreenProps) {
  const { sessions } = useSessionStore();
  const recentSessions = sessions.slice(0, 5);

  return (
    <div className="flex-1 flex items-center justify-center bg-[#0a0a0b] p-8 relative overflow-hidden">
      {/* Subtle background gradient orbs */}
      <div className="absolute top-1/4 left-1/3 w-[500px] h-[500px] bg-indigo-500/[0.03] rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/3 w-[400px] h-[400px] bg-purple-500/[0.02] rounded-full blur-[100px] pointer-events-none" />

      <div className="max-w-xl w-full animate-fadeIn relative z-10">
        {/* Logo & Title */}
        <div className="text-center mb-14">
          <div className="inline-flex items-center justify-center w-24 h-24 bg-gradient-to-br from-indigo-500 via-purple-500 to-indigo-600 rounded-[26px] shadow-2xl shadow-indigo-500/25 mb-7 ring-1 ring-white/10 animate-scaleIn hover:scale-105 transition-transform duration-500">
            <span className="text-white font-black text-3xl tracking-tighter">F3D</span>
          </div>
          <h1 className="text-4xl font-bold text-zinc-100 tracking-tight">Face3D</h1>
          <p className="text-[15px] text-zinc-500 mt-4 leading-relaxed max-w-sm mx-auto">
            Next-gen 3D face reconstruction pipeline
          </p>
        </div>

        {/* Two action cards */}
        <div className="grid grid-cols-2 gap-5 mb-10 animate-stagger">
          {/* New Scan card */}
          <button
            onClick={onNewScan}
            className="group p-6 rounded-2xl border border-zinc-800 bg-zinc-900/40 hover:border-indigo-500/40 hover:bg-indigo-500/[0.04] hover:shadow-lg hover:shadow-indigo-500/5 hover:-translate-y-0.5 transition-all duration-300 text-left active:scale-[0.98] card-glow"
          >
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500/20 to-purple-500/10 border border-indigo-500/20 flex items-center justify-center mb-4 group-hover:scale-110 group-hover:shadow-lg group-hover:shadow-indigo-500/10 transition-all duration-300">
              <Upload size={20} className="text-indigo-400" />
            </div>
            <h3 className="text-base font-semibold text-zinc-200 mb-1.5 group-hover:text-white transition-colors">New Scan</h3>
            <p className="text-[13px] text-zinc-500 leading-relaxed">
              Import video, photos, and sensor data to start reconstruction
            </p>
            <div className="flex items-center gap-1 mt-3 text-xs text-indigo-400/70 group-hover:text-indigo-400 transition-colors">
              <span>Get started</span>
              <ArrowRight size={12} className="group-hover:translate-x-1 transition-transform duration-200" />
            </div>
          </button>

          {/* Recent Sessions card */}
          <div className="p-6 rounded-2xl border border-zinc-800 bg-zinc-900/40 card-glow">
            <div className="w-12 h-12 rounded-xl bg-zinc-800/80 border border-zinc-700/50 flex items-center justify-center mb-4">
              <Clock size={20} className="text-zinc-400" />
            </div>
            <h3 className="text-base font-semibold text-zinc-200 mb-3">Recent Sessions</h3>
            {recentSessions.length === 0 ? (
              <div className="py-3 text-center">
                <p className="text-[13px] text-zinc-600">No sessions yet</p>
                <p className="text-xs text-zinc-700 mt-1">Start a new scan to create one</p>
              </div>
            ) : (
              <div className="space-y-0.5">
                {recentSessions.map((session) => (
                  <button
                    key={session.id}
                    onClick={() => onSelectSession(session)}
                    className="w-full flex items-center justify-between py-2 px-2.5 -mx-2.5 rounded-lg text-[13px] text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.04] transition-all duration-150 group/item"
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      {session.has_gaussians ? (
                        <CheckCircle2 size={12} className="text-emerald-500 shrink-0" />
                      ) : (
                        <div className="w-3 h-3 rounded-full border border-zinc-700 shrink-0" />
                      )}
                      <span className="truncate">{session.name}</span>
                    </div>
                    <ArrowRight size={12} className="text-zinc-700 group-hover/item:text-zinc-400 shrink-0 transition-all duration-150 group-hover/item:translate-x-0.5" />
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Pipeline flow visualization */}
        <div className="flex items-center justify-center gap-3 text-xs text-zinc-600 animate-fadeIn" style={{ animationDelay: '200ms' }}>
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-zinc-900/50 border border-zinc-800/50">
            <Video size={13} className="text-zinc-500" />
            <span>Video + Photos</span>
          </div>
          <div className="flex items-center">
            <div className="w-6 h-px bg-gradient-to-r from-zinc-700 to-zinc-600" />
            <ArrowRight size={12} className="text-zinc-600 -ml-1" />
          </div>
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-zinc-900/50 border border-zinc-800/50">
            <Cpu size={13} className="text-zinc-500" />
            <span>15-Stage Pipeline</span>
          </div>
          <div className="flex items-center">
            <div className="w-6 h-px bg-gradient-to-r from-zinc-600 to-indigo-500/50" />
            <ArrowRight size={12} className="text-indigo-500/50 -ml-1" />
          </div>
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-indigo-500/[0.06] border border-indigo-500/15">
            <Sparkles size={13} className="text-indigo-400" />
            <span className="text-indigo-400">3D Gaussian Splat</span>
          </div>
        </div>
      </div>
    </div>
  );
}
