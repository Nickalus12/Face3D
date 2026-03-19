import { Upload, Clock, ArrowRight, Video, Cpu, Box } from 'lucide-react';
import useSessionStore from '../store/sessionStore';

interface WelcomeScreenProps {
  onNewScan: () => void;
  onSelectSession: (session: any) => void;
}

export default function WelcomeScreen({ onNewScan, onSelectSession }: WelcomeScreenProps) {
  const { sessions } = useSessionStore();
  const recentSessions = sessions.slice(0, 5);

  return (
    <div className="flex-1 flex items-center justify-center bg-[#0a0a0b] p-8">
      <div className="max-w-lg w-full animate-fadeIn">
        {/* Logo & Title */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-indigo-500 via-purple-500 to-indigo-600 rounded-2xl shadow-xl shadow-indigo-500/20 mb-5 ring-1 ring-white/10">
            <span className="text-white font-black text-2xl tracking-tighter">F3</span>
          </div>
          <h1 className="text-2xl font-semibold text-zinc-100 mt-2">Face3D</h1>
          <p className="text-sm text-zinc-500 mt-2 leading-relaxed">
            3D Face Reconstruction from Samsung Galaxy S25 Ultra
          </p>
        </div>

        {/* Two action cards */}
        <div className="grid grid-cols-2 gap-4 mb-8">
          {/* New Scan card */}
          <button
            onClick={onNewScan}
            className="group p-5 rounded-xl border border-zinc-800 bg-zinc-900/30 hover:border-indigo-500/40 hover:bg-indigo-500/[0.04] transition-all duration-200 text-left active:scale-[0.98]"
          >
            <div className="w-10 h-10 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center mb-3 group-hover:bg-indigo-500/20 transition-colors duration-200">
              <Upload size={18} className="text-indigo-400" />
            </div>
            <h3 className="text-sm font-semibold text-zinc-200 mb-1">New Scan</h3>
            <p className="text-xs text-zinc-500 leading-relaxed">
              Import video and sensor data to start a reconstruction
            </p>
          </button>

          {/* Recent Sessions card */}
          <div className="p-5 rounded-xl border border-zinc-800 bg-zinc-900/30">
            <div className="w-10 h-10 rounded-lg bg-zinc-800/80 border border-zinc-700/50 flex items-center justify-center mb-3">
              <Clock size={18} className="text-zinc-400" />
            </div>
            <h3 className="text-sm font-semibold text-zinc-200 mb-2">Recent Sessions</h3>
            {recentSessions.length === 0 ? (
              <p className="text-xs text-zinc-600 italic">No sessions yet</p>
            ) : (
              <div className="space-y-1">
                {recentSessions.map((session) => (
                  <button
                    key={session.id}
                    onClick={() => onSelectSession(session)}
                    className="w-full flex items-center justify-between py-1.5 px-2 -mx-2 rounded-md text-xs text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.04] transition-all duration-150 group/item"
                  >
                    <span className="truncate">{session.name}</span>
                    <ArrowRight size={12} className="text-zinc-600 group-hover/item:text-zinc-400 shrink-0 transition-colors duration-150" />
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Feature highlights */}
        <div className="flex items-center justify-center gap-6 text-[11px] text-zinc-600">
          <div className="flex items-center gap-1.5">
            <Video size={13} className="text-zinc-500" />
            <span>Video + Photos</span>
          </div>
          <div className="text-zinc-700">+</div>
          <div className="flex items-center gap-1.5">
            <Cpu size={13} className="text-zinc-500" />
            <span>IMU Sensors</span>
          </div>
          <div className="text-zinc-700">=</div>
          <div className="flex items-center gap-1.5">
            <Box size={13} className="text-indigo-400" />
            <span className="text-indigo-400">3D Model</span>
          </div>
        </div>
      </div>
    </div>
  );
}
