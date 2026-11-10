import React from 'react';

export type NavTab = 'samples' | 'training' | 'generate' | 'settings';

interface SidebarProps {
  activeTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  isBackendOnline: boolean;
  isTrainingActive: boolean;
  trainingProgress?: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onSelectTab,
  isBackendOnline,
  isTrainingActive,
  trainingProgress = 0,
}) => {
  const navItems: Array<{ id: NavTab; label: string; icon: string }> = [
    { id: 'samples', label: 'Samples', icon: '🎙️' },
    { id: 'training', label: 'Training', icon: '⚡' },
    { id: 'generate', label: 'Generate', icon: '🪄' },
    { id: 'settings', label: 'Settings', icon: '⚙️' },
  ];

  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col justify-between h-screen select-none flex-shrink-0">
      {/* Branding Header */}
      <div>
        <div className="p-6 border-b border-slate-800">
          <div className="flex items-center space-x-3">
            <span className="text-2xl">🌊</span>
            <div>
              <h1 className="font-bold text-lg text-slate-100 tracking-tight leading-none">
                Voice Sampler
              </h1>
              <p className="text-xs text-slate-500 mt-1">Polish Voice Cloning</p>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="p-4 space-y-1.5" aria-label="Main Navigation">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectTab(item.id)}
                aria-current={isActive ? 'page' : undefined}
                className={`w-full flex items-center justify-between px-4 py-3 rounded-xl text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-indigo-600/15 text-indigo-400 border border-indigo-500/30 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`}
              >
                <div className="flex items-center space-x-3">
                  <span className="text-lg">{item.icon}</span>
                  <span>{item.label}</span>
                </div>

                {/* Training Badge */}
                {item.id === 'training' && isTrainingActive && (
                  <span className="text-[10px] bg-indigo-500/30 text-indigo-300 font-semibold px-2 py-0.5 rounded-full animate-pulse">
                    {trainingProgress > 0 ? `${trainingProgress.toFixed(0)}%` : 'Running'}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Connectivity Footer */}
      <div className="p-4 border-t border-slate-800 bg-slate-950/40">
        <div className="flex items-center justify-between text-xs px-2 py-1">
          <div className="flex items-center space-x-2.5">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                isBackendOnline ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'
              }`}
            />
            <span className="font-medium text-slate-300">
              {isBackendOnline ? 'Backend Online' : 'Backend Offline'}
            </span>
          </div>
          <span className="text-[10px] text-slate-500 font-mono">v0.1.0</span>
        </div>
      </div>
    </aside>
  );
};
