import React, { useState, useEffect } from 'react';
import { api, API_BASE_URL } from '../api/client';
import { GPUInfoResponse, HealthResponse } from '../types';

export const SettingsView: React.FC = () => {
  const [gpuInfo, setGpuInfo] = useState<GPUInfoResponse | null>(null);
  const [healthInfo, setHealthInfo] = useState<HealthResponse | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState<string>('');

  useEffect(() => {
    fetchDiagnostics();
  }, []);

  const fetchDiagnostics = async (forceRefresh = false) => {
    setIsLoading(true);
    const start = performance.now();
    try {
      const [gpu, health] = await Promise.all([
        api.getGpuSettings(forceRefresh),
        api.getHealth(),
      ]);
      const elapsed = Math.round(performance.now() - start);
      setLatencyMs(elapsed);
      setGpuInfo(gpu);
      setHealthInfo(health);
      setLastRefreshed(new Date().toLocaleTimeString());
    } catch {
      setLatencyMs(null);
    } finally {
      setIsLoading(false);
    }
  };

  const formatVram = (mb: number | null | undefined) => {
    if (mb === null || mb === undefined) return 'N/A';
    if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB (${mb} MB)`;
    return `${mb} MB`;
  };

  const formatUptime = (seconds: number | undefined) => {
    if (seconds === undefined) return '--';
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hrs}h ${mins}m ${secs}s`;
  };

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8 text-slate-100">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">System & Settings</h1>
          <p className="text-slate-400 mt-1">
            Hardware acceleration diagnostics, CUDA availability, and backend health status.
          </p>
        </div>
        <button
          onClick={() => fetchDiagnostics(true)}
          disabled={isLoading}
          className="bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-sm font-medium py-2 px-4 rounded-lg transition-colors shadow flex items-center space-x-2"
        >
          <span>{isLoading ? 'Scanning...' : 'Refresh Diagnostics'}</span>
        </button>
      </div>

      {/* Prominent NO-CUDA WARNING BANNER */}
      {gpuInfo && !gpuInfo.cuda_available && (
        <div className="bg-amber-950/60 border border-amber-600/80 rounded-xl p-5 text-amber-200 shadow-md space-y-2">
          <div className="flex items-center space-x-3">
            <span className="text-2xl">⚠️</span>
            <h2 className="text-lg font-bold text-amber-300">
              No CUDA GPU detected. GPU is required for real training. Running in simulation mode.
            </h2>
          </div>
          <p className="text-sm text-amber-200/90 pl-9">
            {gpuInfo.message ||
              'A compatible NVIDIA GPU with CUDA drivers was not found on this machine. The application is operating in high-fidelity simulation mode. You can still test dataset preparation, mock fine-tuning, loss telemetry, Faiss indexing, and voice conversion workflows.'}
          </p>
        </div>
      )}

      {/* Grid: GPU Diagnostics & Backend Health */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* GPU Diagnostics Card */}
        <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-6 shadow-sm space-y-5">
          <div className="flex justify-between items-center pb-2 border-b border-slate-700">
            <h3 className="text-xl font-semibold">GPU & Hardware Acceleration</h3>
            <span
              className={`text-xs px-2.5 py-1 rounded-full font-semibold ${
                gpuInfo?.cuda_available
                  ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                  : 'bg-amber-950 text-amber-300 border border-amber-800'
              }`}
            >
              {gpuInfo?.cuda_available ? 'CUDA ACTIVE' : 'SIMULATION MODE'}
            </span>
          </div>

          <div className="space-y-3 text-sm">
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">Detected GPU Device:</span>
              <span className="font-medium text-slate-200">
                {gpuInfo?.gpu_name || gpuInfo?.hardware_gpu_name || 'Generic / None'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">CUDA Availability:</span>
              <span className="font-medium text-slate-200">
                {gpuInfo?.cuda_available ? 'Supported (Yes)' : 'Not Available (No)'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">CUDA Runtime Version:</span>
              <span className="font-mono text-slate-200">
                {gpuInfo?.cuda_version || 'None'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">Total Video RAM (VRAM):</span>
              <span className="font-medium text-slate-200">
                {formatVram(gpuInfo?.vram_total_mb)}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">Allocated VRAM:</span>
              <span className="font-medium text-slate-200">
                {formatVram(gpuInfo?.vram_allocated_mb)}
              </span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">PyTorch Version:</span>
              <span className="font-mono text-slate-200">
                {gpuInfo?.torch_version || 'N/A'}
              </span>
            </div>
          </div>
        </div>

        {/* Backend Health Card */}
        <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-6 shadow-sm space-y-5">
          <div className="flex justify-between items-center pb-2 border-b border-slate-700">
            <h3 className="text-xl font-semibold">FastAPI Backend Status</h3>
            <span
              className={`text-xs px-2.5 py-1 rounded-full font-semibold ${
                healthInfo?.backend_ready
                  ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                  : 'bg-red-950 text-red-300 border border-red-800'
              }`}
            >
              {healthInfo?.backend_ready ? 'ONLINE' : 'DISCONNECTED'}
            </span>
          </div>

          <div className="space-y-3 text-sm">
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">Endpoint URL:</span>
              <span className="font-mono text-slate-200">{API_BASE_URL}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">API Software Version:</span>
              <span className="font-mono text-slate-200">
                v{healthInfo?.version || '0.1.0'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">Roundtrip Latency:</span>
              <span className="font-mono text-emerald-400">
                {latencyMs !== null ? `${latencyMs} ms` : 'Unreachable'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">Server Uptime:</span>
              <span className="font-medium text-slate-200">
                {formatUptime(healthInfo?.uptime_seconds)}
              </span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Last Diagnostics Check:</span>
              <span className="text-slate-400">{lastRefreshed || 'Just now'}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
