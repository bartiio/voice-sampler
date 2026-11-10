import React, { useState, useEffect } from 'react';
import { Sidebar, NavTab } from './components/Sidebar';
import { SamplesView } from './views/SamplesView';
import { TrainingView } from './views/TrainingView';
import { GenerateView } from './views/GenerateView';
import { SettingsView } from './views/SettingsView';
import { api } from './api/client';
import { trainingWebSocket } from './api/websocket';
import { TrainingTelemetryFrame } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavTab>('samples');
  const [isBackendOnline, setIsBackendOnline] = useState<boolean>(true);
  const [isTrainingActive, setIsTrainingActive] = useState<boolean>(false);
  const [trainingProgress, setTrainingProgress] = useState<number>(0);
  const [toastNotification, setToastNotification] = useState<string | null>(null);

  // Health Polling loop
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await api.getHealth();
        setIsBackendOnline(health.backend_ready);
      } catch {
        setIsBackendOnline(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  // Global WebSocket Telemetry Listener for background training tracking
  useEffect(() => {
    trainingWebSocket.connect();
    const unsubscribe = trainingWebSocket.subscribe((frame: TrainingTelemetryFrame) => {
      if (frame.type === 'progress') {
        setIsTrainingActive(true);
        setTrainingProgress(frame.progress_percent);
      } else if (frame.type === 'completed') {
        setIsTrainingActive(false);
        setTrainingProgress(100);
        showToast(`Model '${frame.model_name}' finished training successfully!`);
      } else if (frame.type === 'stopped' || frame.type === 'error') {
        setIsTrainingActive(false);
      }
    });

    return () => {
      unsubscribe();
    };
  }, []);

  const showToast = (msg: string) => {
    setToastNotification(msg);
    setTimeout(() => setToastNotification(null), 6000);
  };

  return (
    <div className="flex h-screen w-screen bg-slate-950 font-sans text-slate-100 overflow-hidden">
      {/* Accessible Sidebar */}
      <Sidebar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        isBackendOnline={isBackendOnline}
        isTrainingActive={isTrainingActive}
        trainingProgress={trainingProgress}
      />

      {/* Main Content Area */}
      <main className="flex-1 overflow-y-auto bg-slate-950">
        {toastNotification && (
          <div className="fixed top-5 right-5 z-50 bg-emerald-950 border border-emerald-600 text-emerald-200 px-5 py-3 rounded-xl shadow-2xl flex items-center space-x-3 text-sm animate-bounce">
            <span>🎉</span>
            <span>{toastNotification}</span>
            <button onClick={() => setToastNotification(null)} className="font-bold ml-2">✕</button>
          </div>
        )}

        {activeTab === 'samples' && <SamplesView />}
        {activeTab === 'training' && <TrainingView />}
        {activeTab === 'generate' && <GenerateView />}
        {activeTab === 'settings' && <SettingsView />}
      </main>
    </div>
  );
};

export default App;
