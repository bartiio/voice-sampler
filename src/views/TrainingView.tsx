import React, { useState, useEffect, useRef } from 'react';
import { api } from '../api/client';
import { trainingWebSocket } from '../api/websocket';
import {
  Sample,
  RegisteredModel,
  TrainingTelemetryFrame,
  TrainingStatusResponse,
} from '../types';
import { LossChart } from '../components/LossChart';

interface LogEntry {
  id: string;
  timestamp: string;
  text: string;
  type: 'info' | 'progress' | 'complete' | 'stopped' | 'error';
}

export const TrainingView: React.FC = () => {
  // Engine Selection
  const [engine, setEngine] = useState<'f5-tts' | 'rvc'>('f5-tts');

  // Hyperparameters
  const [modelName, setModelName] = useState('');
  const [epochs, setEpochs] = useState<number>(10);
  const [batchSize, setBatchSize] = useState<number>(2);
  const [learningRate, setLearningRate] = useState<number>(0.0001);
  const [sampleRate, setSampleRate] = useState<32000 | 40000 | 48000>(40000);

  // Sample Selection
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selectedSampleIds, setSelectedSampleIds] = useState<Set<string>>(new Set());
  const [isLoadingSamples, setIsLoadingSamples] = useState(false);

  // Training Live State
  const [isTraining, setIsTraining] = useState(false);
  const [_activeRunId, setActiveRunId] = useState<string | null>(null);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [currentLoss, setCurrentLoss] = useState<number | null>(null);
  const [progressPercent, setProgressPercent] = useState(0);
  const [statusMessage, setStatusMessage] = useState('Idle');
  const [lossHistory, setLossHistory] = useState<Array<{ step: number; loss: number }>>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);

  // Model Registry
  const [models, setModels] = useState<RegisteredModel[]>([]);
  const [modelFilter, setModelFilter] = useState<'all' | 'f5-tts' | 'rvc'>('all');
  const [modelToDelete, setModelToDelete] = useState<string | null>(null);

  // Form Validation
  const [validationError, setValidationError] = useState<string | null>(null);
  const consoleEndRef = useRef<HTMLDivElement>(null);

  // Auto-switch defaults on engine tab change
  const handleEngineChange = (newEngine: 'f5-tts' | 'rvc') => {
    setEngine(newEngine);
    if (newEngine === 'f5-tts') {
      setEpochs(10);
      setBatchSize(2);
      setLearningRate(0.0001);
    } else {
      setEpochs(20);
      setBatchSize(4);
      setSampleRate(40000);
    }
  };

  const appendLog = (text: string, type: LogEntry['type']) => {
    const time = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev.slice(-250), { id: Math.random().toString(), timestamp: time, text, type }]);
  };

  const handleTelemetryFrame = (frame: TrainingTelemetryFrame) => {
    if (frame.type === 'progress') {
      setIsTraining(true);
      setActiveRunId(frame.run_id);
      setCurrentEpoch(frame.epoch);
      setTotalEpochs(frame.total_epochs);
      setCurrentStep(frame.step);
      setTotalSteps(frame.total_steps);
      setCurrentLoss(frame.loss);
      setProgressPercent(frame.progress_percent);
      setStatusMessage(frame.message);

      setLossHistory((prev) => [...prev, { step: frame.step, loss: frame.loss }]);
      appendLog(frame.message, 'progress');
    } else if (frame.type === 'completed') {
      setIsTraining(false);
      setProgressPercent(100);
      setStatusMessage('Completed');
      appendLog(`Training completed successfully for model ${frame.model_name}! Final loss: ${frame.final_loss}`, 'complete');
      loadModels();
    } else if (frame.type === 'stopped') {
      setIsTraining(false);
      setStatusMessage('Stopped');
      appendLog(`Training stopped: ${frame.message}`, 'stopped');
    } else if (frame.type === 'error') {
      setIsTraining(false);
      setStatusMessage('Failed');
      appendLog(`Training error: ${frame.error_message}`, 'error');
    }
  };

  const loadSamples = async () => {
    setIsLoadingSamples(true);
    try {
      const res = await api.getSamples();
      setSamples(res.samples);
      setSelectedSampleIds(new Set(res.samples.map((s) => s.id)));
    } catch (err: any) {
      appendLog(`Failed to load samples: ${err.message}`, 'error');
    } finally {
      setIsLoadingSamples(false);
    }
  };

  const loadModels = async () => {
    try {
      const res = await api.getModels();
      setModels(res.models);
    } catch (err: any) {
      appendLog(`Failed to load models: ${err.message}`, 'error');
    }
  };

  const checkTrainingStatus = async () => {
    try {
      const statusRes: TrainingStatusResponse = await api.getTrainingStatus();
      if (statusRes.active && statusRes.run) {
        setIsTraining(true);
        setActiveRunId(statusRes.run.id);
        setCurrentEpoch(statusRes.run.current_epoch);
        setTotalEpochs(statusRes.run.total_epochs);
        setCurrentStep(statusRes.run.current_step);
        setTotalSteps(statusRes.run.total_steps);
        setCurrentLoss(statusRes.run.current_loss);
        setProgressPercent(statusRes.run.progress_percent);
        setStatusMessage(statusRes.run.status_message || 'Training active');
      }
    } catch {
      // Backend may be initializing
    }
  };

  // Initial Load: Samples, Models, and Active Status
  useEffect(() => {
    loadSamples();
    loadModels();
    checkTrainingStatus();

    // Connect WebSocket
    trainingWebSocket.connect();
    const unsubscribe = trainingWebSocket.subscribe(handleTelemetryFrame);

    return () => {
      unsubscribe();
    };
  }, []);

  // Auto-scroll logs
  useEffect(() => {
    consoleEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Validation
  const validate = (): boolean => {
    if (!modelName.trim()) {
      setValidationError('Model name cannot be empty');
      return false;
    }
    if (!/^[a-zA-Z0-9_\-]+$/.test(modelName.trim())) {
      setValidationError('Model name must only contain alphanumeric characters, underscores, or dashes');
      return false;
    }
    if (selectedSampleIds.size === 0) {
      setValidationError('Please select at least one voice sample for training');
      return false;
    }
    if (epochs < 1) {
      setValidationError('Epochs must be >= 1');
      return false;
    }
    if (batchSize < 1) {
      setValidationError('Batch size must be >= 1');
      return false;
    }
    setValidationError(null);
    return true;
  };

  const handleStartTraining = async () => {
    if (!validate()) return;

    setLossHistory([]);
    setLogs([]);
    appendLog(`Initializing ${engine === 'f5-tts' ? 'F5-TTS' : 'RVC v2'} training run for model '${modelName}'...`, 'info');

    try {
      const sampleIds = Array.from(selectedSampleIds);
      if (engine === 'f5-tts') {
        await api.trainF5TTS({
          model_name: modelName.trim(),
          epochs,
          batch_size: batchSize,
          learning_rate: learningRate,
          sample_ids: sampleIds,
        });
      } else {
        await api.trainRVC({
          model_name: modelName.trim(),
          epochs,
          batch_size: batchSize,
          sample_rate: sampleRate,
          sample_ids: sampleIds,
        });
      }
      setIsTraining(true);
    } catch (err: any) {
      appendLog(`Error starting training: ${err.message}`, 'error');
    }
  };

  const handleStopTraining = async () => {
    try {
      appendLog('Requesting training cancellation...', 'info');
      await api.stopTraining();
    } catch (err: any) {
      appendLog(`Failed to stop training: ${err.message}`, 'error');
    }
  };

  const handleDeleteModel = async (name: string) => {
    try {
      await api.deleteModel(name);
      setModelToDelete(null);
      loadModels();
      appendLog(`Deleted model '${name}'`, 'info');
    } catch (err: any) {
      appendLog(`Failed to delete model: ${err.message}`, 'error');
    }
  };

  const toggleSample = (id: string) => {
    setSelectedSampleIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const selectAllSamples = () => {
    setSelectedSampleIds(new Set(samples.map((s) => s.id)));
  };

  const deselectAllSamples = () => {
    setSelectedSampleIds(new Set());
  };

  const filteredModels = models.filter((m) => {
    if (modelFilter === 'all') return true;
    return m.engine === modelFilter;
  });

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8 text-slate-100">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Model Training</h1>
        <p className="text-slate-400 mt-1">
          Train custom Polish voice models using F5-TTS (Diffusion-Transformer) or RVC v2 (Voice Conversion).
        </p>
      </div>

      {/* Engine Selector Tabs */}
      <div className="flex border-b border-slate-700 space-x-8">
        <button
          onClick={() => handleEngineChange('f5-tts')}
          disabled={isTraining}
          className={`pb-3 font-semibold text-sm transition-colors border-b-2 ${
            engine === 'f5-tts'
              ? 'border-indigo-500 text-indigo-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          F5-TTS Engine (Text-to-Speech)
        </button>
        <button
          onClick={() => handleEngineChange('rvc')}
          disabled={isTraining}
          className={`pb-3 font-semibold text-sm transition-colors border-b-2 ${
            engine === 'rvc'
              ? 'border-indigo-500 text-indigo-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          RVC v2 Engine (Speech-to-Speech)
        </button>
      </div>

      {/* Grid: Config Form & Sample Selector */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Hyperparameters Card */}
        <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-6 shadow-sm space-y-5">
          <h2 className="text-xl font-semibold">Hyperparameters</h2>

          {validationError && (
            <div className="p-3 bg-red-950/60 border border-red-800/80 rounded-lg text-red-200 text-sm">
              {validationError}
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">
              Model Name *
            </label>
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              disabled={isTraining}
              placeholder="e.g. polish_voice_v1"
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">
                Epochs
              </label>
              <input
                type="number"
                min={1}
                max={engine === 'f5-tts' ? 100 : 200}
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value) || 1)}
                disabled={isTraining}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">
                Batch Size
              </label>
              <input
                type="number"
                min={1}
                max={32}
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)}
                disabled={isTraining}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>
          </div>

          {engine === 'f5-tts' ? (
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">
                Learning Rate
              </label>
              <input
                type="number"
                step="0.00001"
                min="0.00001"
                max="0.001"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.0001)}
                disabled={isTraining}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>
          ) : (
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">
                Sample Rate
              </label>
              <select
                value={sampleRate}
                onChange={(e) => setSampleRate(parseInt(e.target.value) as any)}
                disabled={isTraining}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value={32000}>32,000 Hz</option>
                <option value={40000}>40,000 Hz (Recommended)</option>
                <option value={48000}>48,000 Hz</option>
              </select>
            </div>
          )}

          {/* Action Buttons */}
          <div className="pt-2 flex space-x-4">
            {!isTraining ? (
              <button
                onClick={handleStartTraining}
                disabled={selectedSampleIds.size === 0}
                className="flex-1 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium py-2.5 px-4 rounded-lg transition-colors shadow"
              >
                Start Training
              </button>
            ) : (
              <button
                onClick={handleStopTraining}
                className="flex-1 bg-red-600 hover:bg-red-500 text-white font-medium py-2.5 px-4 rounded-lg transition-colors shadow"
              >
                Stop Training
              </button>
            )}
          </div>
        </div>

        {/* Sample Selection Card */}
        <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-6 shadow-sm flex flex-col">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h2 className="text-xl font-semibold">Training Samples</h2>
              <p className="text-xs text-slate-400 mt-0.5">
                {selectedSampleIds.size} of {samples.length} selected
              </p>
            </div>
            <div className="space-x-3 text-xs">
              <button
                onClick={selectAllSamples}
                disabled={isTraining || samples.length === 0}
                className="text-indigo-400 hover:text-indigo-300 font-medium"
              >
                Select All
              </button>
              <span className="text-slate-600">|</span>
              <button
                onClick={deselectAllSamples}
                disabled={isTraining || samples.length === 0}
                className="text-indigo-400 hover:text-indigo-300 font-medium"
              >
                Deselect All
              </button>
            </div>
          </div>

          {isLoadingSamples ? (
            <div className="flex-1 flex items-center justify-center text-slate-400">
              Loading samples...
            </div>
          ) : samples.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center p-8 text-center border border-dashed border-slate-700 rounded-lg">
              <p className="text-slate-400 text-sm mb-3">No voice samples available.</p>
              <span className="text-slate-500 text-xs">
                Switch to the Samples tab to record or upload audio clips.
              </span>
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto max-h-[300px] divide-y divide-slate-700/60 border border-slate-700/60 rounded-lg bg-slate-900/40">
              {samples.map((s) => (
                <label
                  key={s.id}
                  className="flex items-center space-x-3 p-3 hover:bg-slate-800/50 cursor-pointer text-sm"
                >
                  <input
                    type="checkbox"
                    checked={selectedSampleIds.has(s.id)}
                    onChange={() => toggleSample(s.id)}
                    disabled={isTraining}
                    className="rounded border-slate-700 text-indigo-600 focus:ring-indigo-500 bg-slate-900"
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex justify-between">
                      <span className="font-medium text-slate-200 truncate">{s.filename}</span>
                      <span className="text-xs text-slate-400 ml-2">{(s.duration || s.duration_seconds || 0).toFixed(1)}s</span>
                    </div>
                    {s.transcript && (
                      <p className="text-xs text-slate-400 truncate mt-0.5 italic">
                        "{s.transcript}"
                      </p>
                    )}
                  </div>
                </label>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Real-time Dashboard */}
      <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-6 shadow-sm space-y-6">
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-semibold">Training Progress</h2>
          <div className="flex items-center space-x-3">
            <span
              className={`inline-block w-2.5 h-2.5 rounded-full ${
                isTraining ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500'
              }`}
            />
            <span className="text-sm font-medium text-slate-300">{statusMessage}</span>
          </div>
        </div>

        {/* Progress Bar & Counters */}
        <div className="space-y-2">
          <div className="flex justify-between text-xs text-slate-400 font-medium">
            <span>
              Epoch {currentEpoch} / {totalEpochs || 0} (Step {currentStep} / {totalSteps || 0})
            </span>
            <span>{progressPercent.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-slate-900 rounded-full h-3 overflow-hidden border border-slate-700">
            <div
              className="bg-indigo-500 h-full transition-all duration-300 ease-out"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>

        {/* Live Loss Chart & Log Console */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Dynamic SVG Loss Chart */}
          <div className="bg-slate-900/80 p-4 rounded-lg border border-slate-700 flex flex-col">
            <div className="flex justify-between items-center mb-3">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Live Loss Curve
              </span>
              <span className="text-xs font-mono text-indigo-400">
                Current: {currentLoss !== null ? currentLoss.toFixed(4) : '--'}
              </span>
            </div>
            <div className="flex-1 min-h-[180px]">
              <LossChart data={lossHistory} />
            </div>
          </div>

          {/* Real-Time Status Log Console */}
          <div className="bg-slate-900/80 p-4 rounded-lg border border-slate-700 flex flex-col font-mono text-xs">
            <div className="flex justify-between items-center mb-2 pb-2 border-b border-slate-800">
              <span className="font-semibold uppercase tracking-wider text-slate-400">
                Training Console
              </span>
              <button
                onClick={() => setLogs([])}
                className="text-slate-500 hover:text-slate-400 text-xs"
              >
                Clear
              </button>
            </div>
            <div className="flex-1 overflow-y-auto max-h-[180px] space-y-1">
              {logs.length === 0 ? (
                <div className="text-slate-600 italic">Awaiting training telemetry...</div>
              ) : (
                logs.map((log) => (
                  <div key={log.id} className="leading-tight">
                    <span className="text-slate-500 mr-2">[{log.timestamp}]</span>
                    <span
                      className={
                        log.type === 'error'
                          ? 'text-red-400'
                          : log.type === 'complete'
                          ? 'text-emerald-400'
                          : log.type === 'stopped'
                          ? 'text-amber-400'
                          : 'text-slate-300'
                      }
                    >
                      {log.text}
                    </span>
                  </div>
                ))
              )}
              <div ref={consoleEndRef} />
            </div>
          </div>
        </div>
      </div>

      {/* Model Registry Table */}
      <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-6 shadow-sm space-y-4">
        <div className="flex justify-between items-center">
          <div>
            <h2 className="text-xl font-semibold">Model Registry</h2>
            <p className="text-xs text-slate-400">Saved model checkpoints ready for voice generation</p>
          </div>
          <div className="flex space-x-2">
            {(['all', 'f5-tts', 'rvc'] as const).map((f) => (
              <button
                key={f}
                onClick={() => setModelFilter(f)}
                className={`text-xs px-3 py-1 rounded-md transition-colors ${
                  modelFilter === f
                    ? 'bg-indigo-600 text-white'
                    : 'bg-slate-900 text-slate-400 hover:text-slate-200'
                }`}
              >
                {f.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-300">
            <thead className="bg-slate-900/80 text-xs uppercase text-slate-400 border-b border-slate-700">
              <tr>
                <th className="p-3">Model Name</th>
                <th className="p-3">Engine</th>
                <th className="p-3">Samples</th>
                <th className="p-3">Created</th>
                <th className="p-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/60">
              {filteredModels.length === 0 ? (
                <tr>
                  <td colSpan={5} className="p-6 text-center text-slate-500 italic">
                    No trained models found in registry.
                  </td>
                </tr>
              ) : (
                filteredModels.map((m) => (
                  <tr key={m.id} className="hover:bg-slate-800/40">
                    <td className="p-3 font-medium text-slate-200">{m.name}</td>
                    <td className="p-3">
                      <span
                        className={`text-xs px-2 py-0.5 rounded-full font-semibold ${
                          m.engine === 'f5-tts'
                            ? 'bg-blue-950 text-blue-300 border border-blue-800'
                            : 'bg-purple-950 text-purple-300 border border-purple-800'
                        }`}
                      >
                        {m.engine.toUpperCase()}
                      </span>
                    </td>
                    <td className="p-3 text-slate-400">{m.sample_count}</td>
                    <td className="p-3 text-slate-400">
                      {new Date(m.created_at).toLocaleString()}
                    </td>
                    <td className="p-3 text-right">
                      <button
                        onClick={() => setModelToDelete(m.name)}
                        className="text-xs text-red-400 hover:text-red-300 font-medium px-2 py-1 rounded hover:bg-red-950/40"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {modelToDelete && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center p-4 z-50">
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 max-w-md w-full space-y-4 shadow-xl">
            <h3 className="text-lg font-bold text-slate-100">Delete Model</h3>
            <p className="text-sm text-slate-300">
              Are you sure you want to permanently delete model <strong>'{modelToDelete}'</strong> and all associated checkpoints?
            </p>
            <div className="flex justify-end space-x-3 pt-2">
              <button
                onClick={() => setModelToDelete(null)}
                className="px-4 py-2 text-sm text-slate-400 hover:text-slate-200"
              >
                Cancel
              </button>
              <button
                onClick={() => handleDeleteModel(modelToDelete)}
                className="px-4 py-2 text-sm bg-red-600 hover:bg-red-500 text-white rounded-lg font-medium shadow"
              >
                Delete Model
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
