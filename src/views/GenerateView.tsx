import React, { useState, useEffect, useRef } from 'react';
import { api } from '../api/client';
import {
  RegisteredModel,
  Sample,
  GenerateResponse,
  GenerationMetadata,
} from '../types';

const POLISH_DIACRITICS_LOWER = ['ą', 'ć', 'ę', 'ł', 'ń', 'ó', 'ś', 'ź', 'ż'];
const POLISH_DIACRITICS_UPPER = ['Ą', 'Ć', 'Ę', 'Ł', 'Ń', 'Ó', 'Ś', 'Ź', 'Ż'];

const PANGRAMS = [
  { label: 'Zażółć gęślą jaźń', text: 'Zażółć gęślą jaźń.' },
  { label: 'Mężny bądź...', text: 'Mężny bądź, chroń pułk twój i sześć flag.' },
  { label: 'Pchnąć w tę łódź...', text: 'Pchnąć w tę łódź jeża lub ośm skrzyń fig.' },
];

export const GenerateView: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'tts' | 'sts'>('tts');

  // Available Resources
  const [f5Models, setF5Models] = useState<RegisteredModel[]>([]);
  const [rvcModels, setRvcModels] = useState<RegisteredModel[]>([]);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [history, setHistory] = useState<GenerationMetadata[]>([]);

  // TTS State
  const [ttsText, setTtsText] = useState('Cześć, to jest mój sklonowany głos w języku polskim.');
  const [ttsModel, setTtsModel] = useState('');
  const [ttsRefSample, setTtsRefSample] = useState('');
  const [ttsSpeed, setTtsSpeed] = useState(1.0);
  const [diacriticShift, setDiacriticShift] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // STS State
  const [stsInputSample, setStsInputSample] = useState('');
  const [stsModel, setStsModel] = useState('');
  const [pitchShift, setPitchShift] = useState(0);
  const [indexRate, setIndexRate] = useState(0.75);

  // Generation & Player State
  const [isGenerating, setIsGenerating] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [currentAudio, setCurrentAudio] = useState<{
    id: string;
    url: string;
    filename: string;
    duration?: number;
    model: string;
    type: 'tts' | 'sts';
  } | null>(null);

  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [modelsRes, samplesRes, historyRes] = await Promise.all([
        api.getModels(),
        api.getSamples(),
        api.getGenerationHistory(undefined, 25),
      ]);

      const f5 = modelsRes.models.filter((m) => m.engine === 'f5-tts');
      const rvc = modelsRes.models.filter((m) => m.engine === 'rvc');
      setF5Models(f5);
      setRvcModels(rvc);
      setSamples(samplesRes.samples);
      setHistory(historyRes.generations);

      if (f5.length > 0) setTtsModel(f5[0].name);
      if (rvc.length > 0) setStsModel(rvc[0].name);
      if (samplesRes.samples.length > 0) {
        setTtsRefSample(samplesRes.samples[0].id);
        setStsInputSample(samplesRes.samples[0].id);
      }
    } catch (err: any) {
      setErrorMessage(`Failed to load resources: ${err.message}`);
    }
  };

  const handleInsertDiacritic = (char: string) => {
    if (!textareaRef.current) return;
    const textarea = textareaRef.current;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const text = ttsText;
    const newText = text.substring(0, start) + char + text.substring(end);
    setTtsText(newText);

    setTimeout(() => {
      textarea.focus();
      textarea.setSelectionRange(start + 1, start + 1);
    }, 0);
  };

  const handleGenerateTTS = async () => {
    if (!ttsText.trim()) {
      setErrorMessage('Please enter text to synthesize');
      return;
    }
    if (!ttsModel) {
      setErrorMessage('Please select a trained F5-TTS model');
      return;
    }

    setIsGenerating(true);
    setErrorMessage(null);

    try {
      const res: GenerateResponse = await api.generateTTS({
        text: ttsText.trim(),
        model_name: ttsModel,
        reference_sample_id: ttsRefSample || undefined,
        speed: ttsSpeed,
      });

      setCurrentAudio({
        id: res.generation_id,
        url: api.getGeneratedAudioUrl(res.filename),
        filename: res.filename,
        duration: res.duration,
        model: ttsModel,
        type: 'tts',
      });

      const hist = await api.getGenerationHistory(undefined, 25);
      setHistory(hist.generations);
    } catch (err: any) {
      setErrorMessage(err.message || 'Generation failed');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleGenerateSTS = async () => {
    if (!stsInputSample) {
      setErrorMessage('Please select a source audio sample');
      return;
    }
    if (!stsModel) {
      setErrorMessage('Please select a trained RVC model');
      return;
    }

    setIsGenerating(true);
    setErrorMessage(null);

    try {
      const res: GenerateResponse = await api.generateSTS({
        input_sample_id: stsInputSample,
        model_name: stsModel,
        pitch_shift: pitchShift,
        index_rate: indexRate,
      });

      setCurrentAudio({
        id: res.generation_id,
        url: api.getGeneratedAudioUrl(res.filename),
        filename: res.filename,
        duration: res.duration,
        model: stsModel,
        type: 'sts',
      });

      const hist = await api.getGenerationHistory(undefined, 25);
      setHistory(hist.generations);
    } catch (err: any) {
      setErrorMessage(err.message || 'Voice conversion failed');
    } finally {
      setIsGenerating(false);
    }
  };

  const handlePlayFromHistory = (gen: GenerationMetadata) => {
    setCurrentAudio({
      id: gen.id,
      url: api.getGeneratedAudioUrl(gen.output_filename),
      filename: gen.output_filename,
      duration: gen.duration_seconds,
      model: gen.model_id,
      type: gen.type,
    });
    if (audioRef.current) {
      audioRef.current.load();
      audioRef.current.play().catch(() => {});
    }
  };

  const handleDeleteHistory = async (id: string) => {
    try {
      await api.deleteGeneration(id);
      setHistory((prev) => prev.filter((h) => h.id !== id));
      if (currentAudio?.id === id) {
        setCurrentAudio(null);
      }
    } catch (err: any) {
      setErrorMessage(`Failed to delete record: ${err.message}`);
    }
  };

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8 text-slate-100">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Voice Generation</h1>
        <p className="text-slate-400 mt-1">
          Synthesize new speech (F5-TTS) or convert existing audio voices (RVC v2).
        </p>
      </div>

      {errorMessage && (
        <div className="p-4 bg-red-950/70 border border-red-800 rounded-xl text-red-200 text-sm flex justify-between items-center">
          <span>{errorMessage}</span>
          <button onClick={() => setErrorMessage(null)} className="text-red-400 hover:text-red-300 font-bold ml-4">
            ✕
          </button>
        </div>
      )}

      {/* Tabs */}
      <div className="flex border-b border-slate-700 space-x-8">
        <button
          onClick={() => setActiveTab('tts')}
          className={`pb-3 font-semibold text-sm transition-colors border-b-2 ${
            activeTab === 'tts'
              ? 'border-indigo-500 text-indigo-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          Text-to-Speech (F5-TTS)
        </button>
        <button
          onClick={() => setActiveTab('sts')}
          className={`pb-3 font-semibold text-sm transition-colors border-b-2 ${
            activeTab === 'sts'
              ? 'border-indigo-500 text-indigo-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          Speech-to-Speech (RVC v2)
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Controls Column (2 cols) */}
        <div className="lg:col-span-2 space-y-6">
          {activeTab === 'tts' ? (
            /* TTS Form */
            <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-6 shadow-sm space-y-5">
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-sm font-medium text-slate-300">Polish Text Prompt</label>
                  <span className="text-xs text-slate-500">{ttsText.length} characters</span>
                </div>
                <textarea
                  ref={textareaRef}
                  value={ttsText}
                  onChange={(e) => setTtsText(e.target.value)}
                  placeholder="Wpisz tekst do syntezy..."
                  rows={4}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-y"
                  style={{ fieldSizing: 'content', minHeight: '90px' } as any}
                />
              </div>

              {/* Polish Diacritic Quick Buttons */}
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Polish Diacritics
                  </span>
                  <button
                    onClick={() => setDiacriticShift(!diacriticShift)}
                    className="text-xs text-indigo-400 hover:text-indigo-300 font-medium"
                  >
                    {diacriticShift ? 'Shift [UPPER]' : 'Shift [lower]'}
                  </button>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {(diacriticShift ? POLISH_DIACRITICS_UPPER : POLISH_DIACRITICS_LOWER).map((c) => (
                    <button
                      key={c}
                      type="button"
                      onClick={() => handleInsertDiacritic(c)}
                      className="w-8 h-8 rounded bg-slate-900 border border-slate-700 hover:bg-slate-700 text-slate-200 font-semibold text-sm transition-colors flex items-center justify-center active:scale-95"
                    >
                      {c}
                    </button>
                  ))}
                </div>
              </div>

              {/* Pangram Presets */}
              <div className="space-y-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Quick Pangrams
                </span>
                <div className="flex flex-wrap gap-2">
                  {PANGRAMS.map((p) => (
                    <button
                      key={p.label}
                      type="button"
                      onClick={() => setTtsText(p.text)}
                      className="text-xs bg-slate-900/80 border border-slate-700 hover:border-indigo-500 text-slate-300 hover:text-white px-3 py-1.5 rounded-lg transition-colors"
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-1">
                    F5-TTS Model *
                  </label>
                  <select
                    value={ttsModel}
                    onChange={(e) => setTtsModel(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    {f5Models.length === 0 ? (
                      <option value="">No F5-TTS models trained yet</option>
                    ) : (
                      f5Models.map((m) => (
                        <option key={m.id} value={m.name}>
                          {m.name} ({m.sample_count} samples)
                        </option>
                      ))
                    )}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-1">
                    Reference Audio Clip
                  </label>
                  <select
                    value={ttsRefSample}
                    onChange={(e) => setTtsRefSample(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="">Default Voice Profile</option>
                    {samples.map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.filename} ({(s.duration || s.duration_seconds || 0).toFixed(1)}s)
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm text-slate-300 mb-1">
                  <span>Speech Rate</span>
                  <span>{ttsSpeed.toFixed(2)}x</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="2.0"
                  step="0.05"
                  value={ttsSpeed}
                  onChange={(e) => setTtsSpeed(parseFloat(e.target.value))}
                  className="w-full accent-indigo-500 bg-slate-900"
                />
              </div>

              <button
                onClick={handleGenerateTTS}
                disabled={isGenerating || f5Models.length === 0}
                className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium py-3 rounded-lg transition-colors shadow flex items-center justify-center space-x-2"
              >
                {isGenerating ? (
                  <span>Generating Polish Speech...</span>
                ) : (
                  <span>Generate Speech</span>
                )}
              </button>
            </div>
          ) : (
            /* STS Form */
            <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-6 shadow-sm space-y-5">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">
                  Source Audio Sample *
                </label>
                <select
                  value={stsInputSample}
                  onChange={(e) => setStsInputSample(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  {samples.length === 0 ? (
                    <option value="">No audio samples recorded/uploaded</option>
                  ) : (
                    samples.map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.filename} ({(s.duration || s.duration_seconds || 0).toFixed(1)}s) - {s.transcript || 'No transcript'}
                      </option>
                    ))
                  )}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">
                  RVC v2 Model *
                </label>
                <select
                  value={stsModel}
                  onChange={(e) => setStsModel(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  {rvcModels.length === 0 ? (
                    <option value="">No RVC models trained yet</option>
                  ) : (
                    rvcModels.map((m) => (
                      <option key={m.id} value={m.name}>
                        {m.name} (RVC v2)
                      </option>
                    ))
                  )}
                </select>
              </div>

              <div>
                <div className="flex justify-between text-sm text-slate-300 mb-1">
                  <span>Pitch Shift</span>
                  <span className="font-mono text-indigo-400">
                    {pitchShift === 0 ? '0 (Original)' : `${pitchShift > 0 ? '+' : ''}${pitchShift} semitones`}
                  </span>
                </div>
                <input
                  type="range"
                  min="-24"
                  max="24"
                  step="1"
                  value={pitchShift}
                  onChange={(e) => setPitchShift(parseInt(e.target.value))}
                  className="w-full accent-indigo-500 bg-slate-900"
                />
                <div className="flex justify-between text-[10px] text-slate-500 px-1 mt-0.5">
                  <span>-24 (Octaves down)</span>
                  <span>0 (Neutral)</span>
                  <span>+24 (Octaves up)</span>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm text-slate-300 mb-1">
                  <span>Index Rate (Feature Retrieval Blend)</span>
                  <span className="font-mono text-indigo-400">{indexRate.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={indexRate}
                  onChange={(e) => setIndexRate(parseFloat(e.target.value))}
                  className="w-full accent-indigo-500 bg-slate-900"
                />
              </div>

              <button
                onClick={handleGenerateSTS}
                disabled={isGenerating || rvcModels.length === 0 || !stsInputSample}
                className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium py-3 rounded-lg transition-colors shadow flex items-center justify-center space-x-2"
              >
                {isGenerating ? (
                  <span>Converting Voice...</span>
                ) : (
                  <span>Convert Voice</span>
                )}
              </button>
            </div>
          )}
        </div>

        {/* Audio Output Column (1 col) */}
        <div className="space-y-6">
          <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-6 shadow-sm space-y-4">
            <h2 className="text-xl font-semibold">Audio Output</h2>

            {currentAudio ? (
              <div className="space-y-4">
                <div className="bg-slate-900/80 p-4 rounded-lg border border-slate-700 space-y-3">
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-semibold text-slate-300 truncate max-w-[180px]">
                      {currentAudio.filename}
                    </span>
                    <span className="bg-indigo-950 text-indigo-300 px-2 py-0.5 rounded uppercase font-semibold text-[10px]">
                      {currentAudio.type}
                    </span>
                  </div>

                  {/* HTML5 Audio Player */}
                  <audio
                    ref={audioRef}
                    controls
                    src={currentAudio.url}
                    className="w-full mt-2"
                  />

                  {currentAudio.duration && (
                    <div className="text-xs text-slate-400">
                      Duration: {currentAudio.duration.toFixed(2)}s
                    </div>
                  )}
                </div>

                <a
                  href={api.getDownloadAudioUrl(currentAudio.filename)}
                  download={currentAudio.filename}
                  className="w-full bg-slate-700 hover:bg-slate-600 text-white font-medium py-2.5 px-4 rounded-lg transition-colors shadow flex items-center justify-center space-x-2 text-sm"
                >
                  <span>Download Audio (.wav)</span>
                </a>
              </div>
            ) : (
              <div className="py-12 text-center text-slate-500 border border-dashed border-slate-700 rounded-lg">
                <p className="text-sm">No audio generated yet.</p>
                <p className="text-xs text-slate-600 mt-1">
                  Synthesize text or convert speech to hear output.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Generation History Table */}
      <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-6 shadow-sm space-y-4">
        <h2 className="text-xl font-semibold">Generation History</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-300">
            <thead className="bg-slate-900/80 text-xs uppercase text-slate-400 border-b border-slate-700">
              <tr>
                <th className="p-3">Time</th>
                <th className="p-3">Type</th>
                <th className="p-3">Model</th>
                <th className="p-3">Prompt / Input</th>
                <th className="p-3">Duration</th>
                <th className="p-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/60">
              {history.length === 0 ? (
                <tr>
                  <td colSpan={6} className="p-6 text-center text-slate-500 italic">
                    No speech generation history found.
                  </td>
                </tr>
              ) : (
                history.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-800/40">
                    <td className="p-3 text-slate-400 whitespace-nowrap">
                      {new Date(item.created_at).toLocaleTimeString()}
                    </td>
                    <td className="p-3">
                      <span
                        className={`text-xs px-2 py-0.5 rounded font-semibold ${
                          item.type === 'tts'
                            ? 'bg-blue-950 text-blue-300'
                            : 'bg-purple-950 text-purple-300'
                        }`}
                      >
                        {item.type.toUpperCase()}
                      </span>
                    </td>
                    <td className="p-3 font-medium text-slate-200">{item.model_id}</td>
                    <td className="p-3 max-w-xs truncate text-slate-300">
                      {item.input_text || item.input_audio_path || '--'}
                    </td>
                    <td className="p-3 text-slate-400 whitespace-nowrap">
                      {(item.duration_seconds || 0).toFixed(1)}s
                    </td>
                    <td className="p-3 text-right space-x-2 whitespace-nowrap">
                      <button
                        onClick={() => handlePlayFromHistory(item)}
                        className="text-xs text-indigo-400 hover:text-indigo-300 font-medium px-2 py-1 rounded hover:bg-indigo-950/40"
                      >
                        Play
                      </button>
                      <a
                        href={api.getDownloadAudioUrl(item.output_filename)}
                        download={item.output_filename}
                        className="text-xs text-slate-400 hover:text-slate-200 font-medium px-2 py-1 rounded hover:bg-slate-700"
                      >
                        Download
                      </a>
                      <button
                        onClick={() => handleDeleteHistory(item.id)}
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
    </div>
  );
};
