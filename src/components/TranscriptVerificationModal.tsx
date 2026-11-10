import React, { useState, useEffect, useRef } from 'react';
import { Sample } from '../types';
import { api } from '../api/client';
import { StatusBadge } from './StatusBadge';

interface TranscriptVerificationModalProps {
  isOpen: boolean;
  sample: Sample | null;
  onClose: () => void;
  onSampleUpdated: (updatedSample: Sample) => void;
}

const POLISH_DIACRITICS = ['ą', 'ć', 'ę', 'ł', 'ń', 'ó', 'ś', 'ź', 'ż'];
const POLISH_DIACRITICS_UPPER = ['Ą', 'Ć', 'Ę', 'Ł', 'Ń', 'Ó', 'Ś', 'Ź', 'Ż'];

export const TranscriptVerificationModal: React.FC<TranscriptVerificationModalProps> = ({
  isOpen,
  sample,
  onClose,
  onSampleUpdated,
}) => {
  const dialogRef = useRef<HTMLDialogElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Editing state
  const [text, setText] = useState<string>('');
  const [diacriticUpper, setDiacriticUpper] = useState<boolean>(false);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [isRetranscribing, setIsRetranscribing] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Expandable sections state
  const [showDiagnostics, setShowDiagnostics] = useState<boolean>(false);
  const [isExpandedTextarea, setIsExpandedTextarea] = useState<boolean>(false);

  // Playback state
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const [playbackRate, setPlaybackRate] = useState<number>(1.0);
  const [isBuffering, setIsBuffering] = useState<boolean>(false);

  // Synchronize text & duration when sample changes
  useEffect(() => {
    if (sample) {
      setText(sample.transcript || '');
      setDuration(sample.duration || sample.duration_seconds || 0);
      setErrorMessage(null);
      setSuccessMessage(null);
      setIsPlaying(false);
      setCurrentTime(0);
    }
  }, [sample]);

  // Modal open/close sync
  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    if (isOpen) {
      if (!dialog.open) {
        dialog.showModal();
      }
    } else {
      if (dialog.open) {
        dialog.close();
      }
      if (audioRef.current) {
        audioRef.current.pause();
      }
      setIsPlaying(false);
    }
  }, [isOpen]);

  if (!sample) return null;

  const audioUrl = api.getAudioUrl(sample.id);

  // Playback handlers
  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      audio.play().then(() => setIsPlaying(true)).catch((e) => {
        console.warn('Playback error:', e);
        setIsPlaying(false);
      });
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
      if (audioRef.current.duration && !isNaN(audioRef.current.duration) && audioRef.current.duration > 0) {
        setDuration(audioRef.current.duration);
      }
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newTime = parseFloat(e.target.value);
    setCurrentTime(newTime);
    if (audioRef.current) {
      audioRef.current.currentTime = newTime;
    }
  };

  const changePlaybackRate = (rate: number) => {
    setPlaybackRate(rate);
    if (audioRef.current) {
      audioRef.current.playbackRate = rate;
    }
  };

  const skipTime = (delta: number) => {
    if (audioRef.current) {
      const target = Math.max(0, Math.min(duration, audioRef.current.currentTime + delta));
      audioRef.current.currentTime = target;
      setCurrentTime(target);
    }
  };

  // Polish Diacritic quick insert
  const insertDiacritic = (char: string) => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const newText = text.substring(0, start) + char + text.substring(end);
    setText(newText);

    setTimeout(() => {
      textarea.focus();
      textarea.setSelectionRange(start + char.length, start + char.length);
    }, 0);
  };

  // Save changes
  const handleSave = async (statusOverride = 'manual') => {
    try {
      setIsSaving(true);
      setErrorMessage(null);
      const updated = await api.updateTranscript(sample.id, text, statusOverride);
      onSampleUpdated(updated);
      setSuccessMessage('Zapisano transkrypcję pomyślnie!');
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err: any) {
      setErrorMessage(`Błąd zapisu: ${err.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  // Verify and approve
  const handleVerifyAndApprove = async () => {
    await handleSave('manual');
    setTimeout(() => {
      onClose();
    }, 400);
  };

  // Re-transcribe with Whisper
  const handleRetranscribe = async () => {
    try {
      setIsRetranscribing(true);
      setErrorMessage(null);
      const updated = await api.retranscribeSample(sample.id);
      onSampleUpdated(updated);
      setSuccessMessage('Uruchomiono ponowną automatyczną transkrypcję Whisper!');
      setTimeout(() => setSuccessMessage(null), 4000);
    } catch (err: any) {
      setErrorMessage(`Błąd uruchamiania transkrypcji: ${err.message}`);
    } finally {
      setIsRetranscribing(false);
    }
  };

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  // Polish statistics
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const charCount = text.length;
  const presentDiacritics = POLISH_DIACRITICS.filter((c) =>
    text.toLowerCase().includes(c)
  );

  const hasUnsavedChanges = text !== (sample.transcript || '');

  return (
    <dialog
      ref={dialogRef}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      className="backdrop:bg-slate-950/80 backdrop:backdrop-blur-sm bg-slate-900 border border-slate-800 text-slate-100 p-0 rounded-2xl shadow-2xl max-w-3xl w-full m-auto overflow-hidden animate-in fade-in duration-200"
    >
      <div className="flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-6 py-4 bg-slate-950/60 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-600/20 text-indigo-400 border border-indigo-500/30 flex items-center justify-center">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                Weryfikacja i podgląd transkrypcji
                <StatusBadge status={sample.status} />
              </h2>
              <p className="text-xs text-slate-400 font-mono mt-0.5 truncate max-w-md">
                {sample.filename} • {formatTime(duration)} • {sample.sample_rate || 44100}Hz
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
            aria-label="Zamknij"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 overflow-y-auto space-y-5">
          {/* Audio Player Card */}
          <div className="bg-slate-950/80 border border-slate-800/80 rounded-xl p-4 flex flex-col gap-3 shadow-inner">
            <audio
              ref={audioRef}
              src={audioUrl}
              preload="auto"
              onTimeUpdate={handleTimeUpdate}
              onEnded={() => {
                setIsPlaying(false);
                setCurrentTime(0);
              }}
              onWaiting={() => setIsBuffering(true)}
              onPlaying={() => setIsBuffering(false)}
            />

            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                <svg className="w-4 h-4 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                </svg>
                Odsłuch nagrania
              </span>

              {/* Speed Buttons */}
              <div className="flex items-center gap-1 bg-slate-900 border border-slate-800 rounded-lg p-0.5 text-xs font-mono">
                {[0.75, 1.0, 1.25].map((rate) => (
                  <button
                    key={rate}
                    type="button"
                    onClick={() => changePlaybackRate(rate)}
                    className={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
                      playbackRate === rate
                        ? 'bg-indigo-600 text-white font-bold'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {rate}x
                  </button>
                ))}
              </div>
            </div>

            {/* Playback Controls & Scrubber */}
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => skipTime(-3)}
                title="Cofnij 3 sekundy"
                className="w-8 h-8 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 flex items-center justify-center transition-colors text-xs font-bold"
              >
                -3s
              </button>

              <button
                type="button"
                onClick={togglePlay}
                className="w-10 h-10 rounded-full bg-indigo-600 hover:bg-indigo-500 text-white flex items-center justify-center transition-all shadow-md shadow-indigo-600/30 flex-shrink-0"
              >
                {isBuffering ? (
                  <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                ) : isPlaying ? (
                  <svg className="w-5 h-5 fill-current" viewBox="0 0 24 24">
                    <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5 fill-current ml-0.5" viewBox="0 0 24 24">
                    <path d="M8 5v14l11-7z" />
                  </svg>
                )}
              </button>

              <button
                type="button"
                onClick={() => skipTime(3)}
                title="Przewiń 3 sekundy"
                className="w-8 h-8 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 flex items-center justify-center transition-colors text-xs font-bold"
              >
                +3s
              </button>

              <div className="flex-1 flex items-center gap-2">
                <input
                  type="range"
                  min="0"
                  max={duration > 0 ? duration : 1}
                  step="0.05"
                  value={currentTime}
                  onChange={handleSeek}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500 focus:outline-none"
                />
              </div>

              <div className="font-mono text-xs text-slate-400 whitespace-nowrap min-w-[70px] text-right">
                {formatTime(currentTime)} / {formatTime(duration)}
              </div>
            </div>
          </div>

          {/* Collapsible Diagnostics Section */}
          <div className="border border-slate-800/80 rounded-xl bg-slate-950/40 overflow-hidden">
            <button
              type="button"
              onClick={() => setShowDiagnostics(!showDiagnostics)}
              className="w-full px-4 py-2 flex items-center justify-between text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-800/40 transition-colors"
            >
              <span className="flex items-center gap-2 font-medium">
                <svg className="w-4 h-4 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <span>Statystyki tekstu i polskie diakrytyki</span>
                <span className="text-[11px] text-slate-500 font-mono">
                  ({wordCount} słów, {charCount} znaków, {presentDiacritics.length}/9 diakrytyków)
                </span>
              </span>
              <span className="text-xs font-semibold text-indigo-400 flex items-center gap-1">
                {showDiagnostics ? 'Zwiń ▲' : 'Rozwiń ▼'}
              </span>
            </button>

            {showDiagnostics && (
              <div className="p-3 pt-1 border-t border-slate-800/60 space-y-3">
                {/* Diagnostics Card */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs">
                  <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3 flex flex-col">
                    <span className="text-slate-400 font-medium">Liczba słów</span>
                    <span className="text-base font-bold text-slate-100 font-mono mt-0.5">{wordCount}</span>
                  </div>
                  <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3 flex flex-col">
                    <span className="text-slate-400 font-medium">Liczba znaków</span>
                    <span className="text-base font-bold text-slate-100 font-mono mt-0.5">{charCount}</span>
                  </div>
                  <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3 flex flex-col">
                    <span className="text-slate-400 font-medium">Polskie znaki</span>
                    <span className="text-base font-bold text-indigo-400 font-mono mt-0.5">
                      {presentDiacritics.length} / 9
                    </span>
                  </div>
                  <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3 flex flex-col">
                    <span className="text-slate-400 font-medium">Status weryfikacji</span>
                    <span className="text-xs font-bold text-slate-200 mt-1 truncate">
                      {sample.status === 'manual' ? '✅ Zweryfikowana' : '🤖 Auto Whisper'}
                    </span>
                  </div>
                </div>

                {/* Polish Diacritic Badges */}
                <div className="flex flex-wrap items-center gap-1.5 p-2.5 bg-slate-950/60 border border-slate-800/60 rounded-xl text-xs">
                  <span className="text-slate-400 text-[11px] font-medium mr-1">Wykryte polskie litery:</span>
                  {POLISH_DIACRITICS.map((char) => {
                    const isPresent = text.toLowerCase().includes(char);
                    return (
                      <span
                        key={char}
                        className={`px-2 py-0.5 rounded font-mono font-bold text-xs transition-colors ${
                          isPresent
                            ? 'bg-indigo-950/80 border border-indigo-700/80 text-indigo-300'
                            : 'bg-slate-900 border border-slate-800/50 text-slate-600'
                        }`}
                        title={isPresent ? `Znak "${char}" obecny w transkrypcji` : `Brak znaku "${char}"`}
                      >
                        {char}
                      </span>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* Transcript Editor Section */}
          <div className="space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <label className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <span>Treść transkrypcji (tekst mówiony)</span>
                  {hasUnsavedChanges && (
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-950/80 text-amber-300 border border-amber-800">
                      Niezapisane zmiany
                    </span>
                  )}
                </label>
                <button
                  type="button"
                  onClick={() => setIsExpandedTextarea(!isExpandedTextarea)}
                  className="px-2 py-0.5 text-xs text-indigo-400 hover:text-indigo-300 hover:bg-indigo-950/50 border border-indigo-800/40 rounded flex items-center gap-1 transition-colors"
                  title={isExpandedTextarea ? 'Zwiń pole edycji' : 'Rozwiń pole edycji (wygodny podgląd dużego tekstu)'}
                >
                  <svg
                    className={`w-3.5 h-3.5 transition-transform ${isExpandedTextarea ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                  </svg>
                  <span>{isExpandedTextarea ? 'Zwiń pole' : 'Rozwiń pole'}</span>
                </button>
              </div>

              {/* Quick Diacritics insertion toolbar */}
              <div className="flex items-center gap-1">
                {(diacriticUpper ? POLISH_DIACRITICS_UPPER : POLISH_DIACRITICS).map((char) => (
                  <button
                    key={char}
                    type="button"
                    onClick={() => insertDiacritic(char)}
                    className="w-6 h-6 rounded bg-slate-800 hover:bg-indigo-600 hover:text-white text-slate-300 text-xs font-mono font-bold flex items-center justify-center transition-colors border border-slate-700"
                    title={`Wstaw ${char}`}
                  >
                    {char}
                  </button>
                ))}
                <button
                  type="button"
                  onClick={() => setDiacriticUpper(!diacriticUpper)}
                  className="text-[10px] text-indigo-400 hover:text-indigo-300 font-medium ml-1"
                >
                  {diacriticUpper ? 'Shift [UP]' : 'Shift [dw]'}
                </button>
              </div>
            </div>

            <textarea
              ref={textareaRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Tutaj pojawi się transkrypcja Whisper lub wpisz/popraw własny tekst..."
              rows={isExpandedTextarea ? 12 : 4}
              className={`w-full px-4 py-3 text-sm bg-slate-950 border border-slate-800 rounded-xl text-slate-100 placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/80 focus:border-indigo-500 resize-y font-normal leading-relaxed shadow-inner transition-[height] ${
                isExpandedTextarea ? 'min-h-[16rem] max-h-[26rem]' : 'min-h-[5.5rem] max-h-36'
              }`}
            />
          </div>

          {/* Feedback messages */}
          {errorMessage && (
            <div className="p-3 bg-red-950/60 border border-red-800 text-red-200 rounded-xl text-xs flex items-center justify-between">
              <span>{errorMessage}</span>
              <button onClick={() => setErrorMessage(null)} className="text-red-400 font-bold ml-2">✕</button>
            </div>
          )}
          {successMessage && (
            <div className="p-3 bg-emerald-950/60 border border-emerald-800 text-emerald-200 rounded-xl text-xs flex items-center justify-between">
              <span>{successMessage}</span>
              <button onClick={() => setSuccessMessage(null)} className="text-emerald-400 font-bold ml-2">✕</button>
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className="px-6 py-4 bg-slate-950/60 border-t border-slate-800 flex flex-wrap items-center justify-between gap-3">
          <button
            type="button"
            onClick={handleRetranscribe}
            disabled={isRetranscribing || sample.status === 'transcribing'}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold text-indigo-300 bg-indigo-950/60 hover:bg-indigo-900/60 border border-indigo-800/80 transition-colors disabled:opacity-50"
            title="Uruchom ponownie automatyczną transkrypcję Whisper"
          >
            <svg
              className={`w-3.5 h-3.5 ${isRetranscribing || sample.status === 'transcribing' ? 'animate-spin' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            {isRetranscribing || sample.status === 'transcribing' ? 'Transkrybuję...' : 'Ponów transkrypcję (Whisper)'}
          </button>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
            >
              Zamknij
            </button>

            <button
              type="button"
              onClick={() => handleSave('manual')}
              disabled={isSaving}
              className="px-4 py-2 rounded-xl text-xs font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 border border-slate-700 transition-colors disabled:opacity-50"
            >
              {isSaving ? 'Zapisuję...' : 'Zapisz'}
            </button>

            <button
              type="button"
              onClick={handleVerifyAndApprove}
              disabled={isSaving}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold text-white bg-emerald-600 hover:bg-emerald-500 shadow-md shadow-emerald-600/20 transition-colors disabled:opacity-50"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
              </svg>
              Zatwierdź jako poprawne
            </button>
          </div>
        </div>
      </div>
    </dialog>
  );
};
