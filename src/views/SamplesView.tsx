import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Sample, NotificationEvent } from '../types';
import { api } from '../api/client';
import { notificationWebSocket } from '../api/websocket';
import { WaveformVisualizer } from '../components/WaveformVisualizer';
import { StatusBadge } from '../components/StatusBadge';
import { AudioPlayer } from '../components/AudioPlayer';
import { DeleteConfirmModal } from '../components/DeleteConfirmModal';
import { TranscriptVerificationModal } from '../components/TranscriptVerificationModal';

const POLISH_DIACRITICS = ['ą', 'ć', 'ę', 'ł', 'ń', 'ó', 'ś', 'ź', 'ż'];
const POLISH_DIACRITICS_UPPER = ['Ą', 'Ć', 'Ę', 'Ł', 'Ń', 'Ó', 'Ś', 'Ź', 'Ż'];

export const SamplesView: React.FC = () => {
  // Sample Data State
  const [samples, setSamples] = useState<Sample[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Audio Playback Coordination (Only 1 plays at a time)
  const [activePlayingId, setActivePlayingId] = useState<string | null>(null);

  // Microphone Recording State
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [recordingSeconds, setRecordingSeconds] = useState<number>(0);
  const [micStream, setMicStream] = useState<MediaStream | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerIntervalRef = useRef<any>(null);

  // Drag and Drop Upload State
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<{ [filename: string]: number }>({});
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Transcript Inline Editor State
  const [editingSampleId, setEditingSampleId] = useState<string | null>(null);
  const [editingText, setEditingText] = useState<string>('');
  const [isSavingTranscript, setIsSavingTranscript] = useState<boolean>(false);
  const [diacriticUpper, setDiacriticUpper] = useState<boolean>(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Delete Modal State
  const [deleteTarget, setDeleteTarget] = useState<Sample | null>(null);
  const [showBulkDeleteModal, setShowBulkDeleteModal] = useState<boolean>(false);
  const [isBulkDeleting, setIsBulkDeleting] = useState<boolean>(false);

  // Transcript Verification Modal State
  const [inspectingSample, setInspectingSample] = useState<Sample | null>(null);

  // Expandable Table Transcripts State
  const [expandedTranscriptIds, setExpandedTranscriptIds] = useState<Set<string>>(new Set());

  const toggleExpandTranscript = (id: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    setExpandedTranscriptIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  // 1. Initial Load & Fetch Samples
  const loadSamples = useCallback(async () => {
    try {
      setLoading(true);
      const data = await api.getSamples();
      setSamples(data.samples);
      setErrorMessage(null);
    } catch (err: any) {
      setErrorMessage('Failed to connect to backend server. Make sure Python service is running.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSamples();
  }, [loadSamples]);

  // 2. WebSocket Notifications for Live Sample & Transcription Updates
  useEffect(() => {
    notificationWebSocket.connect();
    const unsubscribe = notificationWebSocket.subscribe((payload: NotificationEvent) => {
      if (payload.type === 'notification') {
        switch (payload.event) {
          case 'sample_created':
            setSamples((prev) => {
              if (prev.some((s) => s.id === payload.sample_id)) return prev;
              const newSample: Sample = {
                id: payload.sample_id,
                name: payload.filename,
                filename: payload.filename,
                duration: payload.duration,
                duration_seconds: payload.duration,
                transcript: '',
                status: 'pending',
                transcript_status: 'pending',
                sample_rate: 44100,
                channels: 1,
                created_at: payload.timestamp,
              };
              return [newSample, ...prev];
            });
            break;

          case 'transcription_started':
            setSamples((prev) =>
              prev.map((s) => (s.id === payload.sample_id ? { ...s, status: 'transcribing' } : s))
            );
            setInspectingSample((current) =>
              current && current.id === payload.sample_id
                ? { ...current, status: 'transcribing' }
                : current
            );
            break;

          case 'transcription_completed':
            setSamples((prev) =>
              prev.map((s) =>
                s.id === payload.sample_id
                  ? {
                      ...s,
                      status: 'transcribed',
                      transcript: payload.transcript,
                      duration: payload.duration > 0 ? payload.duration : s.duration,
                      duration_seconds: payload.duration > 0 ? payload.duration : s.duration_seconds,
                    }
                  : s
              )
            );
            setInspectingSample((current) =>
              current && current.id === payload.sample_id
                ? {
                    ...current,
                    status: 'transcribed',
                    transcript: payload.transcript,
                    duration: payload.duration > 0 ? payload.duration : current.duration,
                    duration_seconds: payload.duration > 0 ? payload.duration : current.duration_seconds,
                  }
                : current
            );
            break;

          case 'transcription_failed':
            setSamples((prev) =>
              prev.map((s) => (s.id === payload.sample_id ? { ...s, status: 'error' } : s))
            );
            setInspectingSample((current) =>
              current && current.id === payload.sample_id
                ? { ...current, status: 'error' }
                : current
            );
            break;

          case 'sample_updated':
            setSamples((prev) =>
              prev.map((s) =>
                s.id === payload.sample_id
                  ? { ...s, transcript: payload.transcript, status: payload.status as any }
                  : s
              )
            );
            setInspectingSample((current) =>
              current && current.id === payload.sample_id
                ? { ...current, transcript: payload.transcript, status: payload.status as any }
                : current
            );
            break;

          case 'sample_deleted':
            setSamples((prev) => prev.filter((s) => s.id !== payload.sample_id));
            setInspectingSample((current) => (current && current.id === payload.sample_id ? null : current));
            break;
        }
      }
    });

    return () => {
      unsubscribe();
    };
  }, []);

  // 3. Microphone Unmount Cleanup Effect (prevents background recording / hardware leaks)
  useEffect(() => {
    return () => {
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
      if (mediaRecorderRef.current) {
        if (mediaRecorderRef.current.state !== 'inactive') {
          try {
            mediaRecorderRef.current.onstop = null;
            mediaRecorderRef.current.stop();
          } catch {
            // ignore error during unmount stop
          }
        }
        mediaRecorderRef.current = null;
      }
      if (micStreamRef.current) {
        micStreamRef.current.getTracks().forEach((track) => {
          try {
            track.stop();
          } catch {
            // ignore error
          }
        });
        micStreamRef.current = null;
      }
    };
  }, []);

  // 4. Recording Mechanics
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      micStreamRef.current = stream;
      setMicStream(stream);

      // Determine supported container format
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm';

      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `recording_${timestamp}.webm`;

        try {
          const newSample = await api.recordSample(audioBlob, filename);
          setSamples((prev) => [newSample, ...prev.filter((s) => s.id !== newSample.id)]);
        } catch (err: any) {
          setErrorMessage('Failed to save recorded audio to backend.');
        } finally {
          stream.getTracks().forEach((track) => track.stop());
          micStreamRef.current = null;
          setMicStream(null);
        }
      };

      mediaRecorder.start(100);
      setIsRecording(true);
      setIsPaused(false);
      setRecordingSeconds(0);

      timerIntervalRef.current = setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err: any) {
      setErrorMessage(`Microphone access error: ${err.message || 'Permission denied'}`);
    }
  };

  const pauseRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      if (isPaused) {
        mediaRecorderRef.current.resume();
        setIsPaused(false);
        timerIntervalRef.current = setInterval(() => {
          setRecordingSeconds((prev) => prev + 1);
        }, 1000);
      } else {
        mediaRecorderRef.current.pause();
        setIsPaused(true);
        if (timerIntervalRef.current) {
          clearInterval(timerIntervalRef.current);
          timerIntervalRef.current = null;
        }
      }
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsPaused(false);
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
    }
  };

  // 4. File Upload & Drag and Drop
  const handleFiles = async (files: FileList | File[]) => {
    const validExtensions = ['.wav', '.mp3', '.flac', '.ogg'];
    const validFiles = Array.from(files).filter((file) =>
      validExtensions.some((ext) => file.name.toLowerCase().endsWith(ext))
    );

    if (validFiles.length === 0) {
      setErrorMessage('Please provide valid audio files (.wav, .mp3, .flac).');
      return;
    }

    for (const file of validFiles) {
      try {
        setUploadProgress((prev) => ({ ...prev, [file.name]: 0 }));
        const createdSample = await api.uploadSample(file, (percent) => {
          setUploadProgress((prev) => ({ ...prev, [file.name]: percent }));
        });
        setSamples((prev) => [createdSample, ...prev.filter((s) => s.id !== createdSample.id)]);
      } catch (err: any) {
        setErrorMessage(`Upload failed for ${file.name}: ${err.message}`);
      } finally {
        setUploadProgress((prev) => {
          const next = { ...prev };
          delete next[file.name];
          return next;
        });
      }
    }
  };

  // 5. Transcript Inline Editing
  const startEditingTranscript = (sample: Sample) => {
    setEditingSampleId(sample.id);
    setEditingText(sample.transcript);
    setTimeout(() => {
      textareaRef.current?.focus();
    }, 50);
  };

  const saveTranscript = async () => {
    if (!editingSampleId) return;
    try {
      setIsSavingTranscript(true);
      const updated = await api.updateTranscript(editingSampleId, editingText);
      setSamples((prev) => prev.map((s) => (s.id === updated.id ? updated : s)));
      setEditingSampleId(null);
    } catch (err: any) {
      setErrorMessage('Failed to save updated transcript.');
    } finally {
      setIsSavingTranscript(false);
    }
  };

  const cancelEditingTranscript = () => {
    setEditingSampleId(null);
    setEditingText('');
  };

  const insertDiacritic = (char: string) => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const text = editingText;
    const updated = text.substring(0, start) + char + text.substring(end);
    setEditingText(updated);

    setTimeout(() => {
      textarea.focus();
      textarea.setSelectionRange(start + char.length, start + char.length);
    }, 0);
  };

  // 6. Delete Action
  const confirmDelete = async () => {
    if (!deleteTarget) return;
    try {
      await api.deleteSample(deleteTarget.id);
      setSamples((prev) => prev.filter((s) => s.id !== deleteTarget.id));
      if (activePlayingId === deleteTarget.id) {
        setActivePlayingId(null);
      }
      setDeleteTarget(null);
    } catch (err: any) {
      setErrorMessage(`Failed to delete sample: ${err.message}`);
    }
  };

  const confirmBulkDelete = async (onlyMissing = false) => {
    try {
      setIsBulkDeleting(true);
      await api.deleteAllSamples(onlyMissing);
      setShowBulkDeleteModal(false);
      setActivePlayingId(null);
      await loadSamples();
    } catch (err: any) {
      setErrorMessage(`Failed to bulk delete samples: ${err.message}`);
    } finally {
      setIsBulkDeleting(false);
    }
  };

  const formatTimer = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="flex flex-col gap-6 max-w-7xl mx-auto px-6 py-8">
      {/* View Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-100">Voice Samples</h1>
          <p className="text-sm text-slate-400 mt-1">
            Record voice clips or upload audio files to train your custom Polish voice cloning models.
          </p>
        </div>

        {/* Quick Stats Pill */}
        <div className="flex items-center gap-4 bg-slate-900 border border-slate-800 px-4 py-2 rounded-xl text-sm">
          <div>
            <span className="text-slate-400">Total Samples:</span>{' '}
            <span className="font-semibold text-slate-100">{samples.length}</span>
          </div>
          <div className="w-px h-4 bg-slate-700" />
          <div>
            <span className="text-slate-400">Total Duration:</span>{' '}
            <span className="font-semibold text-slate-100">
              {formatTimer(Math.round(samples.reduce((acc, s) => acc + (s.duration || s.duration_seconds || 0), 0)))}
            </span>
          </div>
        </div>
      </div>

      {/* Error Banner */}
      {errorMessage && (
        <div className="flex items-center justify-between p-4 bg-red-950/60 border border-red-800 text-red-200 rounded-xl text-sm">
          <span>{errorMessage}</span>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-red-400 hover:text-red-200 font-bold ml-4"
          >
            ✕
          </button>
        </div>
      )}

      {/* Audio Ingestion Grid: Recorder & Dropzone */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Card 1: Microphone Recorder & 60fps Visualizer */}
        <div className="flex flex-col bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-base font-semibold text-slate-200 flex items-center gap-2">
              <svg className="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
              Microphone Recorder
            </h2>
            {isRecording && (
              <span className="font-mono text-sm px-2.5 py-1 rounded-md bg-red-950/80 text-red-400 border border-red-800 animate-pulse font-bold">
                {formatTimer(recordingSeconds)}
              </span>
            )}
          </div>

          {/* 60fps Waveform Visualizer */}
          <WaveformVisualizer
            mediaStream={micStream}
            isRecording={isRecording}
            isPaused={isPaused}
            height={130}
          />

          {/* Recorder Controls */}
          <div className="flex items-center justify-center gap-3 mt-4">
            {!isRecording ? (
              <button
                type="button"
                onClick={startRecording}
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-white bg-indigo-600 hover:bg-indigo-500 shadow-md shadow-indigo-600/20 transition-all focus:outline-none focus:ring-2 focus:ring-indigo-400"
              >
                <span className="w-3 h-3 rounded-full bg-red-400" />
                Start Recording
              </button>
            ) : (
              <>
                <button
                  type="button"
                  onClick={pauseRecording}
                  className="px-4 py-2 rounded-xl text-sm font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 border border-slate-700 transition-colors"
                >
                  {isPaused ? 'Resume' : 'Pause'}
                </button>
                <button
                  type="button"
                  onClick={stopRecording}
                  className="px-5 py-2.5 rounded-xl text-sm font-medium text-white bg-red-600 hover:bg-red-500 shadow-md shadow-red-600/20 transition-colors"
                >
                  Stop & Auto-Save
                </button>
              </>
            )}
          </div>
        </div>

        {/* Card 2: Drag and Drop Upload Area */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
              handleFiles(e.dataTransfer.files);
            }
          }}
          onClick={() => fileInputRef.current?.click()}
          className={`flex flex-col items-center justify-center p-6 border-2 border-dashed rounded-2xl cursor-pointer transition-all ${
            isDragging
              ? 'border-indigo-400 bg-indigo-950/20 scale-[1.01]'
              : 'border-slate-800 hover:border-slate-700 bg-slate-900/60'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.flac,.ogg"
            multiple
            className="hidden"
            onChange={(e) => {
              if (e.target.files) handleFiles(e.target.files);
            }}
          />
          <div className="w-12 h-12 rounded-full bg-slate-800 text-slate-300 flex items-center justify-center mb-3">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
          </div>
          <p className="text-base font-semibold text-slate-200">Drag & drop voice files here</p>
          <p className="text-xs text-slate-400 mt-1">Supports WAV, MP3, FLAC (auto-normalized to 44.1kHz)</p>
          <span className="mt-3 text-xs font-medium text-indigo-400 bg-indigo-950/60 border border-indigo-800/60 px-3 py-1 rounded-full">
            or browse local files
          </span>

          {/* Active upload progress */}
          {Object.keys(uploadProgress).length > 0 && (
            <div className="w-full mt-4 flex flex-col gap-2">
              {Object.entries(uploadProgress).map(([fname, pct]) => (
                <div key={fname} className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-indigo-500 h-full transition-all duration-150"
                    style={{ width: `${pct}%` }}
                  />
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Samples Table & Transcript Editor Section */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl shadow-xl overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h2 className="text-base font-semibold text-slate-200">Sample Registry & Transcripts</h2>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-400 font-mono">
              {samples.length}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {samples.length > 0 && (
              <button
                type="button"
                onClick={() => setShowBulkDeleteModal(true)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-red-400 hover:text-red-300 hover:bg-red-950/40 border border-red-900/40 transition-colors"
                title="Mass delete samples"
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                Clear Samples
              </button>
            )}
            <button
              type="button"
              onClick={loadSamples}
              aria-label="Refresh samples list"
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          </div>
        </div>

        {loading ? (
          <div className="py-16 text-center text-slate-400">
            <svg className="animate-spin h-8 w-8 mx-auto text-indigo-500 mb-3" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Loading voice samples...
          </div>
        ) : samples.length === 0 ? (
          <div className="py-16 text-center text-slate-500">
            <p className="text-base font-medium">No samples recorded yet</p>
            <p className="text-sm mt-1">Record a sample using your microphone or drag and drop audio files above.</p>
          </div>
        ) : (
          <div className="overflow-x-auto" tabIndex={0} role="region" aria-label="Voice samples table">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-xs font-semibold uppercase tracking-wider text-slate-400 bg-slate-900/90">
                  <th scope="col" className="py-3.5 px-4 w-12 text-center">Play</th>
                  <th scope="col" className="py-3.5 px-4 min-w-[200px]">Filename & Audio</th>
                  <th scope="col" className="py-3.5 px-4 min-w-[320px]">Transcript (Polish Whisper)</th>
                  <th scope="col" className="py-3.5 px-4 w-32">Status</th>
                  <th scope="col" className="py-3.5 px-4 w-32">Added</th>
                  <th scope="col" className="py-3.5 px-4 w-28 text-center">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-sm">
                {samples.map((sample) => {
                  const isEditing = editingSampleId === sample.id;
                  const isPlaying = activePlayingId === sample.id;
                  const durationVal = sample.duration || sample.duration_seconds || 0;

                  return (
                    <tr key={sample.id} className="hover:bg-slate-800/30 transition-colors">
                      {/* Playback cell */}
                      <td className="py-4 px-4 align-top text-center">
                        <button
                          type="button"
                          onClick={() => setActivePlayingId(isPlaying ? null : sample.id)}
                          aria-label={isPlaying ? 'Pause sample' : 'Play sample'}
                          className="w-8 h-8 rounded-full bg-slate-800 hover:bg-indigo-600 text-slate-200 hover:text-white flex items-center justify-center transition-colors mx-auto"
                        >
                          {isPlaying ? (
                            <svg className="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24">
                              <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                            </svg>
                          ) : (
                            <svg className="w-3.5 h-3.5 fill-current ml-0.5" viewBox="0 0 24 24">
                              <path d="M8 5v14l11-7z" />
                            </svg>
                          )}
                        </button>
                      </td>

                      {/* Filename & Player */}
                      <td className="py-4 px-4 align-top">
                        <div className="font-mono text-sm font-medium text-slate-200 truncate max-w-xs" title={sample.filename}>
                          {sample.filename}
                        </div>
                        <div className="text-xs text-slate-400 mt-0.5">
                          {formatTimer(Math.round(durationVal))} ({sample.sample_rate || 44100}Hz)
                        </div>
                        <div className="mt-2">
                          <AudioPlayer
                            sampleId={sample.id}
                            duration={durationVal}
                            isPlaying={isPlaying}
                            onPlayToggle={(id) => setActivePlayingId(activePlayingId === id ? null : id)}
                          />
                        </div>
                      </td>

                      {/* Inline Transcript Editor */}
                      <td className="py-4 px-4 align-top">
                        {isEditing ? (
                          <div className="flex flex-col gap-2">
                            {/* Polish Diacritic Quick Bar */}
                            <div className="flex flex-wrap items-center gap-1">
                              <span className="text-xs text-slate-500 mr-1">Polish chars:</span>
                              {(diacriticUpper ? POLISH_DIACRITICS_UPPER : POLISH_DIACRITICS).map((char) => (
                                <button
                                  key={char}
                                  type="button"
                                  onClick={() => insertDiacritic(char)}
                                  className="w-6 h-6 rounded bg-slate-800 hover:bg-indigo-600 hover:text-white text-slate-300 text-xs font-mono font-bold flex items-center justify-center transition-colors border border-slate-700"
                                >
                                  {char}
                                </button>
                              ))}
                              <button
                                type="button"
                                onClick={() => setDiacriticUpper(!diacriticUpper)}
                                className="text-[10px] text-indigo-400 hover:text-indigo-300 font-medium ml-1"
                              >
                                {diacriticUpper ? 'Shift [UPPER]' : 'Shift [lower]'}
                              </button>
                            </div>

                            {/* Auto-sizing Textarea */}
                            <textarea
                              ref={textareaRef}
                              value={editingText}
                              onChange={(e) => setEditingText(e.target.value)}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                                  e.preventDefault();
                                  saveTranscript();
                                } else if (e.key === 'Escape') {
                                  e.preventDefault();
                                  cancelEditingTranscript();
                                }
                              }}
                              className="w-full px-3 py-2 text-sm bg-slate-950 border border-indigo-500/80 rounded-lg text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-400 resize-y"
                              style={{ fieldSizing: 'content', minHeight: '4.5rem' } as any}
                              placeholder="Enter Polish transcript..."
                            />

                            {/* Actions */}
                            <div className="flex items-center justify-end gap-2">
                              <span className="text-xs text-slate-500 mr-auto">
                                Press <kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-400 font-mono">Ctrl+Enter</kbd> to save
                              </span>
                              <button
                                type="button"
                                onClick={cancelEditingTranscript}
                                disabled={isSavingTranscript}
                                className="px-3 py-1 text-xs rounded-md bg-slate-800 text-slate-300 hover:bg-slate-700 transition-colors"
                              >
                                Cancel
                              </button>
                              <button
                                type="button"
                                onClick={saveTranscript}
                                disabled={isSavingTranscript}
                                className="px-3 py-1 text-xs rounded-md bg-indigo-600 text-white font-medium hover:bg-indigo-500 transition-colors"
                              >
                                {isSavingTranscript ? 'Saving...' : 'Save Transcript'}
                              </button>
                            </div>
                          </div>
                        ) : (
                          <div className="flex flex-col gap-1.5">
                            {(() => {
                              const isExpanded = expandedTranscriptIds.has(sample.id);
                              const textLen = (sample.transcript || '').length;
                              const isLong = textLen > 110;

                              return (
                                <>
                                  <div
                                    onClick={() => startEditingTranscript(sample)}
                                    className="group cursor-pointer rounded-lg p-2.5 -m-2 hover:bg-slate-800/60 transition-colors"
                                    title="Kliknij, aby edytować tekst w wierszu"
                                  >
                                    <p
                                      className={`text-sm text-slate-200 leading-relaxed font-normal transition-all ${
                                        !isExpanded && isLong ? 'line-clamp-2 max-h-12 overflow-hidden' : 'max-h-96 overflow-y-auto'
                                      }`}
                                    >
                                      {sample.transcript ? (
                                        sample.transcript
                                      ) : (
                                        <span className="text-slate-500 italic">Brak transkrypcji (kliknij aby edytować).</span>
                                      )}
                                    </p>
                                    <span className="opacity-0 group-hover:opacity-100 text-[11px] text-indigo-400 mt-1 inline-flex items-center gap-1 transition-opacity">
                                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                                      </svg>
                                      Edytuj w wierszu
                                    </span>
                                  </div>

                                  <div className="flex items-center gap-2 flex-wrap mt-0.5">
                                    {isLong && (
                                      <button
                                        type="button"
                                        onClick={(e) => toggleExpandTranscript(sample.id, e)}
                                        className="inline-flex items-center gap-1 text-[11px] font-semibold text-indigo-400 hover:text-indigo-300 transition-colors py-0.5"
                                      >
                                        {isExpanded ? (
                                          <>
                                            <svg className="w-3 h-3 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 15l7-7 7 7" />
                                            </svg>
                                            <span>Zwiń tekst</span>
                                          </>
                                        ) : (
                                          <>
                                            <svg className="w-3 h-3 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                                            </svg>
                                            <span>Rozwiń tekst ({textLen} znaków)</span>
                                          </>
                                        )}
                                      </button>
                                    )}

                                    <button
                                      type="button"
                                      onClick={() => setInspectingSample(sample)}
                                      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium text-indigo-300 bg-indigo-950/50 hover:bg-indigo-900/60 border border-indigo-800/50 transition-colors"
                                    >
                                      <svg className="w-3.5 h-3.5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                                      </svg>
                                      <span>Weryfikuj i odsłuchaj</span>
                                    </button>
                                  </div>
                                </>
                              );
                            })()}
                          </div>
                        )}
                      </td>

                      {/* Status */}
                      <td className="py-4 px-4 align-top">
                        <StatusBadge status={sample.status} />
                      </td>

                      {/* Created date */}
                      <td className="py-4 px-4 align-top text-xs text-slate-400 font-mono whitespace-nowrap">
                        {new Date(sample.created_at).toLocaleDateString()}
                        <div className="text-[11px] text-slate-500">
                          {new Date(sample.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </div>
                      </td>

                      {/* Actions */}
                      <td className="py-4 px-4 align-top text-center">
                        <div className="flex items-center justify-center gap-1.5">
                          <button
                            type="button"
                            onClick={() => setInspectingSample(sample)}
                            aria-label={`Weryfikuj transkrypcję ${sample.filename}`}
                            className="p-1.5 rounded-lg text-indigo-400 hover:text-indigo-200 hover:bg-indigo-950/60 border border-indigo-900/40 transition-colors"
                            title="Otwórz okno weryfikacji transkrypcji"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                            </svg>
                          </button>

                          <button
                            type="button"
                            onClick={() => setDeleteTarget(sample)}
                            aria-label={`Delete ${sample.filename}`}
                            className="p-1.5 rounded-lg text-slate-400 hover:text-red-400 hover:bg-red-950/40 transition-colors"
                            title="Usuń próbkę"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      <DeleteConfirmModal
        isOpen={deleteTarget !== null}
        filename={deleteTarget?.filename || ''}
        onConfirm={confirmDelete}
        onCancel={() => setDeleteTarget(null)}
      />

      {/* Bulk Delete Modal */}
      {showBulkDeleteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/75 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-slate-900 border border-slate-800 text-slate-100 p-6 rounded-2xl shadow-2xl max-w-md w-full">
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 rounded-full bg-red-900/40 text-red-400 border border-red-800/50 flex items-center justify-center flex-shrink-0">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-slate-100">Mass Delete Samples</h3>
                <p className="text-sm text-slate-400 mt-2">
                  You currently have <span className="font-semibold text-slate-200">{samples.length}</span> sample(s) registered.
                  Choose an action:
                </p>

                <div className="flex flex-col gap-2.5 mt-5">
                  <button
                    type="button"
                    onClick={() => confirmBulkDelete(true)}
                    disabled={isBulkDeleting}
                    className="w-full px-4 py-3 rounded-xl text-xs font-semibold text-amber-300 bg-amber-950/40 hover:bg-amber-900/50 border border-amber-800/60 transition-colors text-left flex flex-col gap-0.5"
                  >
                    <span className="flex items-center gap-1.5 font-bold">
                      <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                      Delete Broken / Missing Only
                    </span>
                    <span className="text-[11px] text-amber-400/80 font-normal ml-5.5">
                      Removes only ghost database records whose audio files are missing from disk.
                    </span>
                  </button>

                  <button
                    type="button"
                    onClick={() => confirmBulkDelete(false)}
                    disabled={isBulkDeleting}
                    className="w-full px-4 py-3 rounded-xl text-xs font-semibold text-white bg-red-600 hover:bg-red-500 shadow-md shadow-red-600/20 transition-colors text-left flex flex-col gap-0.5"
                  >
                    <span className="flex items-center gap-1.5 font-bold">
                      <svg className="w-4 h-4 text-red-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                      Delete All Samples
                    </span>
                    <span className="text-[11px] text-red-200 font-normal ml-5.5">
                      Purges all samples completely from the database and audio folders.
                    </span>
                  </button>
                </div>

                <div className="flex justify-end mt-5">
                  <button
                    type="button"
                    onClick={() => setShowBulkDeleteModal(false)}
                    disabled={isBulkDeleting}
                    className="px-4 py-2 rounded-xl text-sm font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Transcript Verification Modal */}
      <TranscriptVerificationModal
        isOpen={inspectingSample !== null}
        sample={inspectingSample}
        onClose={() => setInspectingSample(null)}
        onSampleUpdated={(updated) => {
          setSamples((prev) => prev.map((s) => (s.id === updated.id ? updated : s)));
          setInspectingSample(updated);
        }}
      />
    </div>
  );
};
