import React, { useEffect, useRef } from 'react';

interface WaveformVisualizerProps {
  mediaStream: MediaStream | null;
  isRecording: boolean;
  isPaused: boolean;
  height?: number;
  className?: string;
}

export const WaveformVisualizer: React.FC<WaveformVisualizerProps> = ({
  mediaStream,
  isRecording,
  isPaused,
  height = 120,
  className = '',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const timeRef = useRef<number>(0);

  useEffect(() => {
    if (!mediaStream || !isRecording) {
      // Teardown Web Audio resources when not recording
      if (sourceRef.current) {
        try {
          sourceRef.current.disconnect();
        } catch {}
        sourceRef.current = null;
      }
      if (analyserRef.current) {
        try {
          analyserRef.current.disconnect();
        } catch {}
        analyserRef.current = null;
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch(() => {});
        audioContextRef.current = null;
      }
      return;
    }

    // Initialize Web Audio Context and AnalyserNode
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    const audioCtx = new AudioContextClass();
    audioContextRef.current = audioCtx;

    if (audioCtx.state === 'suspended') {
      audioCtx.resume().catch(() => {});
    }

    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.85;
    analyserRef.current = analyser;

    try {
      const source = audioCtx.createMediaStreamSource(mediaStream);
      // Connect to analyser ONLY - DO NOT connect to destination to prevent speaker feedback!
      source.connect(analyser);
      sourceRef.current = source;
    } catch (err) {
      console.error('Failed to create MediaStreamSource:', err);
    }

    return () => {
      if (sourceRef.current) {
        try {
          sourceRef.current.disconnect();
        } catch {}
        sourceRef.current = null;
      }
      if (analyserRef.current) {
        try {
          analyserRef.current.disconnect();
        } catch {}
        analyserRef.current = null;
      }
      if (audioCtx && audioCtx.state !== 'closed') {
        audioCtx.close().catch(() => {});
      }
    };
  }, [mediaStream, isRecording]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const bufferLength = 1024;
    const dataArray = new Uint8Array(bufferLength);

    const render = () => {
      animationFrameRef.current = requestAnimationFrame(render);

      // HiDPI / Retina display scaling
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const targetWidth = Math.floor(rect.width * dpr);
      const targetHeight = Math.floor(rect.height * dpr);

      if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
        canvas.width = targetWidth;
        canvas.height = targetHeight;
      }

      ctx.save();
      ctx.scale(dpr, dpr);

      const width = rect.width;
      const h = rect.height;
      const centerY = h / 2;

      // Dark slate background
      ctx.fillStyle = '#0F172A';
      ctx.fillRect(0, 0, width, h);

      // Horizontal center line
      ctx.strokeStyle = '#1E293B';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();

      timeRef.current += 1;
      const t = timeRef.current;

      // State 1: Active Recording
      if (isRecording && !isPaused && analyserRef.current) {
        analyserRef.current.getByteTimeDomainData(dataArray);

        // Compute RMS level
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
          const val = (dataArray[i] - 128) / 128;
          sum += val * val;
        }
        const rms = Math.sqrt(sum / bufferLength);

        // Dynamic stroke styling based on speech energy
        const gradient = ctx.createLinearGradient(0, 0, width, 0);
        gradient.addColorStop(0, '#6366F1'); // Indigo-500
        gradient.addColorStop(0.5, rms > 0.05 ? '#10B981' : '#8B5CF6'); // Emerald when speaking, violet idle
        gradient.addColorStop(1, '#EC4899'); // Pink-500

        ctx.lineWidth = 2.5;
        ctx.strokeStyle = gradient;
        ctx.shadowColor = rms > 0.05 ? 'rgba(16, 185, 129, 0.4)' : 'rgba(99, 102, 241, 0.3)';
        ctx.shadowBlur = 8;
        ctx.beginPath();

        const sliceWidth = width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const v = dataArray[i] / 128.0; // 0..2
          const y = v * centerY;

          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
          x += sliceWidth;
        }

        ctx.lineTo(width, centerY);
        ctx.stroke();

        // Overlay small "RECORDING" indicator
        ctx.fillStyle = '#EF4444';
        ctx.beginPath();
        ctx.arc(16, 16, 5, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#F87171';
        ctx.font = '11px Inter, system-ui, sans-serif';
        ctx.fillText('REC 60FPS', 28, 20);
      }
      // State 2: Paused Recording
      else if (isRecording && isPaused) {
        ctx.strokeStyle = '#F59E0B';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(0, centerY);
        ctx.lineTo(width, centerY);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#F59E0B';
        ctx.font = '11px Inter, system-ui, sans-serif';
        ctx.fillText('PAUSED', 16, 20);
      }
      // State 3: Idle (Subtle Sinusoidal Breathing Wave)
      else {
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 1.5;
        ctx.beginPath();

        for (let x = 0; x < width; x += 2) {
          const wave1 = Math.sin(x * 0.015 + t * 0.02) * 5;
          const wave2 = Math.sin(x * 0.03 - t * 0.01) * 3;
          const y = centerY + (wave1 + wave2) * Math.sin(t * 0.01);

          if (x === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();

        ctx.fillStyle = '#64748B';
        ctx.font = '11px Inter, system-ui, sans-serif';
        ctx.fillText('STANDBY — Ready to Record', 16, 20);
      }

      ctx.restore();
    };

    render();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isRecording, isPaused]);

  return (
    <div className={`relative w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-900 shadow-inner ${className}`}>
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: `${height}px`, display: 'block' }}
      />
    </div>
  );
};
