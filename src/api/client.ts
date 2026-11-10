import {
  GPUInfoResponse,
  HealthResponse,
  Sample,
  SampleListResponse,
  UpdateTranscriptRequest,
  DeleteSampleResponse,
  TrainF5TTSRequest,
  TrainRVCRequest,
  TrainStartResponse,
  StopTrainingResponse,
  TrainingStatusResponse,
  ModelListResponse,
  DeleteModelResponse,
  GenerateTTSRequest,
  GenerateSTSRequest,
  GenerateResponse,
  GenerationListResponse,
} from '../types';

export let API_BASE_URL = window.VOICE_SAMPLER_CONFIG?.apiUrl || 'http://127.0.0.1:8000';

export function setApiBaseUrl(url: string) {
  API_BASE_URL = url;
}

// Automatically sync port from Electron if running in desktop shell
if (window.electronAPI) {
  window.electronAPI.getBackendConfig().then((cfg: { url?: string; port?: number; host?: string }) => {
    if (cfg?.url) {
      setApiBaseUrl(cfg.url);
    }
  }).catch(() => {});
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${path}`;
  const isFormData = options.body instanceof FormData;

  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string> || {}),
  };

  if (!isFormData && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }

  const response = await fetch(url, { ...options, headers });
  if (!response.ok) {
    let errorDetail = `HTTP ${response.status} ${response.statusText}`;
    try {
      const errJson = await response.json();
      errorDetail = errJson.detail || errorDetail;
    } catch {
      // fallback
    }
    throw new Error(errorDetail);
  }
  return response.json();
}

export const api = {
  // Health & Settings
  getHealth: () => request<HealthResponse>('/api/health'),
  getGpuSettings: (refresh = false) =>
    request<GPUInfoResponse>(`/api/settings/gpu${refresh ? '?refresh=true' : ''}`),

  // Samples
  getSamples: (statusFilter?: string) => {
    const params = statusFilter ? `?status=${encodeURIComponent(statusFilter)}` : '';
    return request<SampleListResponse>(`/api/samples${params}`);
  },

  uploadSample: async (file: File, onProgress?: (percent: number) => void): Promise<Sample> => {
    const formData = new FormData();
    formData.append('file', file, file.name);

    if (onProgress && typeof XMLHttpRequest !== 'undefined') {
      return new Promise<Sample>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', `${API_BASE_URL}/api/samples/upload`);

        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            const percent = Math.round((event.loaded / event.total) * 100);
            onProgress(percent);
          }
        };

        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              resolve(JSON.parse(xhr.responseText));
            } catch (e) {
              reject(e);
            }
          } else {
            let errorMsg = `Upload failed: ${xhr.status} ${xhr.statusText}`;
            try {
              const res = JSON.parse(xhr.responseText);
              if (res.detail) errorMsg = res.detail;
            } catch {}
            reject(new Error(errorMsg));
          }
        };

        xhr.onerror = () => reject(new Error('Network error during file upload'));
        xhr.send(formData);
      });
    }

    return request<Sample>('/api/samples/upload', {
      method: 'POST',
      body: formData,
    });
  },

  recordSample: async (blob: Blob, filename = `recording_${Date.now()}.webm`): Promise<Sample> => {
    const formData = new FormData();
    formData.append('file', blob, filename);

    return request<Sample>('/api/samples/record', {
      method: 'POST',
      body: formData,
    });
  },

  updateTranscript: async (sampleId: string, transcript: string, status = 'manual'): Promise<Sample> => {
    const payload: UpdateTranscriptRequest = {
      transcript: transcript.normalize('NFC'),
      status: status as any,
    };
    return request<Sample>(`/api/samples/${sampleId}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    });
  },

  retranscribeSample: (sampleId: string) =>
    request<Sample>(`/api/samples/${sampleId}/transcribe`, { method: 'POST' }),

  deleteSample: (sampleId: string) =>
    request<DeleteSampleResponse>(`/api/samples/${sampleId}`, { method: 'DELETE' }),

  deleteAllSamples: (onlyMissing = false) =>
    request<{ status: string; deleted_count: number; message: string }>(
      `/api/samples${onlyMissing ? '?only_missing=true' : ''}`,
      { method: 'DELETE' }
    ),

  getAudioUrl: (sampleId: string) => `${API_BASE_URL}/api/samples/${sampleId}/audio`,

  // Training
  trainF5TTS: (data: TrainF5TTSRequest) =>
    request<TrainStartResponse>('/api/train/f5-tts', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  trainRVC: (data: TrainRVCRequest) =>
    request<TrainStartResponse>('/api/train/rvc', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  stopTraining: () =>
    request<StopTrainingResponse>('/api/train/stop', { method: 'POST' }),

  getTrainingStatus: () =>
    request<TrainingStatusResponse>('/api/train/status'),

  // Model Registry
  getModels: (engine?: 'f5-tts' | 'rvc') =>
    request<ModelListResponse>(`/api/models${engine ? `?engine=${engine}` : ''}`),

  deleteModel: (name: string) =>
    request<DeleteModelResponse>(`/api/models/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    }),

  // Generation
  generateTTS: (data: GenerateTTSRequest) =>
    request<GenerateResponse>('/api/generate/tts', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  generateSTS: (data: GenerateSTSRequest) =>
    request<GenerateResponse>('/api/generate/sts', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  getGenerationHistory: (type?: 'tts' | 'sts', limit = 50) =>
    request<GenerationListResponse>(
      `/api/generate/history?limit=${limit}${type ? `&type=${type}` : ''}`
    ),

  deleteGeneration: (id: string) =>
    request<{ status: string; id: string }>(`/api/generate/${id}`, {
      method: 'DELETE',
    }),

  // Audio URLs helpers
  getGeneratedAudioUrl: (filename: string) =>
    `${API_BASE_URL}/api/audio/generated/${encodeURIComponent(filename)}`,

  getDownloadAudioUrl: (filename: string) =>
    `${API_BASE_URL}/api/audio/download/${encodeURIComponent(filename)}`,
};

// Backwards-compatible samplesApi helper
export const samplesApi = {
  listSamples: async (statusFilter?: string): Promise<Sample[]> => {
    const res = await api.getSamples(statusFilter);
    return res.samples;
  },
  uploadSample: api.uploadSample,
  recordSample: api.recordSample,
  updateTranscript: api.updateTranscript,
  deleteSample: api.deleteSample,
  getAudioUrl: api.getAudioUrl,
};
