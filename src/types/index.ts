// System & Settings Types
export interface GPUInfoResponse {
  cuda_available: boolean;
  gpu_name: string | null;
  vram_total_mb: number | null;
  vram_allocated_mb: number;
  cuda_version: string | null;
  simulation_mode: boolean;
  message: string | null;
  device_name?: string | null;
  hardware_gpu_name?: string | null;
  vram_free_mb?: number | null;
  warning_message?: string | null;
  torch_version?: string | null;
}

export type GPUInfo = GPUInfoResponse;

export interface HealthResponse {
  status: string;
  version: string;
  backend_ready: boolean;
  uptime_seconds: number;
  timestamp: string;
}

// Sample Types
export type SampleStatus = 'pending' | 'transcribing' | 'transcribed' | 'manual' | 'error';

export interface Sample {
  id: string;
  name: string;
  filename: string;
  file_path?: string | null;
  preprocessed_path?: string | null;
  duration: number; // in seconds
  duration_seconds: number;
  transcript: string;
  status: SampleStatus;
  transcript_status: SampleStatus | string;
  sample_rate: number;
  channels: number;
  created_at: string;
  updated_at?: string | null;
}

export type SampleItem = Sample;

export interface SampleListResponse {
  samples: Sample[];
}

export interface UpdateTranscriptRequest {
  transcript: string;
  status?: SampleStatus;
}

export interface DeleteSampleResponse {
  status: string;
  id: string;
  message?: string;
}

// Training Types
export interface TrainF5TTSRequest {
  model_name: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  sample_ids?: string[];
}

export interface TrainRVCRequest {
  model_name: string;
  epochs: number;
  batch_size: number;
  sample_rate: 32000 | 40000 | 48000;
  sample_ids?: string[];
}

export interface TrainStartResponse {
  run_id: string;
  status: string;
  engine: 'f5-tts' | 'rvc';
  model_name: string;
}

export interface StopTrainingResponse {
  status: 'stopped' | 'no_active_run';
  run_id?: string;
}

export interface TrainingRunRecord {
  id: string;
  engine: 'f5-tts' | 'rvc';
  model_name: string;
  status: 'queued' | 'running' | 'completed' | 'stopped' | 'failed';
  current_epoch: number;
  total_epochs: number;
  current_step: number;
  total_steps: number;
  current_loss: number | null;
  progress_percent: number;
  status_message: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  completed_at?: string | null;
}

export interface TrainingStatusResponse {
  active: boolean;
  run: TrainingRunRecord | null;
}

// WebSocket Telemetry Frames from /ws/training
export interface TelemetryProgressFrame {
  type: 'progress';
  engine: 'f5-tts' | 'rvc';
  run_id: string;
  epoch: number;
  total_epochs: number;
  step: number;
  total_steps: number;
  loss: number;
  progress_percent: number;
  message: string;
  status: 'training' | 'running';
  timestamp: string;
}

export interface TelemetryCompletedFrame {
  type: 'completed';
  engine: 'f5-tts' | 'rvc';
  run_id: string;
  status: 'completed';
  model_name: string;
  checkpoint_file: string;
  final_loss: number;
  timestamp: string;
}

export interface TelemetryStoppedFrame {
  type: 'stopped';
  engine: 'f5-tts' | 'rvc';
  run_id: string;
  status: 'stopped';
  message: string;
  timestamp: string;
}

export interface TelemetryErrorFrame {
  type: 'error';
  engine: 'f5-tts' | 'rvc';
  run_id: string;
  status: 'failed';
  error_message: string;
  timestamp: string;
}

export type TrainingTelemetryFrame =
  | TelemetryProgressFrame
  | TelemetryCompletedFrame
  | TelemetryStoppedFrame
  | TelemetryErrorFrame;

export type TrainingProgress = TelemetryProgressFrame;

// Model Registry Types
export interface RegisteredModel {
  id: string;
  name: string;
  engine: 'f5-tts' | 'rvc';
  checkpoint_path: string;
  index_path?: string | null;
  vocab_path?: string | null;
  config_path?: string | null;
  parameters: Record<string, any>;
  sample_count: number;
  created_at: string;
  updated_at?: string | null;
}

export type Model = RegisteredModel;

export interface ModelListResponse {
  models: RegisteredModel[];
}

export interface DeleteModelResponse {
  status: 'deleted';
  name: string;
  id: string;
}

// Generation Types
export interface GenerateTTSRequest {
  text: string;
  model_name: string;
  reference_sample_id?: string;
  speed?: number;
}

export interface GenerateSTSRequest {
  input_sample_id: string;
  model_name: string;
  pitch_shift?: number;
  index_rate?: number;
}

export interface GenerateResponse {
  generation_id: string;
  audio_url: string;
  filename: string;
  duration?: number;
  sample_rate?: number;
}

export interface GenerationMetadata {
  id: string;
  type: 'tts' | 'sts';
  model_id: string;
  input_text?: string | null;
  input_audio_path?: string | null;
  reference_sample_id?: string | null;
  output_audio_path: string;
  output_filename: string;
  duration_seconds: number;
  sample_rate: number;
  channels: number;
  parameters: Record<string, any>;
  created_at: string;
}

export type Generation = GenerationMetadata;

export interface GenerationListResponse {
  generations: GenerationMetadata[];
}

// WebSocket Notifications from /ws/notifications
export interface SampleCreatedNotification {
  type: 'notification';
  event: 'sample_created';
  sample_id: string;
  filename: string;
  duration: number;
  status: 'pending' | string;
  timestamp: string;
}

export interface TranscriptionStartedNotification {
  type: 'notification';
  event: 'transcription_started';
  sample_id: string;
  filename: string;
  status: 'transcribing';
  timestamp: string;
}

export interface TranscriptionCompletedNotification {
  type: 'notification';
  event: 'transcription_completed';
  sample_id: string;
  status: 'transcribed';
  transcript: string;
  duration: number;
  timestamp: string;
}

export interface TranscriptionFailedNotification {
  type: 'notification';
  event: 'transcription_failed';
  sample_id: string;
  status: 'error';
  error_message: string;
  timestamp: string;
}

export interface SampleUpdatedNotification {
  type: 'notification';
  event: 'sample_updated';
  sample_id: string;
  transcript: string;
  status: string;
  timestamp: string;
}

export interface SampleDeletedNotification {
  type: 'notification';
  event: 'sample_deleted';
  sample_id: string;
  status: 'deleted';
  timestamp: string;
}

export type NotificationEvent =
  | SampleCreatedNotification
  | TranscriptionStartedNotification
  | TranscriptionCompletedNotification
  | TranscriptionFailedNotification
  | SampleUpdatedNotification
  | SampleDeletedNotification;
