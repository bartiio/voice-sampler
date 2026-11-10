export interface BackendConfig {
  port: number;
  host: string;
  url: string;
}

export interface ElectronAPI {
  getBackendConfig: () => Promise<BackendConfig>;
  getBackendLogs: () => Promise<string[]>;
  selectFile: (options?: {
    title?: string;
    filters?: { name: string; extensions: string[] }[];
  }) => Promise<string[] | null>;
  openExternal: (url: string) => Promise<void>;
  platform: string;
  onBackendLog: (callback: (log: string) => void) => () => void;
}

declare global {
  interface Window {
    electronAPI?: ElectronAPI;
    VOICE_SAMPLER_CONFIG?: {
      apiUrl?: string;
      wsUrl?: string;
    };
  }
}

export {};
