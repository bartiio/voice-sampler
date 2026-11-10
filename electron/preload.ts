import { contextBridge, ipcRenderer } from 'electron';

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

const electronAPI: ElectronAPI = {
  getBackendConfig: () => ipcRenderer.invoke('get-backend-config'),
  getBackendLogs: () => ipcRenderer.invoke('get-backend-logs'),
  selectFile: (options) => ipcRenderer.invoke('select-file', options),
  openExternal: (url) => ipcRenderer.invoke('open-external', url),
  platform: process.platform,
  onBackendLog: (callback: (log: string) => void) => {
    const subscription = (_event: Electron.IpcRendererEvent, log: string) => callback(log);
    ipcRenderer.on('backend-log', subscription);
    return () => {
      ipcRenderer.removeListener('backend-log', subscription);
    };
  },
};

contextBridge.exposeInMainWorld('electronAPI', electronAPI);
