import { TrainingTelemetryFrame, NotificationEvent } from '../types';
import { API_BASE_URL } from './client';

type TelemetryListener = (frame: TrainingTelemetryFrame) => void;
type NotificationListener = (event: NotificationEvent) => void;
type ConnectionListener = (connected: boolean) => void;

export function deriveWsFromHttp(httpUrl: string, endpointPath: string): string {
  try {
    const parsed = new URL(httpUrl);
    const wsProto = parsed.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${wsProto}//${parsed.host}${endpointPath}`;
  } catch {
    return `ws://127.0.0.1:8000${endpointPath}`;
  }
}

let dynamicWsBase: string | null = null;
const registeredServices: ResilientWebSocketService<any>[] = [];

// Dynamically listen to Electron supervisor backend configuration if running in desktop shell
if (typeof window !== 'undefined' && window.electronAPI) {
  window.electronAPI
    .getBackendConfig()
    .then((cfg) => {
      let resolvedBase: string | null = null;
      if (cfg?.port) {
        const host = cfg.host || '127.0.0.1';
        resolvedBase = `ws://${host}:${cfg.port}`;
      } else if (cfg?.url) {
        try {
          const parsed = new URL(cfg.url);
          const wsProto = parsed.protocol === 'https:' ? 'wss:' : 'ws:';
          resolvedBase = `${wsProto}//${parsed.host}`;
        } catch {
          // fallback
        }
      }
      if (resolvedBase && resolvedBase !== dynamicWsBase) {
        dynamicWsBase = resolvedBase;
        registeredServices.forEach((svc) => svc.refreshUrl());
      }
    })
    .catch(() => {});
}

class ResilientWebSocketService<T> {
  private ws: WebSocket | null = null;
  private listeners: Set<(data: T) => void> = new Set();
  private connectionListeners: Set<ConnectionListener> = new Set();
  private reconnectTimeout: number | null = null;
  private pingInterval: number | null = null;
  private reconnectDelay = 1000;
  private isExplicitlyClosed = false;
  private endpointPath: string;

  constructor(endpointPath: string) {
    this.endpointPath = endpointPath;
    registeredServices.push(this);
  }

  public getWsUrl(): string {
    const customWs = window.VOICE_SAMPLER_CONFIG?.wsUrl;
    if (customWs) {
      return customWs;
    }
    if (dynamicWsBase) {
      return `${dynamicWsBase}${this.endpointPath}`;
    }
    if (API_BASE_URL) {
      return deriveWsFromHttp(API_BASE_URL, this.endpointPath);
    }
    return `ws://127.0.0.1:8000${this.endpointPath}`;
  }

  public refreshUrl(): void {
    if (this.isExplicitlyClosed) return;
    if (this.ws) {
      const currentUrl = this.ws.url;
      const targetUrl = this.getWsUrl();
      if (currentUrl !== targetUrl) {
        this.disconnect();
        this.connect();
      }
    }
  }

  public connect(customUrl?: string): void {
    const url = customUrl || this.getWsUrl();
    this.isExplicitlyClosed = false;

    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        this.reconnectDelay = 1000;
        this.notifyConnectionState(true);
        this.startHeartbeat();
      };

      this.ws.onmessage = (event) => {
        if (event.data === 'pong') return;
        try {
          const data: T = JSON.parse(event.data);
          this.listeners.forEach((listener) => listener(data));
        } catch {
          // Ignore invalid JSON frames
        }
      };

      this.ws.onclose = () => {
        this.stopHeartbeat();
        this.notifyConnectionState(false);
        if (!this.isExplicitlyClosed) {
          this.scheduleReconnect();
        }
      };

      this.ws.onerror = () => {
        if (this.ws) this.ws.close();
      };
    } catch {
      this.scheduleReconnect();
    }
  }

  public disconnect(): void {
    this.isExplicitlyClosed = true;
    this.stopHeartbeat();
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.notifyConnectionState(false);
  }

  public subscribe(listener: (data: T) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  public onConnectionChange(listener: ConnectionListener): () => void {
    this.connectionListeners.add(listener);
    listener(this.isConnected());
    return () => this.connectionListeners.delete(listener);
  }

  public isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.pingInterval = window.setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send('ping');
      }
    }, 25000);
  }

  private stopHeartbeat(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimeout) clearTimeout(this.reconnectTimeout);
    this.reconnectTimeout = window.setTimeout(() => {
      this.connect();
      this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, 10000);
    }, this.reconnectDelay);
  }

  private notifyConnectionState(connected: boolean): void {
    this.connectionListeners.forEach((l) => l(connected));
  }
}

export const trainingWebSocket = new ResilientWebSocketService<TrainingTelemetryFrame>('/ws/training');
export const notificationWebSocket = new ResilientWebSocketService<NotificationEvent>('/ws/notifications');
