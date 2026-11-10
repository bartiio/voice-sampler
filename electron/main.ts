import { app, BrowserWindow, dialog, ipcMain, session, shell } from 'electron';
import * as path from 'path';
import { ChildProcess, spawn } from 'child_process';
import * as net from 'net';
import treeKill from 'tree-kill';

// Application state
let mainWindow: BrowserWindow | null = null;
let backendProcess: ChildProcess | null = null;
let backendPort = 8000;
let isQuitting = false;
const logBuffer: string[] = [];
const MAX_LOG_BUFFER = 500;

function appendLog(line: string): void {
  const formatted = `[${new Date().toISOString()}] ${line}`;
  logBuffer.push(formatted);
  if (logBuffer.length > MAX_LOG_BUFFER) {
    logBuffer.shift();
  }
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('backend-log', formatted);
  }
}

/**
 * Scan for an available port starting from defaultPort.
 */
function findAvailablePort(startPort: number = 8000): Promise<number> {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.listen(startPort, '127.0.0.1', () => {
      const addr = server.address() as net.AddressInfo;
      const port = addr.port;
      server.close(() => resolve(port));
    });
    server.on('error', () => {
      resolve(findAvailablePort(startPort + 1));
    });
  });
}

/**
 * Polls GET /api/health until HTTP 200 is returned or timeout expires.
 */
async function waitForBackendHealth(
  port: number,
  timeoutMs: number = 15000,
  intervalMs: number = 300
): Promise<boolean> {
  const startTime = Date.now();
  const healthUrl = `http://127.0.0.1:${port}/api/health`;

  while (Date.now() - startTime < timeoutMs) {
    try {
      const response = await fetch(healthUrl, {
        method: 'GET',
        signal: AbortSignal.timeout(1000),
      });

      if (response.ok) {
        const body = (await response.json()) as { status?: string; backend_ready?: boolean };
        if (body.status === 'ok' && body.backend_ready) {
          appendLog(`Backend health probe OK on port ${port} (${Date.now() - startTime}ms)`);
          return true;
        }
      }
    } catch {
      // Backend is starting up, retry
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }

  return false;
}

/**
 * Spawns the FastAPI backend child process.
 */
async function spawnBackend(port: number): Promise<void> {
  backendPort = port;
  const projectRoot = path.resolve(__dirname, '..');
  const pythonExecutable =
    process.env.PYTHON_PATH || (process.platform === 'win32' ? 'python' : 'python3');
  const runServerScript = path.join(projectRoot, 'run_server.py');

  const args = [
    runServerScript,
    '--port',
    String(port),
    '--host',
    '127.0.0.1',
    '--parent-pid',
    String(process.pid),
  ];

  if (!app.isPackaged) {
    args.push('--dev');
  }

  appendLog(`Spawning backend: ${pythonExecutable} ${args.join(' ')}`);

  backendProcess = spawn(pythonExecutable, args, {
    cwd: projectRoot,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      VOICE_SAMPLER_PORT: String(port),
      VOICE_SAMPLER_HOST: '127.0.0.1',
    },
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  backendProcess.stdout?.on('data', (chunk: Buffer) => {
    const text = chunk.toString().trim();
    if (text) {
      console.log(`[Python stdout] ${text}`);
      appendLog(`[stdout] ${text}`);
    }
  });

  backendProcess.stderr?.on('data', (chunk: Buffer) => {
    const text = chunk.toString().trim();
    if (text) {
      console.warn(`[Python stderr] ${text}`);
      appendLog(`[stderr] ${text}`);
    }
  });

  backendProcess.on('exit', (code, signal) => {
    appendLog(`Backend process exited with code ${code}, signal ${signal}`);
    backendProcess = null;
  });

  backendProcess.on('error', (err) => {
    console.error('Failed to spawn backend process:', err);
    appendLog(`[ERROR] Spawn failure: ${err.message}`);
  });
}

/**
 * Clean shutdown handshake:
 * 1. POST /api/shutdown
 * 2. Wait up to 2 seconds for clean exit
 * 3. Fallback: tree-kill(pid, 'SIGKILL')
 */
async function stopBackend(): Promise<void> {
  if (!backendProcess || !backendProcess.pid) {
    return;
  }

  const pid = backendProcess.pid;
  appendLog(`Initiating clean shutdown for PID ${pid}...`);

  try {
    await fetch(`http://127.0.0.1:${backendPort}/api/shutdown`, {
      method: 'POST',
      signal: AbortSignal.timeout(1500),
    });
    appendLog('Shutdown request accepted by backend.');
  } catch (err) {
    appendLog(`Shutdown endpoint probe note: ${(err as Error).message}`);
  }

  // Wait up to 2 seconds for natural exit
  const exited = await new Promise<boolean>((resolve) => {
    if (!backendProcess || backendProcess.exitCode !== null) {
      return resolve(true);
    }
    const timer = setTimeout(() => resolve(false), 2000);
    backendProcess.once('exit', () => {
      clearTimeout(timer);
      resolve(true);
    });
  });

  if (!exited) {
    appendLog(`Process PID ${pid} did not exit in 2000ms. Executing tree-kill SIGKILL...`);
    await new Promise<void>((resolve) => {
      treeKill(pid, 'SIGKILL', (err) => {
        if (err) {
          console.error(`tree-kill error on PID ${pid}:`, err);
        } else {
          appendLog(`tree-kill successfully terminated PID ${pid} tree.`);
        }
        resolve();
      });
    });
  } else {
    appendLog(`Backend process PID ${pid} exited cleanly.`);
  }

  backendProcess = null;
}

/**
 * Creates the primary BrowserWindow.
 */
function createMainWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 1024,
    minHeight: 720,
    title: 'Voice Sampler',
    backgroundColor: '#0b0f19',
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
    },
  });

  // Enable microphone access for AudioContext / MediaRecorder
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    if (permission === 'media') {
      return callback(true);
    }
    callback(false);
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow?.show();
  });

  const devServerUrl = process.env.VITE_DEV_SERVER_URL || 'http://127.0.0.1:5173';

  if (!app.isPackaged && process.env.NODE_ENV !== 'production') {
    mainWindow.loadURL(devServerUrl);
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// IPC Handlers
ipcMain.handle('get-backend-config', () => {
  return {
    port: backendPort,
    host: '127.0.0.1',
    url: `http://127.0.0.1:${backendPort}`,
  };
});

ipcMain.handle('get-backend-logs', () => {
  return logBuffer;
});

ipcMain.handle('select-file', async (_event, options) => {
  if (!mainWindow) return null;
  const result = await dialog.showOpenDialog(mainWindow, options);
  return result.canceled ? null : result.filePaths;
});

ipcMain.handle('open-external', async (_event, url: string) => {
  if (url.startsWith('http://') || url.startsWith('https://')) {
    await shell.openExternal(url);
  }
});

// App lifecycle
app.whenReady().then(async () => {
  try {
    const port = await findAvailablePort(8000);
    await spawnBackend(port);

    const isHealthy = await waitForBackendHealth(port, 15000);
    if (!isHealthy) {
      dialog.showErrorBox(
        'Backend Initialization Error',
        `Failed to connect to Voice Sampler Python backend on port ${port} within 15 seconds.\nCheck logs for details.`
      );
    }

    createMainWindow();
  } catch (error) {
    dialog.showErrorBox('Startup Fatal Error', String(error));
    app.quit();
  }
});

app.on('before-quit', async (event) => {
  if (!isQuitting) {
    event.preventDefault();
    isQuitting = true;
    try {
      await stopBackend();
    } finally {
      app.quit();
    }
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
