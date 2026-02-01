'use client';

import { useEffect, useRef, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { Terminal } from 'xterm';
import { FitAddon } from 'xterm-addon-fit';
import { WebLinksAddon } from 'xterm-addon-web-links';
import 'xterm/css/xterm.css';
import { ArrowLeft, Maximize2, Minimize2, X } from 'lucide-react';

export default function SSHTerminalPage() {
  const params = useParams();
  const router = useRouter();
  const vmId = params.vmId as string;
  
  const terminalRef = useRef<HTMLDivElement>(null);
  const terminal = useRef<Terminal | null>(null);
  const websocket = useRef<WebSocket | null>(null);
  const fitAddon = useRef<FitAddon | null>(null);
  
  const [isConnected, setIsConnected] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [vmInfo, setVmInfo] = useState<any>(null);

  useEffect(() => {
    // Fetch VM info
    fetch(`http://localhost:8000/cloud/vms/${vmId}`)
      .then(res => res.json())
      .then(data => setVmInfo(data))
      .catch(err => console.error('Failed to fetch VM info:', err));
  }, [vmId]);

  useEffect(() => {
    if (!terminalRef.current) return;

    // Initialize terminal
    terminal.current = new Terminal({
      cursorBlink: true,
      fontSize: 14,
      fontFamily: 'Menlo, Monaco, "Courier New", monospace',
      theme: {
        background: '#1e1e1e',
        foreground: '#d4d4d4',
        cursor: '#ffffff',
        black: '#000000',
        red: '#cd3131',
        green: '#0dbc79',
        yellow: '#e5e510',
        blue: '#2472c8',
        magenta: '#bc3fbc',
        cyan: '#11a8cd',
        white: '#e5e5e5',
        brightBlack: '#666666',
        brightRed: '#f14c4c',
        brightGreen: '#23d18b',
        brightYellow: '#f5f543',
        brightBlue: '#3b8eea',
        brightMagenta: '#d670d6',
        brightCyan: '#29b8db',
        brightWhite: '#e5e5e5'
      },
      allowProposedApi: true
    });

    fitAddon.current = new FitAddon();
    terminal.current.loadAddon(fitAddon.current);
    terminal.current.loadAddon(new WebLinksAddon());

    terminal.current.open(terminalRef.current);
    fitAddon.current.fit();

    // Connect WebSocket
    const wsUrl = `ws://localhost:8000/cloud/ssh/${vmId}/terminal`;
    websocket.current = new WebSocket(wsUrl);

    websocket.current.onopen = () => {
      setIsConnected(true);
      terminal.current?.write('\r\n\x1b[1;32m● Connected to remote server\x1b[0m\r\n\r\n');
    };

    websocket.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      if (message.type === 'connected') {
        terminal.current?.write(`\x1b[1;36m${message.message}\x1b[0m\r\n\r\n`);
      } else if (message.type === 'output') {
        terminal.current?.write(message.data);
      } else if (message.error) {
        terminal.current?.write(`\r\n\x1b[1;31m✗ Error: ${message.error}\x1b[0m\r\n`);
        setConnectionError(message.error);
      }
    };

    websocket.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionError('Connection error occurred');
      terminal.current?.write('\r\n\x1b[1;31m✗ Connection error\x1b[0m\r\n');
    };

    websocket.current.onclose = () => {
      setIsConnected(false);
      terminal.current?.write('\r\n\x1b[1;33m● Connection closed\x1b[0m\r\n');
    };

    // Handle terminal input
    terminal.current.onData((data) => {
      if (websocket.current?.readyState === WebSocket.OPEN) {
        websocket.current.send(data);
      }
    });

    // Handle window resize
    const handleResize = () => {
      fitAddon.current?.fit();
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      websocket.current?.close();
      terminal.current?.dispose();
    };
  }, [vmId]);

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
    setTimeout(() => {
      fitAddon.current?.fit();
    }, 100);
  };

  const reconnect = () => {
    setConnectionError(null);
    websocket.current?.close();
    window.location.reload();
  };

  return (
    <div className={`${isFullscreen ? 'fixed inset-0 z-50' : 'min-h-screen'} bg-gray-900`}>
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-4">
          {!isFullscreen && (
            <button
              onClick={() => router.push('/cloud')}
              className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-gray-400" />
            </button>
          )}
          
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-lg font-semibold text-white">
                SSH Terminal
              </h1>
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="text-sm text-gray-400">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
            {vmInfo && (
              <p className="text-sm text-gray-400 mt-1">
                {vmInfo.name} • {vmInfo.provider} • {vmInfo.region}
              </p>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {connectionError && (
            <button
              onClick={reconnect}
              className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors"
            >
              Reconnect
            </button>
          )}
          
          <button
            onClick={toggleFullscreen}
            className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
            title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
          >
            {isFullscreen ? (
              <Minimize2 className="w-5 h-5 text-gray-400" />
            ) : (
              <Maximize2 className="w-5 h-5 text-gray-400" />
            )}
          </button>

          {isFullscreen && (
            <button
              onClick={() => setIsFullscreen(false)}
              className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
              title="Close"
            >
              <X className="w-5 h-5 text-gray-400" />
            </button>
          )}
        </div>
      </div>

      {/* Terminal */}
      <div 
        ref={terminalRef} 
        className={`${isFullscreen ? 'h-[calc(100vh-64px)]' : 'h-[calc(100vh-120px)]'} p-4`}
        style={{ backgroundColor: '#1e1e1e' }}
      />

      {/* Connection Error */}
      {connectionError && (
        <div className="fixed bottom-4 right-4 bg-red-500/10 border border-red-500 rounded-lg p-4 max-w-md">
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-full bg-red-500/20 flex items-center justify-center flex-shrink-0">
              <X className="w-4 h-4 text-red-500" />
            </div>
            <div className="flex-1">
              <h3 className="text-sm font-semibold text-red-500 mb-1">Connection Error</h3>
              <p className="text-sm text-red-400">{connectionError}</p>
            </div>
          </div>
        </div>
      )}

      {/* Keyboard shortcuts help */}
      {!isFullscreen && (
        <div className="fixed bottom-4 left-4 bg-gray-800/90 backdrop-blur-sm border border-gray-700 rounded-lg p-3">
          <div className="text-xs text-gray-400 space-y-1">
            <div><kbd className="bg-gray-700 px-1.5 py-0.5 rounded">Ctrl+C</kbd> Interrupt</div>
            <div><kbd className="bg-gray-700 px-1.5 py-0.5 rounded">Ctrl+D</kbd> Exit</div>
            <div><kbd className="bg-gray-700 px-1.5 py-0.5 rounded">Tab</kbd> Autocomplete</div>
          </div>
        </div>
      )}
    </div>
  );
}
