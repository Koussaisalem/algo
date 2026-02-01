interface StatusIndicatorProps {
  status: 'online' | 'offline' | 'loading' | 'error' | 'warning';
  label?: string;
  size?: 'sm' | 'md' | 'lg';
  showPulse?: boolean;
}

const statusColors = {
  online: 'bg-emerald-400',
  offline: 'bg-neutral-400',
  loading: 'bg-blue-400',
  error: 'bg-red-400',
  warning: 'bg-amber-400',
};

const statusLabels = {
  online: 'Online',
  offline: 'Offline',
  loading: 'Loading',
  error: 'Error',
  warning: 'Warning',
};

const sizes = {
  sm: 'w-2 h-2',
  md: 'w-2.5 h-2.5',
  lg: 'w-3 h-3',
};

export function StatusIndicator({
  status,
  label,
  size = 'md',
  showPulse = true,
}: StatusIndicatorProps) {
  return (
    <div className="flex items-center gap-2">
      <div className="relative">
        <div
          className={`${sizes[size]} rounded-full ${statusColors[status]}`}
        />
        {showPulse && (status === 'online' || status === 'loading') && (
          <div
            className={`absolute inset-0 ${sizes[size]} rounded-full ${statusColors[status]} animate-ping opacity-75`}
          />
        )}
      </div>
      {label !== undefined && (
        <span className="text-sm text-neutral-400">
          {label || statusLabels[status]}
        </span>
      )}
    </div>
  );
}

interface ConnectionStatusProps {
  connected: boolean;
  latency?: number;
}

export function ConnectionStatus({ connected, latency }: ConnectionStatusProps) {
  return (
    <div className="flex items-center gap-3 px-3 py-2 rounded-lg bg-white/5 border border-white/10">
      <StatusIndicator
        status={connected ? 'online' : 'offline'}
        size="sm"
      />
      <div className="flex flex-col">
        <span className="text-sm text-white">
          {connected ? 'Connected' : 'Disconnected'}
        </span>
        {latency !== undefined && connected && (
          <span className="text-xs text-neutral-500">{latency}ms latency</span>
        )}
      </div>
    </div>
  );
}

interface SystemStatusProps {
  services: {
    name: string;
    status: 'online' | 'offline' | 'loading' | 'error';
  }[];
}

export function SystemStatus({ services }: SystemStatusProps) {
  const allOnline = services.every((s) => s.status === 'online');

  return (
    <div className="glass rounded-2xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">System Status</h3>
        <StatusIndicator
          status={allOnline ? 'online' : 'warning'}
          label={allOnline ? 'All systems operational' : 'Issues detected'}
        />
      </div>
      <div className="space-y-3">
        {services.map((service) => (
          <div
            key={service.name}
            className="flex items-center justify-between py-2"
          >
            <span className="text-sm text-neutral-400">{service.name}</span>
            <StatusIndicator status={service.status} size="sm" />
          </div>
        ))}
      </div>
    </div>
  );
}
