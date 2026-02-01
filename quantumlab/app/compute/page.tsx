'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { 
  ArrowLeft, 
  Cpu, 
  HardDrive, 
  MemoryStick, 
  Monitor,
  Server,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  Info,
  Zap
} from 'lucide-react';

interface SystemSpecs {
  runtime: {
    type: string;
    os: string;
    os_version: string;
    hostname: string;
  };
  cpu: {
    name: string;
    cores_physical: number;
    cores_logical: number;
    frequency_mhz: number;
    usage_percent: number;
  };
  memory: {
    total_gb: number;
    available_gb: number;
    used_gb: number;
    percent_used: number;
  };
  gpus: Array<{
    name: string;
    type: string;
    memory_total_mb?: number;
    memory_free_mb?: number;
    memory_used_mb?: number;
    utilization_percent?: number;
    temperature_c?: number;
  }>;
  storage: Array<{
    device: string;
    mountpoint: string;
    total_gb: number;
    used_gb: number;
    free_gb: number;
    percent_used: number;
  }>;
  environment: {
    python_version: string;
    frameworks: Record<string, any>;
  };
}

export default function ComputePage() {
  const [specs, setSpecs] = useState<SystemSpecs | null>(null);
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchSpecs = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/system/specs');
      const data = await response.json();
      if (data.success) {
        setSpecs(data.specs);
        setRecommendations(data.recommendations);
        setLastUpdate(new Date());
      }
    } catch (error) {
      console.error('Failed to fetch system specs:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchSpecs();
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchSpecs, 30000);
    return () => clearInterval(interval);
  }, []);

  const getRuntimeBadgeColor = (type: string) => {
    const colors: Record<string, string> = {
      codespace: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
      docker: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
      aws: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
      gcp: 'bg-green-500/10 text-green-400 border-green-500/20',
      azure: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
      local: 'bg-gray-500/10 text-gray-400 border-gray-500/20',
    };
    return colors[type] || colors.local;
  };

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white">
      {/* Header */}
      <nav className="border-b border-white/10 bg-black/40 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/dashboard" className="p-2 hover:bg-white/5 rounded-lg transition-colors">
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <div>
              <h1 className="text-xl font-bold">System Specifications</h1>
              <p className="text-sm text-gray-400">
                Auto-detected runtime and hardware configuration
              </p>
            </div>
          </div>
          
          <button
            onClick={fetchSpecs}
            disabled={isLoading}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 rounded-lg transition-colors"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {isLoading && !specs ? (
          <div className="flex items-center justify-center py-20">
            <RefreshCw className="w-8 h-8 animate-spin text-blue-500" />
          </div>
        ) : specs ? (
          <div className="space-y-6">
            {/* Runtime Info */}
            <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-xl p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-3 bg-blue-500/10 rounded-lg">
                    <Server className="w-6 h-6 text-blue-400" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold">Runtime Environment</h2>
                    <p className="text-sm text-gray-400">Auto-detected configuration</p>
                  </div>
                </div>
                <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getRuntimeBadgeColor(specs.runtime.type)}`}>
                  {specs.runtime.type.toUpperCase()}
                </span>
              </div>
              
              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
                <InfoCard label="Operating System" value={specs.runtime.os} />
                <InfoCard label="OS Version" value={specs.runtime.os_version.split(' ')[0]} />
                <InfoCard label="Hostname" value={specs.runtime.hostname} />
                <InfoCard label="Python" value={specs.environment.python_version} />
              </div>

              {/* ML Frameworks */}
              {Object.keys(specs.environment.frameworks).length > 0 && (
                <div className="mt-4 pt-4 border-t border-white/10">
                  <p className="text-sm text-gray-400 mb-2">ML Frameworks</p>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(specs.environment.frameworks).map(([key, value]) => (
                      !['cuda_available', 'cudnn_version', 'cuda_version'].includes(key) && (
                        <span key={key} className="px-2 py-1 bg-white/5 rounded text-xs">
                          {key}: {typeof value === 'string' ? value : value.toString()}
                        </span>
                      )
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* GPU Info */}
            <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-xl p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-3 bg-purple-500/10 rounded-lg">
                  <Monitor className="w-6 h-6 text-purple-400" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold">GPU Status</h2>
                  <p className="text-sm text-gray-400">{specs.gpus.length} GPU(s) detected</p>
                </div>
              </div>

              {specs.gpus.length > 0 ? (
                <div className="space-y-4">
                  {specs.gpus.map((gpu, idx) => (
                    <div key={idx} className="p-4 bg-black/20 rounded-lg border border-white/5">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <h3 className="font-semibold">{gpu.name}</h3>
                          <p className="text-sm text-gray-400">{gpu.type}</p>
                        </div>
                        {gpu.memory_total_mb && (
                          <span className="text-sm text-gray-400">
                            {(gpu.memory_total_mb / 1024).toFixed(1)} GB
                          </span>
                        )}
                      </div>
                      
                      {gpu.utilization_percent !== undefined && (
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span>GPU Usage</span>
                            <span>{gpu.utilization_percent}%</span>
                          </div>
                          <div className="h-2 bg-black/40 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
                              style={{ width: `${gpu.utilization_percent}%` }}
                            />
                          </div>
                        </div>
                      )}
                      
                      {gpu.memory_total_mb && (
                        <div className="mt-3 grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <p className="text-gray-400">Memory Used</p>
                            <p className="font-medium">{(gpu.memory_used_mb! / 1024).toFixed(1)} GB</p>
                          </div>
                          <div>
                            <p className="text-gray-400">Memory Free</p>
                            <p className="font-medium">{(gpu.memory_free_mb! / 1024).toFixed(1)} GB</p>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-400">
                  <Monitor className="w-12 h-12 mx-auto mb-3 opacity-20" />
                  <p>No GPU detected</p>
                  <p className="text-sm mt-1">Training will use CPU only</p>
                </div>
              )}
            </div>

            {/* CPU & Memory */}
            <div className="grid md:grid-cols-2 gap-6">
              {/* CPU */}
              <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-3 bg-green-500/10 rounded-lg">
                    <Cpu className="w-6 h-6 text-green-400" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold">CPU</h2>
                    <p className="text-sm text-gray-400">{specs.cpu.name}</p>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between text-sm">
                    <span>CPU Usage</span>
                    <span>{specs.cpu.usage_percent.toFixed(1)}%</span>
                  </div>
                  <div className="h-2 bg-black/40 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-green-500 to-emerald-500"
                      style={{ width: `${specs.cpu.usage_percent}%` }}
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4 mt-4">
                    <InfoCard label="Physical Cores" value={specs.cpu.cores_physical.toString()} />
                    <InfoCard label="Logical Cores" value={specs.cpu.cores_logical.toString()} />
                  </div>
                </div>
              </div>

              {/* Memory */}
              <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-3 bg-cyan-500/10 rounded-lg">
                    <MemoryStick className="w-6 h-6 text-cyan-400" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold">Memory (RAM)</h2>
                    <p className="text-sm text-gray-400">{specs.memory.total_gb.toFixed(1)} GB Total</p>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between text-sm">
                    <span>Memory Usage</span>
                    <span>{specs.memory.percent_used.toFixed(1)}%</span>
                  </div>
                  <div className="h-2 bg-black/40 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-cyan-500 to-blue-500"
                      style={{ width: `${specs.memory.percent_used}%` }}
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4 mt-4">
                    <InfoCard label="Used" value={`${specs.memory.used_gb.toFixed(1)} GB`} />
                    <InfoCard label="Available" value={`${specs.memory.available_gb.toFixed(1)} GB`} />
                  </div>
                </div>
              </div>
            </div>

            {/* Storage */}
            <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-xl p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-3 bg-orange-500/10 rounded-lg">
                  <HardDrive className="w-6 h-6 text-orange-400" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold">Storage</h2>
                  <p className="text-sm text-gray-400">{specs.storage.length} device(s)</p>
                </div>
              </div>

              <div className="space-y-4">
                {specs.storage.slice(0, 5).map((disk, idx) => (
                  <div key={idx} className="p-4 bg-black/20 rounded-lg border border-white/5">
                    <div className="flex items-center justify-between mb-2">
                      <div>
                        <p className="font-medium">{disk.mountpoint}</p>
                        <p className="text-sm text-gray-400">{disk.device}</p>
                      </div>
                      <span className="text-sm text-gray-400">
                        {disk.free_gb.toFixed(1)} GB free
                      </span>
                    </div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Usage</span>
                      <span>{disk.percent_used.toFixed(1)}%</span>
                    </div>
                    <div className="h-2 bg-black/40 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-orange-500 to-yellow-500"
                        style={{ width: `${disk.percent_used}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Recommendations */}
            {recommendations.length > 0 && (
              <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <Zap className="w-5 h-5 text-yellow-400" />
                  <h2 className="text-lg font-semibold">System Recommendations</h2>
                </div>
                <div className="space-y-2">
                  {recommendations.map((rec, idx) => (
                    <div key={idx} className="flex items-start gap-3 p-3 bg-black/20 rounded-lg">
                      {rec.startsWith('✅') ? (
                        <CheckCircle2 className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                      ) : rec.startsWith('⚠️') ? (
                        <AlertTriangle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                      ) : (
                        <Info className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                      )}
                      <p className="text-sm">{rec.replace(/^[✅⚠️]\s/, '')}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Last Updated */}
            <div className="text-center text-sm text-gray-500">
              Last updated: {lastUpdate.toLocaleString()} • Auto-refreshes every 30s
            </div>
          </div>
        ) : (
          <div className="text-center py-20">
            <p className="text-gray-400">Failed to load system specifications</p>
            <button
              onClick={fetchSpecs}
              className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
            >
              Retry
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function InfoCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-3 bg-black/20 rounded-lg">
      <p className="text-xs text-gray-400 mb-1">{label}</p>
      <p className="font-medium truncate">{value}</p>
    </div>
  );
}
