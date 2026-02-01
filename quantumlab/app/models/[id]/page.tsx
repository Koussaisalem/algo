'use client'

import Link from 'next/link'
import { useState, useEffect } from 'react'
import { 
  ArrowLeft, 
  Brain,
  Download, 
  Play,
  Pause,
  CheckCircle2,
  Clock,
  Activity,
  AlertCircle,
  TrendingDown
} from 'lucide-react'

interface Model {
  id: string
  name: string
  type: string
  dataset: string
  status: string
  created: string
  epochs: number
  currentEpoch?: number
  metrics: Record<string, number>
  config: Record<string, any>
  trainingHistory?: {
    trainLoss: number[]
    valLoss: number[]
  }
}

function MetricCard({ label, value, unit, trend }: { label: string; value: number; unit?: string; trend?: 'up' | 'down' }) {
  return (
    <div className="bg-white/5 rounded-2xl p-5 border border-white/10">
      <p className="text-sm text-gray-400 mb-1">{label}</p>
      <div className="flex items-end gap-2">
        <p className="text-2xl font-semibold text-white">
          {value.toFixed(4)}
        </p>
        {unit && <span className="text-sm text-gray-400 mb-1">{unit}</span>}
        {trend === 'down' && <TrendingDown className="w-4 h-4 text-emerald-400 mb-1" />}
      </div>
    </div>
  )
}

function ConfigRow({ label, value }: { label: string; value: any }) {
  return (
    <div className="flex justify-between py-3 border-b border-white/5 last:border-0">
      <span className="text-gray-400">{label}</span>
      <span className="text-white font-medium">{String(value)}</span>
    </div>
  )
}

function LossChart({ history }: { history: { trainLoss: number[]; valLoss: number[] } }) {
  const maxLoss = Math.max(...history.trainLoss, ...history.valLoss)
  
  return (
    <div className="h-48 flex items-end gap-1 px-4">
      {history.trainLoss.map((loss, i) => {
        const trainHeight = (loss / maxLoss) * 100
        const valHeight = (history.valLoss[i] / maxLoss) * 100
        
        return (
          <div key={i} className="flex-1 flex items-end gap-0.5">
            <div 
              className="flex-1 bg-blue-500/50 rounded-t transition-all hover:bg-blue-500"
              style={{ height: `${trainHeight}%` }}
              title={`Train: ${loss.toFixed(4)}`}
            />
            <div 
              className="flex-1 bg-purple-500/50 rounded-t transition-all hover:bg-purple-500"
              style={{ height: `${valHeight}%` }}
              title={`Val: ${history.valLoss[i].toFixed(4)}`}
            />
          </div>
        )
      })}
    </div>
  )
}

export default function ModelDetailPage({ params }: { params: { id: string } }) {
  const [model, setModel] = useState<Model | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`/api/models/${params.id}`)
      .then(res => res.json())
      .then(data => {
        setModel(data.model)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [params.id])

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  if (!model) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-16 h-16 text-gray-600 mx-auto mb-4" />
          <h1 className="text-xl font-semibold text-white mb-2">Model Not Found</h1>
          <p className="text-gray-400 mb-6">The model you&apos;re looking for doesn&apos;t exist.</p>
          <Link 
            href="/models" 
            className="inline-flex items-center gap-2 px-6 py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-full transition-all"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Models
          </Link>
        </div>
      </div>
    )
  }

  const statusConfig = {
    trained: { icon: CheckCircle2, color: 'text-emerald-400', bg: 'bg-emerald-400/10', label: 'Trained' },
    training: { icon: Activity, color: 'text-blue-400', bg: 'bg-blue-400/10', label: 'Training' },
    queued: { icon: Clock, color: 'text-amber-400', bg: 'bg-amber-400/10', label: 'Queued' },
  }

  const status = statusConfig[model.status as keyof typeof statusConfig] || statusConfig.queued
  const StatusIcon = status.icon

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-gray-950/80 backdrop-blur-xl border-b border-white/10">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link 
                href="/models" 
                className="p-2 rounded-full hover:bg-white/10 transition-colors"
              >
                <ArrowLeft className="w-5 h-5 text-gray-400" />
              </Link>
              <div>
                <div className="flex items-center gap-3">
                  <h1 className="text-xl font-semibold text-white">{model.name}</h1>
                  <span className={`px-3 py-1 rounded-full text-xs font-medium flex items-center gap-1.5 ${status.bg} ${status.color}`}>
                    <StatusIcon className="w-3 h-3" />
                    {status.label}
                  </span>
                </div>
                <p className="text-sm text-gray-400 mt-0.5">
                  {model.type === 'surrogate' ? 'Energy Surrogate Model' : 'Score Model'} · {model.dataset}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button className="p-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 transition-all">
                <Download className="w-5 h-5 text-gray-300" />
              </button>
              {model.status === 'training' ? (
                <button className="px-5 py-2.5 bg-amber-500 hover:bg-amber-600 text-white rounded-xl font-medium transition-all flex items-center gap-2">
                  <Pause className="w-4 h-4" />
                  Pause Training
                </button>
              ) : model.status === 'trained' ? (
                <Link 
                  href="/compute"
                  className="px-5 py-2.5 bg-blue-500 hover:bg-blue-600 text-white rounded-xl font-medium transition-all flex items-center gap-2"
                >
                  <Play className="w-4 h-4" />
                  Use for Generation
                </Link>
              ) : (
                <button className="px-5 py-2.5 bg-blue-500 hover:bg-blue-600 text-white rounded-xl font-medium transition-all flex items-center gap-2">
                  <Play className="w-4 h-4" />
                  Start Training
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8">
        {/* Training Progress */}
        {model.status === 'training' && model.currentEpoch && (
          <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-2xl p-6 border border-blue-500/20 mb-8">
            <div className="flex justify-between items-center mb-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-500/20 rounded-lg">
                  <Activity className="w-5 h-5 text-blue-400 animate-pulse" />
                </div>
                <div>
                  <h2 className="text-lg font-medium text-white">Training in Progress</h2>
                  <p className="text-sm text-gray-400">Epoch {model.currentEpoch} of {model.epochs}</p>
                </div>
              </div>
              <span className="text-2xl font-semibold text-white">
                {Math.round((model.currentEpoch / model.epochs) * 100)}%
              </span>
            </div>
            <div className="h-2 bg-white/10 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all duration-500"
                style={{ width: `${(model.currentEpoch / model.epochs) * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* Metrics */}
        <div className="mb-8">
          <h2 className="text-lg font-medium text-white mb-4">Metrics</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(model.metrics).map(([key, value]) => (
              <MetricCard 
                key={key} 
                label={key.toUpperCase()} 
                value={value}
                trend={key.includes('loss') ? 'down' : undefined}
              />
            ))}
          </div>
        </div>

        {/* Training History Chart */}
        {model.trainingHistory && (
          <div className="bg-white/5 rounded-2xl p-6 border border-white/10 mb-8">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-lg font-medium text-white">Training History</h2>
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-blue-500 rounded-full" />
                  <span className="text-sm text-gray-400">Train Loss</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-purple-500 rounded-full" />
                  <span className="text-sm text-gray-400">Val Loss</span>
                </div>
              </div>
            </div>
            <LossChart history={model.trainingHistory} />
            <div className="flex justify-between px-4 mt-2 text-xs text-gray-500">
              <span>Epoch 1</span>
              <span>Epoch {model.trainingHistory.trainLoss.length * 10}</span>
            </div>
          </div>
        )}

        <div className="grid md:grid-cols-2 gap-6">
          {/* Configuration */}
          <div className="bg-white/5 rounded-2xl p-6 border border-white/10">
            <h2 className="text-lg font-medium text-white mb-4">Configuration</h2>
            <div>
              {Object.entries(model.config).map(([key, value]) => (
                <ConfigRow 
                  key={key} 
                  label={key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())} 
                  value={value} 
                />
              ))}
            </div>
          </div>

          {/* Info */}
          <div className="bg-white/5 rounded-2xl p-6 border border-white/10">
            <h2 className="text-lg font-medium text-white mb-4">Information</h2>
            <div>
              <ConfigRow label="Model ID" value={model.id} />
              <ConfigRow label="Type" value={model.type === 'surrogate' ? 'Energy Surrogate' : 'Score Model'} />
              <ConfigRow label="Dataset" value={model.dataset} />
              <ConfigRow label="Created" value={model.created} />
              <ConfigRow label="Total Epochs" value={model.epochs} />
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
