'use client'

import Link from 'next/link'
import { useState, useEffect } from 'react'
import { 
  ArrowLeft, 
  Database, 
  Download, 
  Trash2, 
  FlaskConical,
  Atom,
  Activity,
  CheckCircle2,
  Clock,
  AlertCircle
} from 'lucide-react'

interface Dataset {
  id: string
  name: string
  molecules: number
  size: string
  created: string
  status: string
  format: string
  properties: string[]
  description?: string
  stats?: {
    avgAtoms: number
    elements: string[]
    energyRange: number[]
    avgEnergy: number
  }
  enrichment?: {
    method: string
    converged: number
    failed: number
    avgTime: string
  }
}

function StatBox({ label, value, unit }: { label: string; value: string | number; unit?: string }) {
  return (
    <div className="bg-white/5 rounded-2xl p-6 border border-white/10">
      <p className="text-sm text-gray-400 mb-1">{label}</p>
      <p className="text-2xl font-semibold text-white">
        {value}
        {unit && <span className="text-sm text-gray-400 ml-1">{unit}</span>}
      </p>
    </div>
  )
}

function PropertyBadge({ name }: { name: string }) {
  return (
    <span className="px-3 py-1.5 bg-blue-500/10 text-blue-400 rounded-full text-sm font-medium border border-blue-500/20">
      {name}
    </span>
  )
}

function ElementBadge({ element }: { element: string }) {
  const colors: Record<string, string> = {
    C: 'bg-gray-600',
    H: 'bg-white text-gray-900',
    O: 'bg-red-500',
    N: 'bg-blue-500',
    F: 'bg-green-500',
    S: 'bg-yellow-500',
    Mo: 'bg-purple-500',
    W: 'bg-indigo-500',
    Se: 'bg-orange-500',
    Te: 'bg-pink-500',
  }
  
  return (
    <span className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold ${colors[element] || 'bg-gray-500'}`}>
      {element}
    </span>
  )
}

export default function DatasetDetailPage({ params }: { params: { id: string } }) {
  const [dataset, setDataset] = useState<Dataset | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`/api/datasets/${params.id}`)
      .then(res => res.json())
      .then(data => {
        setDataset(data.dataset)
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

  if (!dataset) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-16 h-16 text-gray-600 mx-auto mb-4" />
          <h1 className="text-xl font-semibold text-white mb-2">Dataset Not Found</h1>
          <p className="text-gray-400 mb-6">The dataset you&apos;re looking for doesn&apos;t exist.</p>
          <Link 
            href="/datasets" 
            className="inline-flex items-center gap-2 px-6 py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-full transition-all"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Datasets
          </Link>
        </div>
      </div>
    )
  }

  const statusConfig = {
    enriched: { icon: CheckCircle2, color: 'text-emerald-400', bg: 'bg-emerald-400/10' },
    ready: { icon: Clock, color: 'text-blue-400', bg: 'bg-blue-400/10' },
    processing: { icon: Activity, color: 'text-amber-400', bg: 'bg-amber-400/10' },
  }

  const status = statusConfig[dataset.status as keyof typeof statusConfig] || statusConfig.ready

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-gray-950/80 backdrop-blur-xl border-b border-white/10">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link 
                href="/datasets" 
                className="p-2 rounded-full hover:bg-white/10 transition-colors"
              >
                <ArrowLeft className="w-5 h-5 text-gray-400" />
              </Link>
              <div>
                <div className="flex items-center gap-3">
                  <h1 className="text-xl font-semibold text-white">{dataset.name}</h1>
                  <span className={`px-3 py-1 rounded-full text-xs font-medium ${status.bg} ${status.color}`}>
                    {dataset.status}
                  </span>
                </div>
                <p className="text-sm text-gray-400 mt-0.5">Created {dataset.created}</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button className="p-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 transition-all">
                <Download className="w-5 h-5 text-gray-300" />
              </button>
              <button className="p-2.5 rounded-xl bg-white/5 hover:bg-red-500/20 border border-white/10 hover:border-red-500/30 transition-all group">
                <Trash2 className="w-5 h-5 text-gray-300 group-hover:text-red-400" />
              </button>
              <Link 
                href={`/compute?dataset=${dataset.name}`}
                className="px-5 py-2.5 bg-blue-500 hover:bg-blue-600 text-white rounded-xl font-medium transition-all flex items-center gap-2"
              >
                <FlaskConical className="w-4 h-4" />
                Enrich Dataset
              </Link>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8">
        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <StatBox label="Total Molecules" value={dataset.molecules.toLocaleString()} />
          <StatBox label="File Size" value={dataset.size} />
          <StatBox label="Format" value={dataset.format.toUpperCase()} />
          <StatBox label="Properties" value={dataset.properties.length} />
        </div>

        {/* Description */}
        {dataset.description && (
          <div className="bg-white/5 rounded-2xl p-6 border border-white/10 mb-8">
            <h2 className="text-lg font-medium text-white mb-3">Description</h2>
            <p className="text-gray-400 leading-relaxed">{dataset.description}</p>
          </div>
        )}

        <div className="grid md:grid-cols-2 gap-6">
          {/* Properties */}
          <div className="bg-white/5 rounded-2xl p-6 border border-white/10">
            <h2 className="text-lg font-medium text-white mb-4">Computed Properties</h2>
            <div className="flex flex-wrap gap-2">
              {dataset.properties.map(prop => (
                <PropertyBadge key={prop} name={prop} />
              ))}
              {dataset.properties.length === 0 && (
                <p className="text-gray-500 text-sm">No properties computed yet</p>
              )}
            </div>
          </div>

          {/* Elements */}
          {dataset.stats && (
            <div className="bg-white/5 rounded-2xl p-6 border border-white/10">
              <h2 className="text-lg font-medium text-white mb-4">Elements</h2>
              <div className="flex flex-wrap gap-3">
                {dataset.stats.elements.map(el => (
                  <ElementBadge key={el} element={el} />
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Statistics */}
        {dataset.stats && (
          <div className="mt-8">
            <h2 className="text-lg font-medium text-white mb-4">Statistics</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatBox label="Avg. Atoms per Molecule" value={dataset.stats.avgAtoms.toFixed(1)} />
              <StatBox label="Average Energy" value={dataset.stats.avgEnergy.toFixed(2)} unit="eV" />
              <StatBox label="Energy Min" value={dataset.stats.energyRange[0].toFixed(2)} unit="eV" />
              <StatBox label="Energy Max" value={dataset.stats.energyRange[1].toFixed(2)} unit="eV" />
            </div>
          </div>
        )}

        {/* Enrichment Info */}
        {dataset.enrichment && (
          <div className="mt-8 bg-gradient-to-br from-emerald-500/10 to-teal-500/10 rounded-2xl p-6 border border-emerald-500/20">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-emerald-500/20 rounded-lg">
                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              </div>
              <h2 className="text-lg font-medium text-white">Enrichment Complete</h2>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-sm text-gray-400">Method</p>
                <p className="text-white font-medium">{dataset.enrichment.method}</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Converged</p>
                <p className="text-emerald-400 font-medium">{dataset.enrichment.converged}</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Failed</p>
                <p className="text-red-400 font-medium">{dataset.enrichment.failed}</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Avg. Time</p>
                <p className="text-white font-medium">{dataset.enrichment.avgTime}</p>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
