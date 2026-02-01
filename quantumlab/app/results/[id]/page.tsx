'use client'

import Link from 'next/link'
import { useState, useEffect } from 'react'
import { 
  ArrowLeft, 
  Download, 
  Share2,
  CheckCircle2,
  Sparkles,
  Atom,
  AlertCircle,
  Zap,
  Magnet,
  CircleDot,
  Box,
  Copy,
  ExternalLink
} from 'lucide-react'

interface Result {
  id: string
  name: string
  formula: string
  status: string
  isNovel: boolean
  created: string
  source: string
  properties: {
    formationEnergy: number
    bandgap: number
    magneticMoment: number
    bulkModulus: number
  }
  structure: {
    spaceGroup: string
    latticeParams: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number }
    atoms: { element: string; position: number[] }[]
  }
  validation: {
    xtb: { status: string; energy: number; forces: number }
    dft: { status: string; energy: number; forces: number }
    phonon: { status: string; imaginaryModes: number }
  }
}

function PropertyCard({ icon: Icon, label, value, unit }: { icon: any; label: string; value: number; unit: string }) {
  return (
    <div className="bg-white/5 rounded-2xl p-5 border border-white/10 hover:border-white/20 transition-all">
      <div className="flex items-center gap-3 mb-3">
        <div className="p-2 bg-blue-500/10 rounded-lg">
          <Icon className="w-4 h-4 text-blue-400" />
        </div>
        <span className="text-sm text-gray-400">{label}</span>
      </div>
      <p className="text-2xl font-semibold text-white">
        {value.toFixed(2)}
        <span className="text-sm text-gray-400 ml-1">{unit}</span>
      </p>
    </div>
  )
}

function ValidationBadge({ status }: { status: string }) {
  if (status === 'passed') {
    return (
      <span className="flex items-center gap-1.5 text-emerald-400">
        <CheckCircle2 className="w-4 h-4" />
        Passed
      </span>
    )
  }
  return (
    <span className="flex items-center gap-1.5 text-amber-400">
      <AlertCircle className="w-4 h-4" />
      Pending
    </span>
  )
}

function AtomBadge({ element }: { element: string }) {
  const colors: Record<string, string> = {
    Cr: 'bg-blue-500',
    Cu: 'bg-orange-500',
    Se: 'bg-green-500',
    Mo: 'bg-purple-500',
    S: 'bg-yellow-500',
    W: 'bg-indigo-500',
  }
  
  return (
    <span className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white ${colors[element] || 'bg-gray-500'}`}>
      {element}
    </span>
  )
}

function Molecule3DPlaceholder({ atoms }: { atoms: { element: string; position: number[] }[] }) {
  return (
    <div className="relative h-64 bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl overflow-hidden border border-white/10">
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="relative w-40 h-40">
          {atoms.slice(0, 8).map((atom, i) => {
            const angle = (i / 8) * Math.PI * 2
            const radius = 50 + (i % 2) * 20
            const x = Math.cos(angle) * radius + 80
            const y = Math.sin(angle) * radius + 80
            const colors: Record<string, string> = {
              Cr: '#3B82F6',
              Cu: '#F97316',
              Se: '#22C55E',
              Mo: '#8B5CF6',
              S: '#EAB308',
            }
            
            return (
              <div
                key={i}
                className="absolute w-6 h-6 rounded-full shadow-lg animate-pulse"
                style={{
                  left: x,
                  top: y,
                  backgroundColor: colors[atom.element] || '#6B7280',
                  animationDelay: `${i * 0.1}s`,
                  boxShadow: `0 0 20px ${colors[atom.element] || '#6B7280'}50`,
                }}
              />
            )
          })}
        </div>
      </div>
      <div className="absolute bottom-4 left-4 right-4 flex justify-between items-center">
        <span className="text-xs text-gray-400">Interactive 3D viewer</span>
        <button className="text-xs text-blue-400 hover:text-blue-300 transition-colors">
          Open fullscreen
        </button>
      </div>
    </div>
  )
}

export default function ResultDetailPage({ params }: { params: { id: string } }) {
  const [result, setResult] = useState<Result | null>(null)
  const [loading, setLoading] = useState(true)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    fetch(`/api/results/${params.id}`)
      .then(res => res.json())
      .then(data => {
        setResult(data.result)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [params.id])

  const copyFormula = () => {
    if (result) {
      navigator.clipboard.writeText(result.formula)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  if (!result) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-16 h-16 text-gray-600 mx-auto mb-4" />
          <h1 className="text-xl font-semibold text-white mb-2">Result Not Found</h1>
          <p className="text-gray-400 mb-6">The result you&apos;re looking for doesn&apos;t exist.</p>
          <Link 
            href="/results" 
            className="inline-flex items-center gap-2 px-6 py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-full transition-all"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Results
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-gray-950/80 backdrop-blur-xl border-b border-white/10">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link 
                href="/results" 
                className="p-2 rounded-full hover:bg-white/10 transition-colors"
              >
                <ArrowLeft className="w-5 h-5 text-gray-400" />
              </Link>
              <div>
                <div className="flex items-center gap-3">
                  <h1 className="text-xl font-semibold text-white">{result.name}</h1>
                  {result.isNovel && (
                    <span className="px-3 py-1 rounded-full text-xs font-medium bg-purple-500/20 text-purple-400 flex items-center gap-1.5">
                      <Sparkles className="w-3 h-3" />
                      Novel Discovery
                    </span>
                  )}
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/20 text-emerald-400">
                    {result.status}
                  </span>
                </div>
                <div className="flex items-center gap-2 mt-0.5">
                  <p className="text-sm text-gray-400">{result.formula}</p>
                  <button 
                    onClick={copyFormula}
                    className="p-1 rounded hover:bg-white/10 transition-colors"
                  >
                    <Copy className="w-3 h-3 text-gray-500" />
                  </button>
                  {copied && <span className="text-xs text-emerald-400">Copied!</span>}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button className="p-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 transition-all">
                <Share2 className="w-5 h-5 text-gray-300" />
              </button>
              <button className="p-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 transition-all">
                <Download className="w-5 h-5 text-gray-300" />
              </button>
              <a 
                href="#"
                className="px-5 py-2.5 bg-blue-500 hover:bg-blue-600 text-white rounded-xl font-medium transition-all flex items-center gap-2"
              >
                <ExternalLink className="w-4 h-4" />
                Open in Materials Project
              </a>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          {/* 3D Structure */}
          <div>
            <h2 className="text-lg font-medium text-white mb-4">Crystal Structure</h2>
            <Molecule3DPlaceholder atoms={result.structure.atoms} />
            <div className="mt-4 flex items-center gap-2">
              <span className="text-sm text-gray-400">Atoms:</span>
              <div className="flex gap-1">
                {[...new Set(result.structure.atoms.map(a => a.element))].map(el => (
                  <AtomBadge key={el} element={el} />
                ))}
              </div>
            </div>
          </div>

          {/* Properties */}
          <div>
            <h2 className="text-lg font-medium text-white mb-4">Computed Properties</h2>
            <div className="grid grid-cols-2 gap-4">
              <PropertyCard 
                icon={Zap} 
                label="Formation Energy" 
                value={result.properties.formationEnergy} 
                unit="eV/atom" 
              />
              <PropertyCard 
                icon={CircleDot} 
                label="Band Gap" 
                value={result.properties.bandgap} 
                unit="eV" 
              />
              <PropertyCard 
                icon={Magnet} 
                label="Magnetic Moment" 
                value={result.properties.magneticMoment} 
                unit="μB" 
              />
              <PropertyCard 
                icon={Box} 
                label="Bulk Modulus" 
                value={result.properties.bulkModulus} 
                unit="GPa" 
              />
            </div>
          </div>
        </div>

        {/* Lattice Parameters */}
        <div className="bg-white/5 rounded-2xl p-6 border border-white/10 mb-8">
          <h2 className="text-lg font-medium text-white mb-4">Lattice Parameters</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div>
              <p className="text-sm text-gray-400 mb-1">Space Group</p>
              <p className="text-xl font-semibold text-white">{result.structure.spaceGroup}</p>
            </div>
            <div>
              <p className="text-sm text-gray-400 mb-1">a / b / c</p>
              <p className="text-xl font-semibold text-white">
                {result.structure.latticeParams.a.toFixed(2)} / {result.structure.latticeParams.b.toFixed(2)} / {result.structure.latticeParams.c.toFixed(2)} Å
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-400 mb-1">α / β / γ</p>
              <p className="text-xl font-semibold text-white">
                {result.structure.latticeParams.alpha}° / {result.structure.latticeParams.beta}° / {result.structure.latticeParams.gamma}°
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-400 mb-1">Atoms</p>
              <p className="text-xl font-semibold text-white">{result.structure.atoms.length}</p>
            </div>
          </div>
        </div>

        {/* Validation */}
        <div className="bg-white/5 rounded-2xl p-6 border border-white/10">
          <h2 className="text-lg font-medium text-white mb-4">Validation Results</h2>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="p-4 bg-white/5 rounded-xl border border-white/10">
              <div className="flex justify-between items-center mb-3">
                <span className="text-gray-400">xTB Calculation</span>
                <ValidationBadge status={result.validation.xtb.status} />
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500">Energy</span>
                  <span className="text-white">{result.validation.xtb.energy.toFixed(3)} eV</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Max Force</span>
                  <span className="text-white">{result.validation.xtb.forces.toFixed(4)} eV/Å</span>
                </div>
              </div>
            </div>
            
            <div className="p-4 bg-white/5 rounded-xl border border-white/10">
              <div className="flex justify-between items-center mb-3">
                <span className="text-gray-400">DFT Calculation</span>
                <ValidationBadge status={result.validation.dft.status} />
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500">Energy</span>
                  <span className="text-white">{result.validation.dft.energy.toFixed(3)} eV</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Max Force</span>
                  <span className="text-white">{result.validation.dft.forces.toFixed(4)} eV/Å</span>
                </div>
              </div>
            </div>
            
            <div className="p-4 bg-white/5 rounded-xl border border-white/10">
              <div className="flex justify-between items-center mb-3">
                <span className="text-gray-400">Phonon Stability</span>
                <ValidationBadge status={result.validation.phonon.status} />
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500">Imaginary Modes</span>
                  <span className={result.validation.phonon.imaginaryModes === 0 ? 'text-emerald-400' : 'text-red-400'}>
                    {result.validation.phonon.imaginaryModes}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Status</span>
                  <span className="text-white">
                    {result.validation.phonon.imaginaryModes === 0 ? 'Dynamically Stable' : 'Unstable'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
