'use client'

import { useState } from 'react'
import Link from "next/link"
import { 
  Atom, Upload, Cpu, FlaskConical, LineChart, Settings, Download, Eye, Filter,
  ChevronRight, Search, Sparkles, CheckCircle, XCircle, Star, Trash2, Clock
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { useAppStore } from '@/lib/store'

export default function ResultsPage() {
  const { results, removeResult } = useAppStore()
  const [searchQuery, setSearchQuery] = useState('')
  const [filter, setFilter] = useState<'all' | 'validated' | 'stable' | 'novel'>('all')

  const filteredResults = results.filter(r => {
    const matchesSearch = r.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          r.formula.toLowerCase().includes(searchQuery.toLowerCase())
    if (filter === 'all') return matchesSearch
    if (filter === 'novel') return matchesSearch && r.isNovel
    return matchesSearch && r.status === filter
  })

  const stats = {
    total: results.length,
    validated: results.filter(r => r.status === 'validated').length,
    stable: results.filter(r => r.status === 'stable').length,
    novel: results.filter(r => r.isNovel).length,
  }

  return (
    <div className="min-h-screen bg-black">
      {/* Ambient background */}
      <div className="fixed inset-0 bg-gradient-to-br from-purple-950/20 via-black to-emerald-950/20" />
      <div className="fixed inset-0 bg-grid opacity-30" />
      
      {/* Sidebar */}
      <aside className="fixed left-0 top-0 bottom-0 w-72 glass-subtle p-6 z-50">
        <Link href="/" className="flex items-center gap-3 mb-10 px-2">
          <div className="p-2 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600">
            <Atom className="w-5 h-5 text-white" />
          </div>
          <span className="text-lg font-semibold text-white tracking-tight">QuantumLab</span>
        </Link>

        <nav className="space-y-1">
          <NavItem href="/dashboard" icon={<LineChart className="w-4 h-4" />} label="Overview" />
          <NavItem href="/datasets" icon={<Upload className="w-4 h-4" />} label="Datasets" />
          <NavItem href="/models" icon={<Cpu className="w-4 h-4" />} label="Models" />
          <NavItem href="/compute" icon={<FlaskConical className="w-4 h-4" />} label="Compute" />
          <NavItem href="/results" icon={<LineChart className="w-4 h-4" />} label="Results" active />
        </nav>

        <div className="absolute bottom-6 left-6 right-6">
          <NavItem href="/settings" icon={<Settings className="w-4 h-4" />} label="Settings" />
        </div>
      </aside>

      {/* Main Content */}
      <main className="relative z-10 ml-72 px-10 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <div className="flex items-center gap-2 text-sm text-neutral-500 mb-2">
              <Link href="/dashboard" className="hover:text-white transition-colors">Dashboard</Link>
              <ChevronRight className="w-4 h-4" />
              <span className="text-white">Results</span>
            </div>
            <h1 className="text-4xl font-bold text-gradient">Results</h1>
            <p className="text-neutral-400 mt-2">View and analyze your discoveries</p>
          </div>
          <div className="flex items-center gap-3">
            <Button 
              variant="outline" 
              className="border-white/10 text-white hover:bg-white/10"
              onClick={() => {/* TODO: export functionality */}}
            >
              <Download className="w-4 h-4 mr-2" />
              Export All
            </Button>
          </div>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-4 gap-4 mb-8">
          {[
            { label: 'Total Structures', value: stats.total, color: 'blue', icon: Atom },
            { label: 'Validated', value: stats.validated, color: 'emerald', icon: CheckCircle },
            { label: 'Stable', value: stats.stable, color: 'cyan', icon: Sparkles },
            { label: 'Novel', value: stats.novel, color: 'purple', icon: Star },
          ].map((stat, i) => (
            <div key={i} className="glass rounded-2xl p-5 card-hover">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-neutral-400 text-sm">{stat.label}</p>
                  <p className="text-3xl font-bold text-white mt-1">{stat.value}</p>
                </div>
                <div className={`p-3 rounded-xl bg-${stat.color}-500/20`}>
                  <stat.icon className={`w-6 h-6 text-${stat.color}-400`} />
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Search & Filter */}
        <div className="flex items-center gap-4 mb-6">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-500" />
            <input 
              type="text"
              placeholder="Search structures..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-12 pr-4 py-3 rounded-xl bg-white/5 border border-white/10 
                text-white placeholder:text-neutral-500 focus:outline-none focus:border-blue-500/50
                focus:ring-2 focus:ring-blue-500/20 transition-all"
            />
          </div>
          
          <div className="flex gap-2">
            {(['all', 'validated', 'stable', 'novel'] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-4 py-2 rounded-xl text-sm font-medium transition-all
                  ${filter === f 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-white/5 text-neutral-400 hover:bg-white/10 hover:text-white'}`}
              >
                {f.charAt(0).toUpperCase() + f.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Results Grid */}
        <div className="grid grid-cols-3 gap-6">
          {filteredResults.map((result) => (
            <ResultCard
              key={result.id}
              id={result.id}
              name={result.name}
              formula={result.formula}
              energy={result.energy}
              bandgap={result.bandgap}
              status={result.status}
              isNovel={result.isNovel}
              onDelete={() => removeResult(result.id)}
            />
          ))}
        </div>
        
        {filteredResults.length === 0 && (
          <div className="glass rounded-2xl py-16 text-center">
            <Atom className="w-12 h-12 text-neutral-600 mx-auto mb-4" />
            <p className="text-neutral-400">No results found</p>
            <Link 
              href="/compute"
              className="mt-4 inline-block text-blue-400 hover:text-blue-300 transition-colors"
            >
              Run a computation to generate results
            </Link>
          </div>
        )}
      </main>
    </div>
  )
}

function NavItem({ href, icon, label, active }: { href: string; icon: React.ReactNode; label: string; active?: boolean }) {
  return (
    <Link href={href}>
      <div className={`flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all ${
        active 
          ? "bg-white/10 text-white" 
          : "text-gray-400 hover:text-white hover:bg-white/[0.05]"
      }`}>
        {icon}
        <span className="text-sm font-medium">{label}</span>
        {active && <ChevronRight className="w-3 h-3 ml-auto text-gray-500" />}
      </div>
    </Link>
  )
}

function ResultCard({ id, name, formula, energy, bandgap, status, isNovel, onDelete }: {
  id: string
  name: string
  formula: string
  energy: string
  bandgap: string
  status: "validated" | "stable" | "unstable" | "pending"
  isNovel?: boolean
  onDelete: () => void
}) {
  const statusStyles = {
    validated: { bg: "bg-emerald-500/20", text: "text-emerald-400", icon: CheckCircle },
    stable: { bg: "bg-blue-500/20", text: "text-blue-400", icon: Sparkles },
    unstable: { bg: "bg-red-500/20", text: "text-red-400", icon: XCircle },
    pending: { bg: "bg-amber-500/20", text: "text-amber-400", icon: Clock },
  }

  const StatusIcon = statusStyles[status].icon

  return (
    <div className="glass rounded-2xl p-6 card-hover group">
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h3 className="text-lg font-semibold text-white">{name}</h3>
            {isNovel && (
              <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-purple-500/20 text-purple-400 border border-purple-500/30">
                Novel
              </span>
            )}
          </div>
          <p className="text-sm text-neutral-400">{formula}</p>
        </div>
        <span className={`flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium capitalize 
          ${statusStyles[status].bg} ${statusStyles[status].text}`}>
          <StatusIcon className="w-3 h-3" />
          {status}
        </span>
      </div>

      {/* 3D Viewer Placeholder */}
      <div className="aspect-square rounded-xl bg-gradient-to-br from-blue-500/10 to-purple-500/10 border border-white/10 mb-4 flex items-center justify-center relative overflow-hidden">
        <div className="absolute inset-0 bg-grid opacity-50" />
        <Atom className="w-16 h-16 text-neutral-600 relative z-10" />
        <div className="absolute bottom-2 right-2 text-xs text-neutral-500">3D Preview</div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="p-3 rounded-lg bg-white/5">
          <p className="text-xs text-neutral-500 mb-1">Formation Energy</p>
          <p className="text-sm font-medium text-white">{energy}</p>
        </div>
        <div className="p-3 rounded-lg bg-white/5">
          <p className="text-xs text-neutral-500 mb-1">Band Gap</p>
          <p className="text-sm font-medium text-white">{bandgap}</p>
        </div>
      </div>

      <div className="flex gap-2">
        <Link href={`/results/${id}`} className="flex-1">
          <Button size="sm" className="w-full bg-blue-600 hover:bg-blue-500 text-white">
            <Eye className="w-4 h-4 mr-1" />
            View Details
          </Button>
        </Link>
        <Button 
          size="sm" 
          variant="outline" 
          className="border-white/10 text-white hover:bg-white/10"
        >
          <Download className="w-4 h-4" />
        </Button>
        <Button 
          size="sm" 
          variant="outline" 
          className="border-white/10 text-neutral-400 hover:bg-red-500/20 hover:text-red-400 hover:border-red-500/30"
          onClick={onDelete}
        >
          <Trash2 className="w-4 h-4" />
        </Button>
      </div>
    </div>
  )
}
