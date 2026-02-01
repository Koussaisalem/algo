'use client'

import { useState, useEffect } from 'react'
import Link from "next/link"
import { 
  Atom, Upload, Cpu, FlaskConical, LineChart, Settings, Plus, Play, Clock, 
  CheckCircle, XCircle, Loader2, ChevronRight, Zap, Beaker, ArrowUpRight,
  X, Trash2
} from "lucide-react"
import { useAppStore } from '@/lib/store'
import { jobQueue, Job } from '@/lib/job-queue'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Input, Select } from '@/components/ui/dialog'

export default function ComputePage() {
  const { results } = useAppStore()
  const [jobs, setJobs] = useState<Job[]>([])
  const [newJobOpen, setNewJobOpen] = useState(false)
  const [selectedMethod, setSelectedMethod] = useState<'xtb' | 'dft'>('xtb')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [formData, setFormData] = useState({
    structure: '',
    config: {},
  })

  // Poll for job updates
  useEffect(() => {
    const updateJobs = () => setJobs(jobQueue.getAllJobs())
    updateJobs()
    const interval = setInterval(updateJobs, 1000)
    return () => clearInterval(interval)
  }, [])

  const runningJobs = jobs.filter(j => j.status === 'running' || j.status === 'queued')
  const recentJobs = jobs.filter(j => j.status === 'completed' || j.status === 'failed').slice(0, 5)

  const handleSubmit = async () => {
    if (!formData.structure) return
    
    setIsSubmitting(true)
    try {
      const job = jobQueue.createJob(selectedMethod, {
        structure: formData.structure,
        eta: selectedMethod === 'dft' ? '4h 30m' : '5m',
      })
      jobQueue.simulateJobProgress(job.id)
      setNewJobOpen(false)
      setFormData({ structure: '', config: {} })
    } finally {
      setIsSubmitting(false)
    }
  }

  const startQuickJob = (method: 'xtb' | 'dft') => {
    setSelectedMethod(method)
    setNewJobOpen(true)
  }

  return (
    <div className="min-h-screen bg-black">
      {/* Ambient background */}
      <div className="fixed inset-0 bg-gradient-to-br from-emerald-950/20 via-black to-purple-950/20" />
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
          <NavItem href="/compute" icon={<FlaskConical className="w-4 h-4" />} label="Compute" active />
          <NavItem href="/results" icon={<LineChart className="w-4 h-4" />} label="Results" />
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
              <span className="text-white">Compute</span>
            </div>
            <h1 className="text-4xl font-bold text-gradient">Compute</h1>
            <p className="text-neutral-400 mt-2">Run DFT and xTB calculations on your structures</p>
          </div>
          <button 
            onClick={() => setNewJobOpen(true)}
            className="btn-shine flex items-center gap-2 px-6 py-3 rounded-xl font-medium"
          >
            <Plus className="w-5 h-5" />
            New Calculation
          </button>
        </div>

        {/* Stats Row */}
        <div className="grid grid-cols-4 gap-4 mb-8">
          {[
            { label: 'Running', value: jobs.filter(j => j.status === 'running').length, color: 'blue', icon: Loader2 },
            { label: 'Queued', value: jobs.filter(j => j.status === 'queued').length, color: 'amber', icon: Clock },
            { label: 'Completed', value: jobs.filter(j => j.status === 'completed').length, color: 'emerald', icon: CheckCircle },
            { label: 'Results', value: results.length, color: 'purple', icon: LineChart },
          ].map((stat, i) => (
            <div key={i} className="glass rounded-2xl p-5 card-hover">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-neutral-400 text-sm">{stat.label}</p>
                  <p className="text-3xl font-bold text-white mt-1">{stat.value}</p>
                </div>
                <div className={`p-3 rounded-xl bg-${stat.color}-500/20`}>
                  <stat.icon className={`w-6 h-6 text-${stat.color}-400 ${stat.label === 'Running' && stat.value > 0 ? 'animate-spin' : ''}`} />
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Calculation Types */}
        <div className="grid grid-cols-2 gap-6 mb-8">
          <div className="glass rounded-2xl p-6 card-hover group">
            <div className="flex items-start justify-between mb-5">
              <div className="flex items-center gap-4">
                <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-emerald-500/20 to-emerald-600/20 flex items-center justify-center border border-emerald-500/20 group-hover:border-emerald-500/40 transition-colors">
                  <Zap className="w-7 h-7 text-emerald-400" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-white">xTB Calculation</h3>
                  <p className="text-sm text-neutral-500">Semi-empirical, fast screening</p>
                </div>
              </div>
              <span className="px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Fast</span>
            </div>
            <ul className="text-sm text-neutral-400 space-y-2 mb-5">
              <li className="flex items-center gap-2"><CheckCircle className="w-4 h-4 text-emerald-400" /> GFN2-xTB level of theory</li>
              <li className="flex items-center gap-2"><CheckCircle className="w-4 h-4 text-emerald-400" /> Geometry optimization</li>
              <li className="flex items-center gap-2"><CheckCircle className="w-4 h-4 text-emerald-400" /> Formation energy calculation</li>
              <li className="flex items-center gap-2"><Clock className="w-4 h-4 text-neutral-500" /> ~2 min per structure</li>
            </ul>
            <button 
              onClick={() => startQuickJob('xtb')}
              className="w-full px-4 py-3 rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 font-medium transition-all flex items-center justify-center gap-2"
            >
              <Play className="w-5 h-5" />
              Start xTB
            </button>
          </div>

          <div className="glass rounded-2xl p-6 card-hover group">
            <div className="flex items-start justify-between mb-5">
              <div className="flex items-center gap-4">
                <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-purple-500/20 to-purple-600/20 flex items-center justify-center border border-purple-500/20 group-hover:border-purple-500/40 transition-colors">
                  <Beaker className="w-7 h-7 text-purple-400" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-white">DFT Calculation</h3>
                  <p className="text-sm text-neutral-500">Full quantum chemistry</p>
                </div>
              </div>
              <span className="px-3 py-1 rounded-full text-xs font-medium bg-purple-500/20 text-purple-400 border border-purple-500/30">Accurate</span>
            </div>
            <ul className="text-sm text-neutral-400 space-y-2 mb-5">
              <li className="flex items-center gap-2"><CheckCircle className="w-4 h-4 text-purple-400" /> GPAW with PBE functional</li>
              <li className="flex items-center gap-2"><CheckCircle className="w-4 h-4 text-purple-400" /> Band structure calculation</li>
              <li className="flex items-center gap-2"><CheckCircle className="w-4 h-4 text-purple-400" /> Phonon dispersion</li>
              <li className="flex items-center gap-2"><Clock className="w-4 h-4 text-neutral-500" /> ~4 hrs per structure</li>
            </ul>
            <button 
              onClick={() => startQuickJob('dft')}
              className="w-full px-4 py-3 rounded-xl bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 font-medium transition-all flex items-center justify-center gap-2"
            >
              <Play className="w-5 h-5" />
              Start DFT
            </button>
          </div>
        </div>

        {/* Running Jobs */}
        <div className="glass rounded-2xl overflow-hidden mb-6">
          <div className="px-6 py-5 border-b border-white/10 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">Active Jobs</h2>
            {runningJobs.length > 0 && (
              <span className="px-3 py-1 rounded-full text-xs font-medium bg-blue-500/20 text-blue-400 border border-blue-500/30">
                {runningJobs.length} active
              </span>
            )}
          </div>
          <div className="divide-y divide-white/5">
            {runningJobs.map(job => (
              <JobRow
                key={job.id}
                id={job.id}
                structure={job.data?.structure || job.type}
                method={job.type.toUpperCase()}
                status={job.status as 'running' | 'queued'}
                progress={job.progress}
                eta={job.data?.eta || '--'}
                onCancel={() => jobQueue.deleteJob(job.id)}
              />
            ))}
          </div>
          
          {runningJobs.length === 0 && (
            <div className="py-12 text-center">
              <FlaskConical className="w-10 h-10 text-neutral-600 mx-auto mb-3" />
              <p className="text-neutral-400">No active calculations</p>
              <button 
                onClick={() => setNewJobOpen(true)}
                className="mt-3 text-blue-400 hover:text-blue-300 transition-colors text-sm"
              >
                Start a new calculation
              </button>
            </div>
          )}
        </div>

        {/* Recent Calculations */}
        {recentJobs.length > 0 && (
          <div className="glass rounded-2xl overflow-hidden">
            <div className="px-6 py-5 border-b border-white/10">
              <h2 className="text-lg font-semibold text-white">Recent Calculations</h2>
            </div>
            <div className="divide-y divide-white/5">
              {recentJobs.map(job => (
                <JobRow
                  key={job.id}
                  id={job.id}
                  structure={job.data?.structure || job.type}
                  method={job.type.toUpperCase()}
                  status={job.status as 'completed' | 'failed'}
                  progress={job.progress}
                  eta={job.status === 'completed' ? 'Completed' : 'Failed'}
                />
              ))}
            </div>
          </div>
        )}
      </main>

      {/* New Job Dialog */}
      <Dialog open={newJobOpen} onOpenChange={setNewJobOpen}>
        <DialogContent>
          <button 
            onClick={() => setNewJobOpen(false)}
            className="absolute top-4 right-4 p-2 rounded-lg hover:bg-white/10 
              text-neutral-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
          
          <DialogHeader>
            <DialogTitle>New Calculation</DialogTitle>
            <DialogDescription>
              Configure and submit a new {selectedMethod.toUpperCase()} calculation
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 mt-4">
            <div className="flex gap-3">
              <button
                onClick={() => setSelectedMethod('xtb')}
                className={`flex-1 p-4 rounded-xl border-2 transition-all flex items-center gap-3
                  ${selectedMethod === 'xtb' 
                    ? 'border-emerald-500 bg-emerald-500/10' 
                    : 'border-white/10 hover:border-white/20'}`}
              >
                <Zap className={`w-6 h-6 ${selectedMethod === 'xtb' ? 'text-emerald-400' : 'text-neutral-400'}`} />
                <div className="text-left">
                  <p className={`font-medium ${selectedMethod === 'xtb' ? 'text-white' : 'text-neutral-300'}`}>xTB</p>
                  <p className="text-xs text-neutral-500">~2 min</p>
                </div>
              </button>
              <button
                onClick={() => setSelectedMethod('dft')}
                className={`flex-1 p-4 rounded-xl border-2 transition-all flex items-center gap-3
                  ${selectedMethod === 'dft' 
                    ? 'border-purple-500 bg-purple-500/10' 
                    : 'border-white/10 hover:border-white/20'}`}
              >
                <Beaker className={`w-6 h-6 ${selectedMethod === 'dft' ? 'text-purple-400' : 'text-neutral-400'}`} />
                <div className="text-left">
                  <p className={`font-medium ${selectedMethod === 'dft' ? 'text-white' : 'text-neutral-300'}`}>DFT</p>
                  <p className="text-xs text-neutral-500">~4 hrs</p>
                </div>
              </button>
            </div>
            
            <Input
              label="Structure Name"
              placeholder="e.g., CrCuSe2_optimized"
              value={formData.structure}
              onChange={(e) => setFormData({ ...formData, structure: e.target.value })}
            />
            
            <Select
              label="Source"
              options={[
                { value: 'upload', label: 'Upload XYZ/CIF file' },
                { value: 'generated', label: 'From generated structures' },
                { value: 'dataset', label: 'From dataset' },
              ]}
            />
          </div>

          <div className="flex gap-3 mt-6 pt-4 border-t border-white/10">
            <button 
              onClick={() => setNewJobOpen(false)}
              className="flex-1 py-3 rounded-xl bg-white/5 text-white hover:bg-white/10 transition-colors font-medium"
            >
              Cancel
            </button>
            <button 
              onClick={handleSubmit}
              disabled={isSubmitting || !formData.structure}
              className={`flex-1 py-3 rounded-xl font-medium transition-colors
                flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed
                ${selectedMethod === 'xtb' 
                  ? 'bg-emerald-600 text-white hover:bg-emerald-500' 
                  : 'bg-purple-600 text-white hover:bg-purple-500'}`}
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Submitting...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Start {selectedMethod.toUpperCase()}
                </>
              )}
            </button>
          </div>
        </DialogContent>
      </Dialog>
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

function JobRow({ id, structure, method, status, progress, eta, onCancel }: {
  id: string
  structure: string
  method: string
  status: "running" | "queued" | "completed" | "failed"
  progress: number
  eta: string
  onCancel?: () => void
}) {
  const statusConfig = {
    running: { icon: Loader2, color: "text-blue-400", bg: "bg-gradient-to-r from-blue-500 to-purple-500", badge: "bg-blue-500/20 border-blue-500/30" },
    queued: { icon: Clock, color: "text-amber-400", bg: "bg-amber-500", badge: "bg-amber-500/20 border-amber-500/30" },
    completed: { icon: CheckCircle, color: "text-emerald-400", bg: "bg-emerald-500", badge: "bg-emerald-500/20 border-emerald-500/30" },
    failed: { icon: XCircle, color: "text-red-400", bg: "bg-red-500", badge: "bg-red-500/20 border-red-500/30" },
  }

  const config = statusConfig[status]
  const StatusIcon = config.icon

  return (
    <div className="px-6 py-5 flex items-center gap-6 hover:bg-white/5 transition-colors group">
      <div className="flex-1 min-w-0">
        <p className="font-medium text-white truncate">{structure}</p>
        <p className="text-sm text-neutral-500">{id} · {method}</p>
      </div>

      <div className="w-44">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-xs text-neutral-500">{progress}%</span>
          <span className="text-xs text-neutral-500">{eta}</span>
        </div>
        <div className="h-2 bg-white/10 rounded-full overflow-hidden">
          <div 
            className={`h-full ${config.bg} rounded-full transition-all duration-500`}
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${config.badge}`}>
        <StatusIcon className={`w-4 h-4 ${config.color} ${status === "running" ? "animate-spin" : ""}`} />
        <span className={`text-sm font-medium capitalize ${config.color}`}>{status}</span>
      </div>

      <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
        {status === 'completed' && (
          <Link 
            href={`/results/${id}`}
            className="px-3 py-1.5 rounded-lg bg-blue-500/20 text-blue-400 
              hover:bg-blue-500/30 transition-colors text-sm font-medium flex items-center gap-1.5"
          >
            View
            <ArrowUpRight className="w-3 h-3" />
          </Link>
        )}
        {(status === 'running' || status === 'queued') && onCancel && (
          <button 
            onClick={onCancel}
            className="p-2 rounded-lg hover:bg-red-500/20 text-neutral-400 
              hover:text-red-400 transition-colors"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  )
}
