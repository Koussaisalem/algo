'use client'

import { useEffect, useState } from 'react'
import Link from "next/link"
import { 
  Atom, Upload, Cpu, FlaskConical, LineChart, Settings, Plus, ArrowUpRight, 
  Clock, ChevronRight, CheckCircle2, Activity, Sparkles, TrendingUp, Command, Zap,
  Layers, Target, BarChart3
} from "lucide-react"
import { useAppStore } from '@/lib/store'
import { jobQueue, Job } from '@/lib/job-queue'
import { BarChart, DonutChart, ProgressRing } from '@/components/ui/charts'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tooltip } from '@/components/ui/tooltip'
import { Kbd } from '@/components/ui/kbd'

export default function DashboardPage() {
  const { datasets, models, results } = useAppStore()
  const [jobs, setJobs] = useState<Job[]>([])
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    const updateJobs = () => setJobs(jobQueue.getAllJobs())
    updateJobs()
    const interval = setInterval(updateJobs, 1000)
    return () => clearInterval(interval)
  }, [])

  const runningJobs = jobs.filter(j => j.status === 'running').length
  const trainingModels = models.filter(m => m.status === 'training').length
  
  // Chart data
  const activityData = [
    { label: 'Mon', value: 12 },
    { label: 'Tue', value: 19 },
    { label: 'Wed', value: 8 },
    { label: 'Thu', value: 25 },
    { label: 'Fri', value: 18 },
    { label: 'Sat', value: 7 },
    { label: 'Sun', value: 14 },
  ]

  const statusData = [
    { label: 'Completed', value: jobs.filter(j => j.status === 'completed').length + 5, color: '#10b981' },
    { label: 'Running', value: runningJobs + 2, color: '#3b82f6' },
    { label: 'Queued', value: jobs.filter(j => j.status === 'queued').length + 1, color: '#f59e0b' },
  ]

  if (!mounted) return null

  return (
    <div className="min-h-screen bg-black">
      {/* Ambient background */}
      <div className="fixed inset-0 bg-gradient-to-br from-blue-950/20 via-black to-purple-950/20" />
      <div className="fixed inset-0 bg-grid opacity-30" />
      
      {/* Sidebar */}
      <aside className="fixed left-0 top-0 bottom-0 w-72 glass-subtle p-6 z-50">
        <Link href="/" className="flex items-center gap-3 mb-10 px-2">
          <div className="p-2 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 shadow-lg shadow-blue-500/20">
            <Atom className="w-5 h-5 text-white" />
          </div>
          <span className="text-lg font-semibold text-white tracking-tight">QuantumLab</span>
        </Link>

        {/* Command Palette Hint */}
        <button 
          onClick={() => {
            const event = new KeyboardEvent('keydown', { key: 'k', metaKey: true })
            window.dispatchEvent(event)
          }}
          className="w-full mb-6 flex items-center gap-3 px-3 py-2.5 rounded-xl bg-white/5 border border-white/10 text-neutral-400 hover:text-white hover:bg-white/10 transition-all group"
        >
          <Command className="w-4 h-4" />
          <span className="text-sm">Search...</span>
          <div className="ml-auto flex items-center gap-1">
            <Kbd>⌘</Kbd>
            <Kbd>K</Kbd>
          </div>
        </button>

        <nav className="space-y-1">
          <NavItem href="/dashboard" icon={<BarChart3 className="w-4 h-4" />} label="Overview" active />
          <NavItem href="/datasets" icon={<Layers className="w-4 h-4" />} label="Datasets" badge={datasets.length.toString()} />
          <NavItem href="/models" icon={<Cpu className="w-4 h-4" />} label="Models" badge={models.length.toString()} />
          <NavItem href="/compute" icon={<Zap className="w-4 h-4" />} label="Compute" badge={runningJobs > 0 ? `${runningJobs}` : undefined} />
          <NavItem href="/results" icon={<Target className="w-4 h-4" />} label="Results" badge={results.length.toString()} />
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
            <h1 className="text-4xl font-bold text-gradient">Dashboard</h1>
            <p className="text-neutral-400 mt-2">Overview of your materials discovery workspace</p>
          </div>
          <Link 
            href="/datasets"
            className="btn-shine flex items-center gap-2 px-6 py-3 rounded-xl font-medium"
          >
            <Plus className="w-5 h-5" />
            New Project
          </Link>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-4 gap-4 mb-8">
          <StatCard 
            label="Datasets" 
            value={datasets.length.toString()} 
            change={`${datasets.filter(d => d.status === 'ready').length} ready`} 
            color="blue" 
            icon={Upload}
            href="/datasets"
          />
          <StatCard 
            label="Models" 
            value={models.length.toString()} 
            change={trainingModels > 0 ? `${trainingModels} training` : 'All trained'} 
            color="purple" 
            icon={Cpu}
            href="/models"
          />
          <StatCard 
            label="Computations" 
            value={jobs.length.toString()} 
            change={runningJobs > 0 ? `${runningJobs} running` : 'None running'} 
            color="emerald" 
            icon={FlaskConical}
            href="/compute"
          />
          <StatCard 
            label="Discoveries" 
            value={results.length.toString()} 
            change={`${results.filter(r => r.status === 'validated').length} validated`} 
            color="amber" 
            icon={Sparkles}
            href="/results"
          />
        </div>

        {/* Recent Activity */}
        <div className="grid grid-cols-3 gap-6">
          <div className="col-span-2">
            <div className="glass rounded-2xl p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold text-white">Recent Activity</h2>
                <span className="text-sm text-neutral-500">Live updates</span>
              </div>

              <div className="space-y-1">
                {jobs.slice(0, 2).map(job => (
                  <ActivityItem
                    key={job.id}
                    title={`${job.type.toUpperCase()} ${job.status === 'running' ? 'in progress' : job.status}`}
                    description={job.data?.structure || `Job ${job.id}`}
                    time={job.status === 'running' ? `${job.progress}% complete` : job.updatedAt.toLocaleTimeString()}
                    type={job.status === 'completed' ? 'success' : job.status === 'running' ? 'running' : 'error'}
                  />
                ))}
                {models.filter(m => m.status === 'training').slice(0, 1).map(model => (
                  <ActivityItem
                    key={model.id}
                    title="Model training"
                    description={`${model.name} - ${model.metrics.epoch || 'Starting...'}`}
                    time="In progress"
                    type="running"
                  />
                ))}
                {results.slice(0, 2).map(result => (
                  <ActivityItem
                    key={result.id}
                    title={`Structure ${result.status}`}
                    description={`${result.name} - ${result.formula}`}
                    time={result.created}
                    type="success"
                  />
                ))}
                {jobs.length === 0 && models.filter(m => m.status === 'training').length === 0 && results.length === 0 && (
                  <div className="py-8 text-center text-neutral-500">
                    <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p>No recent activity</p>
                    <p className="text-sm mt-1">Start by uploading a dataset or running a computation</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="space-y-5">
            <div className="glass rounded-2xl p-6">
              <h2 className="text-lg font-semibold text-white mb-4">Quick Actions</h2>
              <div className="space-y-1">
                <QuickAction href="/datasets" label="Upload Dataset" icon={<Upload className="w-4 h-4" />} />
                <QuickAction href="/models" label="Train Model" icon={<Cpu className="w-4 h-4" />} />
                <QuickAction href="/compute" label="Run Computation" icon={<FlaskConical className="w-4 h-4" />} />
                <QuickAction href="/results" label="View Results" icon={<Sparkles className="w-4 h-4" />} />
              </div>
            </div>

            <div className="glass rounded-2xl p-6">
              <h2 className="text-lg font-semibold text-white mb-4">System Status</h2>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-neutral-400">Backend</span>
                  <span className="flex items-center gap-1.5 text-sm text-emerald-400">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                    Connected
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-neutral-400">GPU</span>
                  <span className="flex items-center gap-1.5 text-sm text-emerald-400">
                    <span className="w-2 h-2 rounded-full bg-emerald-400" />
                    Available
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-neutral-400">Jobs Queue</span>
                  <span className="text-sm text-neutral-300">{jobs.filter(j => j.status === 'queued').length} pending</span>
                </div>
              </div>
            </div>

            <div className="glass rounded-2xl p-6">
              <h2 className="text-lg font-semibold text-white mb-4">Resources</h2>
              <div className="space-y-3">
                <ResourceLink label="Documentation" href="/docs" />
                <ResourceLink label="API Reference" href="/docs" />
                <ResourceLink label="Tutorials" href="/docs" />
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

function NavItem({ href, icon, label, active, badge }: { href: string; icon: React.ReactNode; label: string; active?: boolean; badge?: string }) {
  return (
    <Link href={href}>
      <div className={`flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all ${
        active 
          ? "bg-white/10 text-white" 
          : "text-gray-400 hover:text-white hover:bg-white/[0.05]"
      }`}>
        {icon}
        <span className="text-sm font-medium">{label}</span>
        {badge && (
          <span className="ml-auto px-2 py-0.5 rounded-full text-xs bg-white/10 text-neutral-400">
            {badge}
          </span>
        )}
        {active && !badge && <ChevronRight className="w-3 h-3 ml-auto text-gray-500" />}
      </div>
    </Link>
  )
}

function StatCard({ label, value, change, color, icon: Icon, href }: { 
  label: string; value: string; change: string; color: string; icon: React.ElementType; href: string 
}) {
  const colors: Record<string, { bg: string; icon: string; text: string }> = {
    blue: { bg: "from-blue-500/20 to-blue-600/10 border-blue-500/30", icon: "bg-blue-500/20 text-blue-400", text: "text-blue-400" },
    purple: { bg: "from-purple-500/20 to-purple-600/10 border-purple-500/30", icon: "bg-purple-500/20 text-purple-400", text: "text-purple-400" },
    emerald: { bg: "from-emerald-500/20 to-emerald-600/10 border-emerald-500/30", icon: "bg-emerald-500/20 text-emerald-400", text: "text-emerald-400" },
    amber: { bg: "from-amber-500/20 to-amber-600/10 border-amber-500/30", icon: "bg-amber-500/20 text-amber-400", text: "text-amber-400" },
  }
  
  return (
    <Link href={href} className="block">
      <div className={`rounded-2xl p-5 border bg-gradient-to-br ${colors[color].bg} card-hover group`}>
        <div className="flex items-start justify-between mb-3">
          <p className="text-sm text-neutral-400">{label}</p>
          <div className={`p-2 rounded-lg ${colors[color].icon}`}>
            <Icon className="w-4 h-4" />
          </div>
        </div>
        <p className="text-3xl font-bold text-white mb-1">{value}</p>
        <div className="flex items-center gap-1 text-xs text-neutral-500">
          <TrendingUp className="w-3 h-3" />
          {change}
        </div>
        <ArrowUpRight className={`w-4 h-4 mt-2 ${colors[color].text} opacity-0 group-hover:opacity-100 transition-opacity`} />
      </div>
    </Link>
  )
}

function ActivityItem({ title, description, time, type }: { title: string; description: string; time: string; type: "success" | "running" | "error" }) {
  const icons = {
    success: <CheckCircle2 className="w-4 h-4 text-emerald-400" />,
    running: <Activity className="w-4 h-4 text-blue-400 animate-pulse" />,
    error: <Activity className="w-4 h-4 text-red-400" />,
  }
  
  return (
    <div className="flex items-start gap-4 py-4 border-b border-white/5 last:border-0">
      <div className="mt-0.5">{icons[type]}</div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-white truncate">{title}</p>
        <p className="text-sm text-neutral-500 truncate">{description}</p>
      </div>
      <div className="flex items-center gap-1.5 text-xs text-neutral-500 shrink-0">
        <Clock className="w-3 h-3" />
        {time}
      </div>
    </div>
  )
}

function QuickAction({ href, label, icon }: { href: string; label: string; icon: React.ReactNode }) {
  return (
    <Link href={href}>
      <div className="flex items-center justify-between px-3 py-2.5 rounded-xl text-neutral-400 hover:text-white hover:bg-white/5 transition-all group">
        <div className="flex items-center gap-3">
          <span className="text-neutral-500 group-hover:text-neutral-300 transition-colors">{icon}</span>
          <span className="text-sm">{label}</span>
        </div>
        <ArrowUpRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
      </div>
    </Link>
  )
}

function ResourceLink({ label, href }: { label: string; href: string }) {
  return (
    <Link href={href} className="block text-sm text-neutral-500 hover:text-white transition-colors">
      {label}
    </Link>
  )
}
