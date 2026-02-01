'use client'

import { useState } from 'react'
import Link from "next/link"
import { 
  Atom, Upload, Cpu, FlaskConical, LineChart, Settings, Plus, Clock, 
  CheckCircle, Play, ChevronRight, Trash2, X, Brain, Zap, Loader2
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { useAppStore } from '@/lib/store'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Input, Select } from '@/components/ui/dialog'
import { apiPost } from '@/lib/hooks'

export default function ModelsPage() {
  const { models, datasets, addModel, removeModel } = useAppStore()
  const [trainOpen, setTrainOpen] = useState(false)
  const [isTraining, setIsTraining] = useState(false)
  const [formData, setFormData] = useState({
    name: '',
    type: 'surrogate',
    dataset: '',
    epochs: 100,
    learningRate: 0.001,
    batchSize: 32,
  })

  const surrogateModels = models.filter(m => m.type === 'surrogate')
  const scoreModels = models.filter(m => m.type === 'score')

  const handleTrain = async () => {
    if (!formData.name || !formData.dataset) return
    
    setIsTraining(true)
    try {
      await apiPost('/api/train', formData)
      addModel({
        id: `model-${Date.now()}`,
        name: formData.name,
        type: formData.type as 'surrogate' | 'score',
        status: 'training',
        created: new Date().toISOString().split('T')[0],
        dataset: formData.dataset,
        metrics: { epoch: '0/' + formData.epochs, loss: '--' },
      })
      setTrainOpen(false)
      setFormData({ name: '', type: 'surrogate', dataset: '', epochs: 100, learningRate: 0.001, batchSize: 32 })
    } catch (err) {
      console.error('Training error:', err)
    } finally {
      setIsTraining(false)
    }
  }

  return (
    <div className="min-h-screen bg-black">
      {/* Ambient background */}
      <div className="fixed inset-0 bg-gradient-to-br from-purple-950/20 via-black to-blue-950/20" />
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
          <NavItem href="/models" icon={<Cpu className="w-4 h-4" />} label="Models" active />
          <NavItem href="/compute" icon={<FlaskConical className="w-4 h-4" />} label="Compute" />
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
              <span className="text-white">Models</span>
            </div>
            <h1 className="text-4xl font-bold text-gradient">Models</h1>
            <p className="text-neutral-400 mt-2">Train and manage your machine learning models</p>
          </div>
          <button 
            onClick={() => setTrainOpen(true)}
            className="btn-shine flex items-center gap-2 px-6 py-3 rounded-xl font-medium"
          >
            <Plus className="w-5 h-5" />
            Train Model
          </button>
        </div>

        {/* Model Types */}
        <div className="grid grid-cols-2 gap-6 mb-8">
          <div className="glass rounded-2xl p-6 card-hover">
            <div className="flex items-center gap-4 mb-4">
              <div className="p-3 rounded-xl bg-emerald-500/20">
                <Zap className="w-6 h-6 text-emerald-400" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">Surrogate Models</h3>
                <p className="text-sm text-neutral-400">GNN-based energy prediction</p>
              </div>
            </div>
            <div className="flex items-center justify-between mt-4 pt-4 border-t border-white/10">
              <span className="text-3xl font-bold text-white">{surrogateModels.length}</span>
              <div className="flex gap-2">
                <span className="px-2 py-1 text-xs rounded-full bg-emerald-500/20 text-emerald-400">
                  {surrogateModels.filter(m => m.status === 'trained').length} trained
                </span>
                {surrogateModels.filter(m => m.status === 'training').length > 0 && (
                  <span className="px-2 py-1 text-xs rounded-full bg-blue-500/20 text-blue-400">
                    {surrogateModels.filter(m => m.status === 'training').length} training
                  </span>
                )}
              </div>
            </div>
          </div>
          
          <div className="glass rounded-2xl p-6 card-hover">
            <div className="flex items-center gap-4 mb-4">
              <div className="p-3 rounded-xl bg-purple-500/20">
                <Brain className="w-6 h-6 text-purple-400" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">Score Models</h3>
                <p className="text-sm text-neutral-400">Diffusion-based generative models</p>
              </div>
            </div>
            <div className="flex items-center justify-between mt-4 pt-4 border-t border-white/10">
              <span className="text-3xl font-bold text-white">{scoreModels.length}</span>
              <div className="flex gap-2">
                <span className="px-2 py-1 text-xs rounded-full bg-purple-500/20 text-purple-400">
                  {scoreModels.filter(m => m.status === 'trained').length} trained
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Models List */}
        <div className="glass rounded-2xl overflow-hidden">
          <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">All Models</h2>
            <span className="text-sm text-neutral-500">{models.length} total</span>
          </div>
          <div className="divide-y divide-white/5">
            {models.map((model) => (
              <ModelRow
                key={model.id}
                id={model.id}
                name={model.name}
                type={model.type}
                dataset={model.dataset}
                status={model.status}
                metrics={model.metrics}
                onDelete={() => removeModel(model.id)}
              />
            ))}
          </div>
          
          {models.length === 0 && (
            <div className="py-16 text-center">
              <Cpu className="w-12 h-12 text-neutral-600 mx-auto mb-4" />
              <p className="text-neutral-400">No models yet</p>
              <button 
                onClick={() => setTrainOpen(true)}
                className="mt-4 text-blue-400 hover:text-blue-300 transition-colors"
              >
                Train your first model
              </button>
            </div>
          )}
        </div>
      </main>

      {/* Train Model Dialog */}
      <Dialog open={trainOpen} onOpenChange={setTrainOpen}>
        <DialogContent>
          <button 
            onClick={() => setTrainOpen(false)}
            className="absolute top-4 right-4 p-2 rounded-lg hover:bg-white/10 
              text-neutral-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
          
          <DialogHeader>
            <DialogTitle>Train New Model</DialogTitle>
            <DialogDescription>
              Configure and start training a new machine learning model
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 mt-4">
            <Input
              label="Model Name"
              placeholder="e.g., surrogate_qm9_v3"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            />
            
            <Select
              label="Model Type"
              value={formData.type}
              onChange={(e) => setFormData({ ...formData, type: e.target.value })}
              options={[
                { value: 'surrogate', label: 'Surrogate (Energy Prediction)' },
                { value: 'score', label: 'Score (Generative Diffusion)' },
              ]}
            />
            
            <Select
              label="Training Dataset"
              value={formData.dataset}
              onChange={(e) => setFormData({ ...formData, dataset: e.target.value })}
              options={[
                { value: '', label: 'Select a dataset...' },
                ...datasets.map(d => ({ value: d.id, label: d.name }))
              ]}
            />
            
            <div className="grid grid-cols-3 gap-4">
              <Input
                label="Epochs"
                type="number"
                value={formData.epochs}
                onChange={(e) => setFormData({ ...formData, epochs: parseInt(e.target.value) })}
              />
              <Input
                label="Learning Rate"
                type="number"
                step="0.0001"
                value={formData.learningRate}
                onChange={(e) => setFormData({ ...formData, learningRate: parseFloat(e.target.value) })}
              />
              <Input
                label="Batch Size"
                type="number"
                value={formData.batchSize}
                onChange={(e) => setFormData({ ...formData, batchSize: parseInt(e.target.value) })}
              />
            </div>
          </div>

          <div className="flex gap-3 mt-6 pt-4 border-t border-white/10">
            <button 
              onClick={() => setTrainOpen(false)}
              className="flex-1 py-3 rounded-xl bg-white/5 text-white hover:bg-white/10 transition-colors font-medium"
            >
              Cancel
            </button>
            <button 
              onClick={handleTrain}
              disabled={isTraining || !formData.name || !formData.dataset}
              className="flex-1 py-3 rounded-xl bg-blue-600 text-white hover:bg-blue-500 
                transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed
                flex items-center justify-center gap-2"
            >
              {isTraining ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Starting...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Start Training
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

function ModelRow({ id, name, type, dataset, status, metrics, onDelete }: { 
  id: string
  name: string
  type: string
  dataset: string
  status: "trained" | "training" | "failed" | "queued"
  metrics: Record<string, string | number>
  onDelete: () => void
}) {
  const statusConfig = {
    trained: { icon: CheckCircle, color: "text-emerald-400", bg: "bg-emerald-500/20", label: "Trained" },
    training: { icon: Loader2, color: "text-blue-400", bg: "bg-blue-500/20", label: "Training" },
    failed: { icon: X, color: "text-red-400", bg: "bg-red-500/20", label: "Failed" },
    queued: { icon: Clock, color: "text-amber-400", bg: "bg-amber-500/20", label: "Queued" },
  }

  const StatusIcon = statusConfig[status].icon
  const typeColor = type === 'surrogate' ? 'emerald' : 'purple'

  return (
    <div className="px-6 py-4 flex items-center justify-between hover:bg-white/5 transition-colors group">
      <div className="flex items-center gap-4">
        <div className={`w-12 h-12 rounded-xl bg-${typeColor}-500/20 flex items-center justify-center`}>
          {type === 'surrogate' ? (
            <Zap className={`w-6 h-6 text-${typeColor}-400`} />
          ) : (
            <Brain className={`w-6 h-6 text-${typeColor}-400`} />
          )}
        </div>
        <div>
          <p className="font-medium text-white">{name}</p>
          <p className="text-sm text-neutral-500">{type} • {dataset}</p>
        </div>
      </div>

      <div className="flex items-center gap-8">
        <div className="text-right min-w-[120px]">
          {Object.entries(metrics).slice(0, 2).map(([key, value]) => (
            <p key={key} className="text-sm text-neutral-400">
              <span className="text-neutral-500">{key}:</span> {value}
            </p>
          ))}
        </div>

        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${statusConfig[status].bg}`}>
          <StatusIcon className={`w-4 h-4 ${statusConfig[status].color} ${status === "training" ? "animate-spin" : ""}`} />
          <span className={`text-sm ${statusConfig[status].color}`}>{statusConfig[status].label}</span>
        </div>

        <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
          {status === "trained" && (
            <Link
              href={`/models/${id}`}
              className="px-3 py-1.5 rounded-lg bg-blue-500/20 text-blue-400 
                hover:bg-blue-500/30 transition-colors text-sm font-medium"
            >
              View
            </Link>
          )}
          <button 
            onClick={onDelete}
            className="p-2 rounded-lg hover:bg-red-500/20 text-neutral-400 
              hover:text-red-400 transition-colors"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  )
}
