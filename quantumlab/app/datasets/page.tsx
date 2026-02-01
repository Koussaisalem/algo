'use client'

import { useState, useRef, useCallback } from 'react'
import Link from 'next/link'
import { 
  Database, Upload, Search, Filter, 
  ChevronRight, Sparkles, FileType, Clock, CheckCircle2,
  AlertCircle, Loader2, X, Download, Trash2, Atom, Cpu, FlaskConical, LineChart, Settings
} from 'lucide-react'
import { useAppStore } from '@/lib/store'
import { uploadFile } from '@/lib/hooks'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'

export default function DatasetsPage() {
  const { datasets, addDataset, removeDataset } = useAppStore()
  const [searchQuery, setSearchQuery] = useState('')
  const [uploadOpen, setUploadOpen] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadError, setUploadError] = useState('')
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const filteredDatasets = datasets.filter(d => 
    d.name.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await handleUpload(e.dataTransfer.files[0])
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleUpload = async (file: File) => {
    setIsUploading(true)
    setUploadError('')

    try {
      const result = await uploadFile(file) as { dataset: { id: string; name: string } }
      addDataset({
        id: result.dataset.id,
        name: result.dataset.name,
        molecules: Math.floor(Math.random() * 5000) + 100,
        size: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
        created: new Date().toISOString().split('T')[0],
        status: 'ready',
        format: file.name.split('.').pop() || 'unknown',
        properties: ['Energy', 'Forces'],
      })
      setUploadOpen(false)
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setIsUploading(false)
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleUpload(e.target.files[0])
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'ready':
        return <CheckCircle2 className="w-4 h-4 text-emerald-400" />
      case 'processing':
        return <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-400" />
      default:
        return <Clock className="w-4 h-4 text-neutral-400" />
    }
  }

  const getStatusBadge = (status: string) => {
    const styles: Record<string, string> = {
      ready: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
      processing: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
      error: 'bg-red-500/20 text-red-400 border-red-500/30',
    }
    return styles[status] || 'bg-neutral-500/20 text-neutral-400'
  }

  return (
    <div className="min-h-screen bg-black">
      {/* Ambient background */}
      <div className="fixed inset-0 bg-gradient-to-br from-blue-950/20 via-black to-purple-950/20" />
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
          <NavItem href="/datasets" icon={<Upload className="w-4 h-4" />} label="Datasets" active />
          <NavItem href="/models" icon={<Cpu className="w-4 h-4" />} label="Models" />
          <NavItem href="/compute" icon={<FlaskConical className="w-4 h-4" />} label="Compute" />
          <NavItem href="/results" icon={<LineChart className="w-4 h-4" />} label="Results" />
        </nav>

        <div className="absolute bottom-6 left-6 right-6">
          <NavItem href="/settings" icon={<Settings className="w-4 h-4" />} label="Settings" />
        </div>
      </aside>
      
      <main className="relative z-10 ml-72 px-10 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <div className="flex items-center gap-2 text-sm text-neutral-500 mb-2">
              <Link href="/dashboard" className="hover:text-white transition-colors">Dashboard</Link>
              <ChevronRight className="w-4 h-4" />
              <span className="text-white">Datasets</span>
            </div>
            <h1 className="text-4xl font-bold text-gradient">Datasets</h1>
            <p className="text-neutral-400 mt-2">Manage your molecular datasets and training data</p>
          </div>
          
          <button 
            onClick={() => setUploadOpen(true)}
            className="btn-shine flex items-center gap-2 px-6 py-3 rounded-xl font-medium"
          >
            <Upload className="w-5 h-5" />
            Upload Dataset
          </button>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          {[
            { label: 'Total Datasets', value: datasets.length.toString(), color: 'blue', icon: Database },
            { label: 'Total Molecules', value: datasets.reduce((a, d) => a + d.molecules, 0).toLocaleString(), color: 'emerald', icon: Sparkles },
            { label: 'Ready', value: datasets.filter(d => d.status === 'ready').length.toString(), color: 'green', icon: CheckCircle2 },
            { label: 'Processing', value: datasets.filter(d => d.status === 'processing').length.toString(), color: 'amber', icon: Loader2 },
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
              placeholder="Search datasets..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-12 pr-4 py-3 rounded-xl bg-white/5 border border-white/10 
                text-white placeholder:text-neutral-500 focus:outline-none focus:border-blue-500/50
                focus:ring-2 focus:ring-blue-500/20 transition-all"
            />
          </div>
          <button className="flex items-center gap-2 px-4 py-3 rounded-xl bg-white/5 border border-white/10
            text-neutral-300 hover:bg-white/10 hover:text-white transition-all">
            <Filter className="w-5 h-5" />
            Filters
          </button>
        </div>

        {/* Datasets Table */}
        <div className="glass rounded-2xl overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-white/10">
                <th className="px-6 py-4 text-left text-sm font-medium text-neutral-400">Name</th>
                <th className="px-6 py-4 text-left text-sm font-medium text-neutral-400">Molecules</th>
                <th className="px-6 py-4 text-left text-sm font-medium text-neutral-400">Properties</th>
                <th className="px-6 py-4 text-left text-sm font-medium text-neutral-400">Size</th>
                <th className="px-6 py-4 text-left text-sm font-medium text-neutral-400">Status</th>
                <th className="px-6 py-4 text-left text-sm font-medium text-neutral-400">Created</th>
                <th className="px-6 py-4 text-right text-sm font-medium text-neutral-400">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {filteredDatasets.map((dataset) => (
                <tr key={dataset.id} className="hover:bg-white/5 transition-colors group">
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-blue-500/20">
                        <FileType className="w-5 h-5 text-blue-400" />
                      </div>
                      <div>
                        <p className="font-medium text-white">{dataset.name}</p>
                        <p className="text-sm text-neutral-500">{dataset.id}</p>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-neutral-300">{dataset.molecules.toLocaleString()}</td>
                  <td className="px-6 py-4">
                    <div className="flex gap-1 flex-wrap">
                      {dataset.properties.slice(0, 3).map((prop, i) => (
                        <span key={i} className="px-2 py-1 text-xs rounded-full bg-white/10 text-neutral-300">
                          {prop}
                        </span>
                      ))}
                      {dataset.properties.length > 3 && (
                        <span className="px-2 py-1 text-xs rounded-full bg-white/10 text-neutral-400">
                          +{dataset.properties.length - 3}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-4 text-neutral-300">{dataset.size}</td>
                  <td className="px-6 py-4">
                    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs 
                      font-medium border ${getStatusBadge(dataset.status)}`}>
                      {getStatusIcon(dataset.status)}
                      {dataset.status.charAt(0).toUpperCase() + dataset.status.slice(1)}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-neutral-400">{dataset.created}</td>
                  <td className="px-6 py-4">
                    <div className="flex items-center justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <Link 
                        href={`/datasets/${dataset.id}`}
                        className="px-3 py-1.5 rounded-lg bg-blue-500/20 text-blue-400 
                          hover:bg-blue-500/30 transition-colors text-sm font-medium"
                      >
                        View
                      </Link>
                      <button className="p-2 rounded-lg hover:bg-white/10 text-neutral-400 
                        hover:text-white transition-colors">
                        <Download className="w-4 h-4" />
                      </button>
                      <button 
                        onClick={() => removeDataset(dataset.id)}
                        className="p-2 rounded-lg hover:bg-red-500/20 text-neutral-400 
                          hover:text-red-400 transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {filteredDatasets.length === 0 && (
            <div className="py-16 text-center">
              <Database className="w-12 h-12 text-neutral-600 mx-auto mb-4" />
              <p className="text-neutral-400">No datasets found</p>
              <button 
                onClick={() => setUploadOpen(true)}
                className="mt-4 text-blue-400 hover:text-blue-300 transition-colors"
              >
                Upload your first dataset
              </button>
            </div>
          )}
        </div>
      </main>

      {/* Upload Dialog */}
      <Dialog open={uploadOpen} onOpenChange={setUploadOpen}>
        <DialogContent>
          <button 
            onClick={() => setUploadOpen(false)}
            className="absolute top-4 right-4 p-2 rounded-lg hover:bg-white/10 
              text-neutral-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
          
          <DialogHeader>
            <DialogTitle>Upload Dataset</DialogTitle>
            <DialogDescription>
              Upload molecular data in .pt, .xyz, .sdf, .mol2, or .pdb format
            </DialogDescription>
          </DialogHeader>

          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`relative mt-4 p-8 rounded-xl border-2 border-dashed cursor-pointer
              transition-all duration-200 text-center
              ${dragActive 
                ? 'border-blue-500 bg-blue-500/10' 
                : 'border-white/20 hover:border-white/40 bg-white/5'
              }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pt,.xyz,.sdf,.mol2,.pdb"
              onChange={handleFileSelect}
              className="hidden"
            />
            
            {isUploading ? (
              <div className="flex flex-col items-center">
                <Loader2 className="w-10 h-10 text-blue-400 animate-spin mb-4" />
                <p className="text-white font-medium">Uploading...</p>
              </div>
            ) : (
              <>
                <Upload className={`w-10 h-10 mx-auto mb-4 ${dragActive ? 'text-blue-400' : 'text-neutral-400'}`} />
                <p className="text-white font-medium mb-1">
                  {dragActive ? 'Drop file here' : 'Drag & drop or click to upload'}
                </p>
                <p className="text-sm text-neutral-500">
                  Supports .pt, .xyz, .sdf, .mol2, .pdb files up to 100MB
                </p>
              </>
            )}
          </div>

          {uploadError && (
            <div className="mt-4 p-3 rounded-lg bg-red-500/20 border border-red-500/30 text-red-400 text-sm">
              {uploadError}
            </div>
          )}

          <div className="mt-6 pt-4 border-t border-white/10">
            <h4 className="text-sm font-medium text-neutral-300 mb-3">Recent Uploads</h4>
            <div className="space-y-2">
              {datasets.slice(0, 3).map(d => (
                <div key={d.id} className="flex items-center justify-between p-2 rounded-lg bg-white/5">
                  <span className="text-sm text-neutral-300">{d.name}</span>
                  <span className="text-xs text-neutral-500">{d.created}</span>
                </div>
              ))}
            </div>
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
