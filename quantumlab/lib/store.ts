import { create } from 'zustand'

export interface Dataset {
  id: string
  name: string
  molecules: number
  size: string
  created: string
  status: 'ready' | 'enriched' | 'processing'
  format?: string
  properties: string[]
  path?: string
}

export interface Model {
  id: string
  name: string
  type: 'surrogate' | 'score'
  dataset: string
  status: 'trained' | 'training' | 'queued' | 'failed'
  created: string
  epochs?: number
  currentEpoch?: number
  metrics: Record<string, string | number>
  config?: Record<string, unknown>
  path?: string
}

export interface Result {
  id: string
  name: string
  formula: string
  status: 'validated' | 'stable' | 'unstable' | 'pending'
  isNovel: boolean
  created: string
  source: string
  energy: string
  bandgap: string
  properties: {
    formationEnergy: number
    bandgap: number
    magneticMoment?: number
    bulkModulus?: number
  }
}

interface AppState {
  // Datasets
  datasets: Dataset[]
  addDataset: (dataset: Dataset) => void
  updateDataset: (id: string, updates: Partial<Dataset>) => void
  removeDataset: (id: string) => void
  
  // Models
  models: Model[]
  addModel: (model: Model) => void
  updateModel: (id: string, updates: Partial<Model>) => void
  removeModel: (id: string) => void
  
  // Results
  results: Result[]
  addResult: (result: Result) => void
  updateResult: (id: string, updates: Partial<Result>) => void
  removeResult: (id: string) => void
  
  // UI State
  selectedDataset: string | null
  selectedModel: string | null
  setSelectedDataset: (id: string | null) => void
  setSelectedModel: (id: string | null) => void
}

export const useAppStore = create<AppState>((set) => ({
  // Initial datasets
  datasets: [
    {
      id: 'ds-001',
      name: 'qm9_micro_5k',
      molecules: 5000,
      size: '124 MB',
      created: '2026-01-15',
      status: 'enriched',
      format: 'pt',
      properties: ['energy', 'forces', 'dipole'],
    },
    {
      id: 'ds-002',
      name: 'tmd_structures',
      molecules: 1200,
      size: '45 MB',
      created: '2026-01-12',
      status: 'ready',
      format: 'xyz',
      properties: ['energy'],
    },
    {
      id: 'ds-003',
      name: 'perovskites_v2',
      molecules: 3500,
      size: '89 MB',
      created: '2026-01-10',
      status: 'processing',
      format: 'pt',
      properties: ['energy', 'bandgap'],
    },
  ],
  
  addDataset: (dataset) => set((state) => ({ 
    datasets: [...state.datasets, dataset] 
  })),
  
  updateDataset: (id, updates) => set((state) => ({
    datasets: state.datasets.map(d => d.id === id ? { ...d, ...updates } : d)
  })),
  
  removeDataset: (id) => set((state) => ({
    datasets: state.datasets.filter(d => d.id !== id)
  })),
  
  // Initial models
  models: [
    {
      id: 'model-001',
      name: 'surrogate_qm9_v2',
      type: 'surrogate' as const,
      dataset: 'qm9_micro_5k',
      status: 'trained' as const,
      created: '2026-01-20',
      epochs: 100,
      metrics: { mae: 0.023, rmse: 0.045, r2: 0.987 } as Record<string, string | number>,
      config: { architecture: 'SchNet', hiddenChannels: 128, numLayers: 6 },
    },
    {
      id: 'model-002',
      name: 'score_manifold_v1',
      type: 'score' as const,
      dataset: 'qm9_micro_5k',
      status: 'training' as const,
      created: '2026-01-22',
      epochs: 100,
      currentEpoch: 85,
      metrics: { loss: 0.0023, validity: 0.95 } as Record<string, string | number>,
      config: { architecture: 'E3NN', diffusionSteps: 1000 },
    },
  ] as Model[],
  
  addModel: (model) => set((state) => ({ 
    models: [...state.models, model] 
  })),
  
  updateModel: (id, updates) => set((state) => ({
    models: state.models.map(m => m.id === id ? { ...m, ...updates } : m)
  })),
  
  removeModel: (id) => set((state) => ({
    models: state.models.filter(m => m.id !== id)
  })),
  
  // Initial results
  results: [
    {
      id: 'CrCuSe2',
      name: 'CrCuSe2',
      formula: 'Cr₂Cu₂Se₄',
      status: 'validated',
      isNovel: true,
      created: '2026-01-24',
      source: 'manifold_diffusion',
      energy: '-4.23 eV/atom',
      bandgap: '0.8 eV',
      properties: { formationEnergy: -4.23, bandgap: 0.8, magneticMoment: 2.4, bulkModulus: 45.2 },
    },
    {
      id: 'MoS2_variant',
      name: 'MoS2_variant',
      formula: 'Mo₂S₄',
      status: 'validated',
      isNovel: false,
      created: '2026-01-23',
      source: 'dataset',
      energy: '-5.12 eV/atom',
      bandgap: '1.2 eV',
      properties: { formationEnergy: -5.12, bandgap: 1.2 },
    },
  ],
  
  addResult: (result) => set((state) => ({ 
    results: [...state.results, result] 
  })),
  
  updateResult: (id, updates) => set((state) => ({
    results: state.results.map(r => r.id === id ? { ...r, ...updates } : r)
  })),
  
  removeResult: (id) => set((state) => ({
    results: state.results.filter(r => r.id !== id)
  })),
  
  // UI State
  selectedDataset: null,
  selectedModel: null,
  setSelectedDataset: (id) => set({ selectedDataset: id }),
  setSelectedModel: (id) => set({ selectedModel: id }),
}))
