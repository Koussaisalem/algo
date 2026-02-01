// In-memory job queue for demo purposes
// In production, use Redis or a proper job queue

export interface Job {
  id: string
  type: 'xtb' | 'dft' | 'train_surrogate' | 'train_score' | 'generate'
  status: 'queued' | 'running' | 'completed' | 'failed'
  progress: number
  structure?: string
  dataset?: string
  model?: string
  config?: Record<string, unknown>
  result?: unknown
  error?: string
  createdAt: Date
  updatedAt: Date
  eta?: string
  data?: {
    structure?: string
    eta?: string
    [key: string]: unknown
  }
}

class JobQueue {
  private jobs: Map<string, Job> = new Map()
  private listeners: Map<string, ((job: Job) => void)[]> = new Map()

  createJob(type: Job['type'], params: Partial<Job>): Job {
    const id = `job-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    const job: Job = {
      id,
      type,
      status: 'queued',
      progress: 0,
      createdAt: new Date(),
      updatedAt: new Date(),
      ...params,
    }
    this.jobs.set(id, job)
    return job
  }

  getJob(id: string): Job | undefined {
    return this.jobs.get(id)
  }

  getAllJobs(): Job[] {
    return Array.from(this.jobs.values()).sort((a, b) => 
      b.createdAt.getTime() - a.createdAt.getTime()
    )
  }

  getJobsByStatus(status: Job['status']): Job[] {
    return this.getAllJobs().filter(job => job.status === status)
  }

  getJobsByType(type: Job['type']): Job[] {
    return this.getAllJobs().filter(job => job.type === type)
  }

  updateJob(id: string, updates: Partial<Job>): Job | undefined {
    const job = this.jobs.get(id)
    if (!job) return undefined
    
    const updatedJob = {
      ...job,
      ...updates,
      updatedAt: new Date(),
    }
    this.jobs.set(id, updatedJob)
    this.notifyListeners(id, updatedJob)
    return updatedJob
  }

  deleteJob(id: string): boolean {
    return this.jobs.delete(id)
  }

  subscribe(jobId: string, callback: (job: Job) => void): () => void {
    if (!this.listeners.has(jobId)) {
      this.listeners.set(jobId, [])
    }
    this.listeners.get(jobId)!.push(callback)
    
    return () => {
      const callbacks = this.listeners.get(jobId)
      if (callbacks) {
        const index = callbacks.indexOf(callback)
        if (index > -1) callbacks.splice(index, 1)
      }
    }
  }

  private notifyListeners(jobId: string, job: Job) {
    const callbacks = this.listeners.get(jobId) || []
    callbacks.forEach(cb => cb(job))
  }

  // Simulate job progress for demo
  async simulateJobProgress(jobId: string): Promise<void> {
    const job = this.getJob(jobId)
    if (!job) return

    this.updateJob(jobId, { status: 'running', progress: 0 })
    
    const totalSteps = job.type === 'dft' ? 100 : job.type === 'xtb' ? 20 : 50
    const stepTime = job.type === 'dft' ? 500 : job.type === 'xtb' ? 100 : 200
    
    for (let i = 1; i <= totalSteps; i++) {
      await new Promise(resolve => setTimeout(resolve, stepTime))
      const progress = Math.round((i / totalSteps) * 100)
      const remainingSteps = totalSteps - i
      const etaSeconds = (remainingSteps * stepTime) / 1000
      const eta = etaSeconds > 60 
        ? `${Math.round(etaSeconds / 60)}m` 
        : `${Math.round(etaSeconds)}s`
      
      this.updateJob(jobId, { progress, eta })
    }

    // Simulate result
    const success = Math.random() > 0.1 // 90% success rate
    if (success) {
      this.updateJob(jobId, { 
        status: 'completed', 
        progress: 100,
        result: {
          energy: -4.23 + Math.random() * 0.5,
          bandgap: 0.5 + Math.random() * 2,
          forces: 0.001 + Math.random() * 0.01,
        }
      })
    } else {
      this.updateJob(jobId, { 
        status: 'failed', 
        error: 'Convergence not achieved after 500 iterations'
      })
    }
  }
}

export const jobQueue = new JobQueue()

// Initialize with some demo jobs
const demoJobs: Partial<Job>[] = [
  {
    type: 'dft',
    structure: 'CrCuSe2_gen_045',
    status: 'running',
    progress: 67,
    eta: '1h 23m',
  },
  {
    type: 'xtb',
    structure: 'MoS2_variant_12',
    status: 'running',
    progress: 89,
    eta: '2m',
  },
  {
    type: 'dft',
    structure: 'WSe2_doped_003',
    status: 'queued',
    progress: 0,
    eta: '4h 30m',
  },
]

demoJobs.forEach(params => {
  jobQueue.createJob(params.type as Job['type'], params)
})
