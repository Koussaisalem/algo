import { NextRequest, NextResponse } from 'next/server'
import { jobQueue } from '@/lib/job-queue'

export async function GET() {
  const jobs = jobQueue.getAllJobs()
  return NextResponse.json({ 
    jobs,
    stats: {
      total: jobs.length,
      running: jobs.filter(j => j.status === 'running').length,
      queued: jobs.filter(j => j.status === 'queued').length,
      completed: jobs.filter(j => j.status === 'completed').length,
      failed: jobs.filter(j => j.status === 'failed').length,
    }
  })
}

export async function POST(request: NextRequest) {
  const body = await request.json()
  const { type, structure, dataset, model, config } = body
  
  if (!type) {
    return NextResponse.json({ error: 'Job type is required' }, { status: 400 })
  }
  
  const job = jobQueue.createJob(type, {
    structure,
    dataset,
    model,
    config,
    eta: type === 'dft' ? '4h 30m' : type === 'xtb' ? '5m' : '2h',
  })
  
  // Start simulated job progress in background
  jobQueue.simulateJobProgress(job.id)
  
  return NextResponse.json({ job })
}
