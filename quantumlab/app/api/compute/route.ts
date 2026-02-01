import { NextRequest, NextResponse } from 'next/server'

const jobs = [
  {
    id: 'job-001',
    structure: 'CrCuSe2_gen_045',
    method: 'dft',
    status: 'running',
    progress: 67,
    eta: '1h 23m',
    created: '2026-01-25T10:30:00',
    config: {
      functional: 'PBE',
      kpoints: [4, 4, 4],
      cutoff: 400,
      convergence: 1e-6,
    },
  },
  {
    id: 'job-002',
    structure: 'MoS2_variant_12',
    method: 'xtb',
    status: 'running',
    progress: 89,
    eta: '2m',
    created: '2026-01-25T11:00:00',
    config: {
      method: 'GFN2-xTB',
      optimization: true,
      maxIterations: 500,
    },
  },
]

export async function GET() {
  return NextResponse.json({ jobs })
}

export async function POST(request: NextRequest) {
  const body = await request.json()
  
  const newJob = {
    id: `job-${Date.now()}`,
    structure: body.structure,
    method: body.method,
    status: 'queued',
    progress: 0,
    eta: body.method === 'dft' ? '4h 30m' : '5m',
    created: new Date().toISOString(),
    config: body.config || {},
  }

  return NextResponse.json({ job: newJob })
}
