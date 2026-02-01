import { NextRequest, NextResponse } from 'next/server'

// Mock data - in production, this would connect to Python backend
const datasets = [
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
]

export async function GET() {
  return NextResponse.json({ datasets })
}

export async function POST(request: NextRequest) {
  const formData = await request.formData()
  const file = formData.get('file') as File
  
  if (!file) {
    return NextResponse.json({ error: 'No file provided' }, { status: 400 })
  }

  // Simulate processing
  const newDataset = {
    id: `ds-${Date.now()}`,
    name: file.name.replace(/\.[^/.]+$/, ''),
    molecules: Math.floor(Math.random() * 5000) + 500,
    size: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
    created: new Date().toISOString().split('T')[0],
    status: 'processing',
    format: file.name.split('.').pop(),
    properties: [],
  }

  return NextResponse.json({ dataset: newDataset })
}
