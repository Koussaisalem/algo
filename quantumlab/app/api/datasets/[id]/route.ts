import { NextRequest, NextResponse } from 'next/server'

const datasets: Record<string, any> = {
  'qm9_micro_5k': {
    id: 'ds-001',
    name: 'qm9_micro_5k',
    molecules: 5000,
    size: '124 MB',
    created: '2026-01-15',
    status: 'enriched',
    format: 'pt',
    properties: ['energy', 'forces', 'dipole'],
    description: 'Subset of QM9 dataset enriched with xTB formation energies',
    stats: {
      avgAtoms: 9.2,
      elements: ['C', 'H', 'O', 'N', 'F'],
      energyRange: [-400.5, -150.2],
      avgEnergy: -245.3,
    },
    enrichment: {
      method: 'GFN2-xTB',
      converged: 4850,
      failed: 150,
      avgTime: '2.3 min',
    },
  },
  'tmd_structures': {
    id: 'ds-002',
    name: 'tmd_structures',
    molecules: 1200,
    size: '45 MB',
    created: '2026-01-12',
    status: 'ready',
    format: 'xyz',
    properties: ['energy'],
    description: 'Transition metal dichalcogenide structures for phononic materials',
    stats: {
      avgAtoms: 24.5,
      elements: ['Mo', 'W', 'S', 'Se', 'Te'],
      energyRange: [-850.2, -320.1],
      avgEnergy: -512.4,
    },
  },
}

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const dataset = datasets[params.id]
  
  if (!dataset) {
    return NextResponse.json({ error: 'Dataset not found' }, { status: 404 })
  }

  return NextResponse.json({ dataset })
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  // Simulate deletion
  return NextResponse.json({ success: true })
}
