import { NextRequest, NextResponse } from 'next/server'

const models = [
  {
    id: 'model-001',
    name: 'surrogate_qm9_v2',
    type: 'surrogate',
    dataset: 'qm9_micro_5k',
    status: 'trained',
    created: '2026-01-20',
    epochs: 100,
    metrics: {
      mae: 0.023,
      rmse: 0.045,
      r2: 0.987,
    },
    config: {
      architecture: 'SchNet',
      hiddenChannels: 128,
      numLayers: 6,
      cutoff: 5.0,
    },
  },
  {
    id: 'model-002',
    name: 'score_manifold_v1',
    type: 'score',
    dataset: 'qm9_micro_5k',
    status: 'training',
    created: '2026-01-22',
    epochs: 100,
    currentEpoch: 85,
    metrics: {
      loss: 0.0023,
      validity: 0.95,
    },
    config: {
      architecture: 'E3NN',
      diffusionSteps: 1000,
      noiseSchedule: 'cosine',
    },
  },
]

export async function GET() {
  return NextResponse.json({ models })
}

export async function POST(request: NextRequest) {
  const body = await request.json()
  
  const newModel = {
    id: `model-${Date.now()}`,
    name: body.name,
    type: body.type,
    dataset: body.dataset,
    status: 'queued',
    created: new Date().toISOString().split('T')[0],
    epochs: body.epochs || 100,
    currentEpoch: 0,
    metrics: {},
    config: body.config || {},
  }

  return NextResponse.json({ model: newModel })
}
