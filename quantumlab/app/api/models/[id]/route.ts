import { NextRequest, NextResponse } from 'next/server'

const models: Record<string, any> = {
  'surrogate_qm9_v2': {
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
      validationLoss: 0.0012,
    },
    config: {
      architecture: 'SchNet',
      hiddenChannels: 128,
      numLayers: 6,
      cutoff: 5.0,
      batchSize: 32,
      learningRate: 0.001,
      optimizer: 'Adam',
    },
    trainingHistory: {
      trainLoss: [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.008, 0.005, 0.003, 0.0012],
      valLoss: [0.55, 0.25, 0.12, 0.06, 0.03, 0.015, 0.01, 0.006, 0.004, 0.0015],
    },
  },
  'score_manifold_v1': {
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
      uniqueness: 0.87,
      novelty: 0.92,
    },
    config: {
      architecture: 'E3NN',
      diffusionSteps: 1000,
      noiseSchedule: 'cosine',
      batchSize: 64,
      learningRate: 0.0001,
    },
  },
}

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const model = models[params.id]
  
  if (!model) {
    return NextResponse.json({ error: 'Model not found' }, { status: 404 })
  }

  return NextResponse.json({ model })
}
