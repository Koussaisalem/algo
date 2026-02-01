import { NextRequest, NextResponse } from 'next/server'

// Mock training function - in production, this would spawn a Python process
async function startTraining(config: {
  name: string
  type: 'surrogate' | 'score'
  dataset: string
  epochs: number
  learningRate: number
  batchSize: number
}) {
  const model = {
    id: `model-${Date.now()}`,
    name: config.name,
    type: config.type,
    dataset: config.dataset,
    status: 'training' as const,
    created: new Date().toISOString().split('T')[0],
    epochs: config.epochs,
    currentEpoch: 0,
    metrics: {},
    config: {
      learningRate: config.learningRate,
      batchSize: config.batchSize,
      architecture: config.type === 'surrogate' ? 'SchNet' : 'E3NN',
    },
  }

  return model
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    const { name, type, dataset, epochs = 100, learningRate = 0.001, batchSize = 32 } = body
    
    if (!name || !type || !dataset) {
      return NextResponse.json({ 
        error: 'name, type, and dataset are required' 
      }, { status: 400 })
    }
    
    if (!['surrogate', 'score'].includes(type)) {
      return NextResponse.json({ 
        error: 'type must be "surrogate" or "score"' 
      }, { status: 400 })
    }

    const model = await startTraining({
      name,
      type,
      dataset,
      epochs,
      learningRate,
      batchSize,
    })

    return NextResponse.json({ 
      success: true,
      model,
      message: `Training started for ${type} model "${name}"`
    })
  } catch (error) {
    console.error('Training error:', error)
    return NextResponse.json({ 
      error: 'Failed to start training' 
    }, { status: 500 })
  }
}
