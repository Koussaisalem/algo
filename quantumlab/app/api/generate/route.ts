import { NextRequest, NextResponse } from 'next/server'

interface GenerateRequest {
  model: string
  numSamples: number
  temperature?: number
  seed?: number
}

export async function POST(request: NextRequest) {
  try {
    const body: GenerateRequest = await request.json()
    
    const { model, numSamples = 10, temperature = 1.0, seed } = body
    
    if (!model) {
      return NextResponse.json({ 
        error: 'model is required' 
      }, { status: 400 })
    }
    
    if (numSamples < 1 || numSamples > 1000) {
      return NextResponse.json({ 
        error: 'numSamples must be between 1 and 1000' 
      }, { status: 400 })
    }

    // Mock generation - in production, this would call the Python backend
    const generatedStructures = Array.from({ length: numSamples }, (_, i) => ({
      id: `gen-${Date.now()}-${i}`,
      name: `generated_${i + 1}`,
      formula: ['CrCuSe2', 'MoS2', 'WSe2', 'TiO2', 'CsPbI3'][Math.floor(Math.random() * 5)],
      status: 'pending' as const,
      isNovel: Math.random() > 0.7,
      created: new Date().toISOString().split('T')[0],
      source: model,
      properties: {
        formationEnergy: -3 - Math.random() * 5,
        bandgap: Math.random() * 3,
      },
    }))

    return NextResponse.json({ 
      success: true,
      structures: generatedStructures,
      message: `Generated ${numSamples} structures using ${model}`
    })
  } catch (error) {
    console.error('Generation error:', error)
    return NextResponse.json({ 
      error: 'Failed to generate structures' 
    }, { status: 500 })
  }
}
