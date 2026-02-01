import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.INFERENCE_BACKEND_URL || 'http://localhost:8000'

interface InferenceRequest {
  numSamples?: number
  numAtoms?: number
  temperature?: number
  numDiffusionSteps?: number
  seed?: number
  elementTypes?: string[]
  targetBandGap?: number
  guidanceStrength?: number
}

export async function POST(request: NextRequest) {
  try {
    const body: InferenceRequest = await request.json()
    
    const {
      numSamples = 3,
      numAtoms = 12,
      temperature = 1.0,
      numDiffusionSteps = 100,
      seed,
      elementTypes = ['C', 'N', 'O', 'H'],
      targetBandGap,
      guidanceStrength = 1.0,
    } = body

    // Validate inputs
    if (numSamples < 1 || numSamples > 50) {
      return NextResponse.json({ 
        error: 'numSamples must be between 1 and 50' 
      }, { status: 400 })
    }

    if (numAtoms < 2 || numAtoms > 100) {
      return NextResponse.json({ 
        error: 'numAtoms must be between 2 and 100' 
      }, { status: 400 })
    }

    // Try to call the Python backend
    try {
      const backendResponse = await fetch(`${BACKEND_URL}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          num_samples: numSamples,
          num_atoms: numAtoms,
          temperature,
          num_diffusion_steps: numDiffusionSteps,
          seed,
          element_types: elementTypes,
          target_band_gap: targetBandGap,
          guidance_strength: guidanceStrength,
        }),
      })

      if (backendResponse.ok) {
        const data = await backendResponse.json()
        return NextResponse.json(data)
      }
    } catch (backendError) {
      console.log('Backend not available, using mock generation')
    }

    // Fallback to mock generation if backend is not available
    const molecules = generateMockMolecules(numSamples, numAtoms, elementTypes, seed)
    
    return NextResponse.json({
      success: true,
      molecules,
      generation_time: 0.5,
      model_info: {
        backend: 'Mock (Python backend not running)',
        diffusion_steps: numDiffusionSteps,
        temperature,
      }
    })
  } catch (error) {
    console.error('Inference error:', error)
    return NextResponse.json({ 
      error: 'Failed to generate molecules' 
    }, { status: 500 })
  }
}

export async function GET() {
  // Check backend status
  try {
    const response = await fetch(`${BACKEND_URL}/models`)
    if (response.ok) {
      const data = await response.json()
      return NextResponse.json({
        status: 'online',
        backend: 'Python inference server',
        ...data
      })
    }
  } catch {
    // Backend not available
  }
  
  return NextResponse.json({
    status: 'fallback',
    backend: 'Mock mode (start backend with: python backend/inference_server.py)',
    models: [
      { name: 'score_model', type: 'score', status: 'available' },
      { name: 'surrogate', type: 'surrogate', status: 'available' },
    ]
  })
}

// Mock molecule generation for when backend is not available
function generateMockMolecules(
  numSamples: number,
  numAtoms: number,
  elementTypes: string[],
  seed?: number
) {
  const ELEMENT_DATA: Record<string, { color: string; radius: number }> = {
    H: { color: '#FFFFFF', radius: 0.31 },
    C: { color: '#909090', radius: 0.77 },
    N: { color: '#3050F8', radius: 0.71 },
    O: { color: '#FF0D0D', radius: 0.66 },
    S: { color: '#FFFF30', radius: 1.05 },
    F: { color: '#90E050', radius: 0.57 },
    Cl: { color: '#1FF01F', radius: 1.02 },
    Br: { color: '#A62929', radius: 1.20 },
    P: { color: '#FF8000', radius: 1.07 },
  }

  // Simple seeded random
  let currentSeed = seed || Date.now()
  const random = () => {
    currentSeed = (currentSeed * 1103515245 + 12345) & 0x7fffffff
    return currentSeed / 0x7fffffff
  }

  return Array.from({ length: numSamples }, (_, molIdx) => {
    // Generate atoms with positions
    const atoms = []
    const positions: [number, number, number][] = [[0, 0, 0]]
    
    for (let i = 0; i < numAtoms; i++) {
      const element = elementTypes[Math.floor(random() * elementTypes.length)]
      const elemData = ELEMENT_DATA[element] || { color: '#CCCCCC', radius: 1.0 }
      
      let x, y, z
      if (i === 0) {
        x = y = z = 0
      } else {
        // Add near existing atom
        const refIdx = Math.floor(random() * positions.length)
        const ref = positions[refIdx]
        const theta = random() * Math.PI * 2
        const phi = random() * Math.PI
        const r = 1.2 + random() * 0.6
        
        x = ref[0] + r * Math.sin(phi) * Math.cos(theta)
        y = ref[1] + r * Math.sin(phi) * Math.sin(theta)
        z = ref[2] + r * Math.cos(phi)
        positions.push([x, y, z])
      }
      
      atoms.push({
        index: i,
        element,
        x,
        y,
        z,
        color: elemData.color,
        radius: elemData.radius,
      })
    }
    
    // Infer bonds
    const bonds = []
    for (let i = 0; i < atoms.length; i++) {
      for (let j = i + 1; j < atoms.length; j++) {
        const dx = atoms[i].x - atoms[j].x
        const dy = atoms[i].y - atoms[j].y
        const dz = atoms[i].z - atoms[j].z
        const dist = Math.sqrt(dx*dx + dy*dy + dz*dz)
        
        const maxDist = (atoms[i].radius + atoms[j].radius) * 1.3
        if (dist < maxDist) {
          bonds.push({
            atom1: i,
            atom2: j,
            order: dist < maxDist * 0.8 ? 2 : 1,
            length: dist,
          })
        }
      }
    }
    
    // Calculate formula
    const counts: Record<string, number> = {}
    atoms.forEach(a => { counts[a.element] = (counts[a.element] || 0) + 1 })
    let formula = ''
    if (counts.C) { formula += `C${counts.C > 1 ? counts.C : ''}`; delete counts.C }
    if (counts.H) { formula += `H${counts.H > 1 ? counts.H : ''}`; delete counts.H }
    Object.keys(counts).sort().forEach(e => {
      formula += `${e}${counts[e] > 1 ? counts[e] : ''}`
    })
    
    // Generate XYZ content
    const xyzLines = [
      String(atoms.length),
      `Generated by QuantumLab - Molecule ${molIdx + 1}`,
      ...atoms.map(a => `${a.element.padEnd(2)}  ${a.x.toFixed(6).padStart(12)}  ${a.y.toFixed(6).padStart(12)}  ${a.z.toFixed(6).padStart(12)}`)
    ]
    
    return {
      id: `mol_${Date.now()}_${molIdx.toString().padStart(3, '0')}`,
      atoms,
      bonds,
      properties: {
        total_energy: -10 - random() * 50,
        formation_energy: -0.5 - random() * 2,
        band_gap: random() * 3,
        dipole_moment: random() * 5,
        num_atoms: numAtoms,
        valid: true,
      },
      xyz_content: xyzLines.join('\n'),
      formula,
      generated_at: new Date().toISOString(),
    }
  })
}
