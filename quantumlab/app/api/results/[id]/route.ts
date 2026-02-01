import { NextRequest, NextResponse } from 'next/server'

const results: Record<string, any> = {
  'CrCuSe2': {
    id: 'result-001',
    name: 'CrCuSe2',
    formula: 'Cr2Cu2Se4',
    status: 'validated',
    isNovel: true,
    created: '2026-01-24',
    source: 'manifold_diffusion',
    properties: {
      formationEnergy: -4.23,
      bandgap: 0.8,
      magneticMoment: 2.4,
      bulkModulus: 45.2,
    },
    structure: {
      spaceGroup: 'P-3m1',
      latticeParams: { a: 3.82, b: 3.82, c: 6.45, alpha: 90, beta: 90, gamma: 120 },
      atoms: [
        { element: 'Cr', position: [0, 0, 0] },
        { element: 'Cr', position: [0.333, 0.667, 0.5] },
        { element: 'Cu', position: [0.667, 0.333, 0.25] },
        { element: 'Cu', position: [0.333, 0.667, 0.75] },
        { element: 'Se', position: [0.333, 0.667, 0.12] },
        { element: 'Se', position: [0.667, 0.333, 0.62] },
        { element: 'Se', position: [0.333, 0.667, 0.38] },
        { element: 'Se', position: [0.667, 0.333, 0.88] },
      ],
    },
    validation: {
      xtb: { status: 'passed', energy: -4.21, forces: 0.02 },
      dft: { status: 'passed', energy: -4.23, forces: 0.005 },
      phonon: { status: 'passed', imaginaryModes: 0 },
    },
  },
}

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const result = results[params.id]
  
  if (!result) {
    return NextResponse.json({ error: 'Result not found' }, { status: 404 })
  }

  return NextResponse.json({ result })
}
