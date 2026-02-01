import { spawn } from 'child_process'
import path from 'path'

const PYTHON_PATH = 'python'
const SCRIPTS_PATH = path.join(process.cwd(), '..', 'projects', 'phononic-discovery', 'framework', 'scripts')
const DATA_PATH = path.join(process.cwd(), '..', 'projects', 'phononic-discovery', 'framework', 'data')
const MODELS_PATH = path.join(process.cwd(), '..', 'core', 'models')

export interface PythonResult {
  success: boolean
  output: string
  error?: string
}

export async function runPythonScript(scriptName: string, args: string[] = []): Promise<PythonResult> {
  return new Promise((resolve) => {
    const scriptPath = path.join(SCRIPTS_PATH, scriptName)
    const process = spawn(PYTHON_PATH, [scriptPath, ...args], {
      cwd: SCRIPTS_PATH,
    })

    let stdout = ''
    let stderr = ''

    process.stdout.on('data', (data) => {
      stdout += data.toString()
    })

    process.stderr.on('data', (data) => {
      stderr += data.toString()
    })

    process.on('close', (code) => {
      resolve({
        success: code === 0,
        output: stdout,
        error: stderr || undefined,
      })
    })

    process.on('error', (err) => {
      resolve({
        success: false,
        output: '',
        error: err.message,
      })
    })
  })
}

export async function runXTBCalculation(structurePath: string): Promise<PythonResult> {
  return runPythonScript('02_enrich_dataset.py', ['--input', structurePath])
}

export async function trainSurrogateModel(datasetPath: string, outputPath: string): Promise<PythonResult> {
  return runPythonScript('03_train_surrogate.py', ['--data', datasetPath, '--output', outputPath])
}

export async function trainScoreModel(datasetPath: string, outputPath: string): Promise<PythonResult> {
  return runPythonScript('04_train_score_model.py', ['--data', datasetPath, '--output', outputPath])
}

export async function generateStructures(modelPath: string, numSamples: number): Promise<PythonResult> {
  return runPythonScript('07_generate_structures.py', ['--model', modelPath, '--n', numSamples.toString()])
}

export async function runDFTValidation(structurePath: string): Promise<PythonResult> {
  return runPythonScript('08_dft_validation.py', ['--structure', structurePath])
}

export { SCRIPTS_PATH, DATA_PATH, MODELS_PATH }
