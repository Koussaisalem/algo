import { NextRequest, NextResponse } from 'next/server'
import { writeFile, mkdir } from 'fs/promises'
import path from 'path'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get('file') as File
    
    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 })
    }

    const bytes = await file.arrayBuffer()
    const buffer = Buffer.from(bytes)

    // Create uploads directory if it doesn't exist
    const uploadsDir = path.join(process.cwd(), 'uploads')
    await mkdir(uploadsDir, { recursive: true })

    // Save file with timestamp
    const fileName = `${Date.now()}-${file.name}`
    const filePath = path.join(uploadsDir, fileName)
    await writeFile(filePath, buffer)

    // Parse file info
    const extension = file.name.split('.').pop()?.toLowerCase() || ''
    const isValid = ['pt', 'xyz', 'sdf', 'mol2', 'pdb'].includes(extension)

    if (!isValid) {
      return NextResponse.json({ 
        error: `Unsupported file format: ${extension}. Supported: .pt, .xyz, .sdf, .mol2, .pdb` 
      }, { status: 400 })
    }

    // Create dataset record
    const dataset = {
      id: `ds-${Date.now()}`,
      name: file.name.replace(/\.[^/.]+$/, ''),
      molecules: 0, // Will be updated after parsing
      size: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
      created: new Date().toISOString().split('T')[0],
      status: 'processing' as const,
      format: extension,
      properties: [],
      path: filePath,
    }

    return NextResponse.json({ 
      success: true,
      dataset,
      message: 'File uploaded successfully. Processing...'
    })
  } catch (error) {
    console.error('Upload error:', error)
    return NextResponse.json({ 
      error: 'Failed to upload file' 
    }, { status: 500 })
  }
}
