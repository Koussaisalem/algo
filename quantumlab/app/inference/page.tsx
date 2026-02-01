'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import Link from 'next/link';
import { 
  ArrowLeft, 
  Atom, 
  Play, 
  Download, 
  RefreshCw, 
  Settings2,
  Sparkles,
  Box,
  Zap,
  ChevronDown,
  Copy,
  Check,
  Loader2,
  Info,
  FlaskConical,
  Maximize2,
  Star,
} from 'lucide-react';

// Types
interface GeneratedAtom {
  index: number;
  element: string;
  x: number;
  y: number;
  z: number;
  color: string;
  radius: number;
}

interface GeneratedBond {
  atom1: number;
  atom2: number;
  order: number;
  length: number;
}

interface MoleculeProperties {
  total_energy: number;
  formation_energy: number;
  band_gap: number;
  dipole_moment: number;
  num_atoms: number;
  valid: boolean;
}

interface GeneratedMolecule {
  id: string;
  atoms: GeneratedAtom[];
  bonds: GeneratedBond[];
  properties: MoleculeProperties;
  xyz_content: string;
  formula: string;
  generated_at: string;
}

interface GenerationParams {
  numSamples: number;
  numAtoms: number;
  temperature: number;
  numDiffusionSteps: number;
  seed: number | null;
  elementTypes: string[];
  targetBandGap: number | null;
  guidanceStrength: number;
}

// Element colors and data for visualization
const ELEMENT_COLORS: Record<string, string> = {
  H: '#FFFFFF',
  C: '#909090',
  N: '#3050F8',
  O: '#FF0D0D',
  S: '#FFFF30',
  F: '#90E050',
  Cl: '#1FF01F',
  Br: '#A62929',
  P: '#FF8000',
  Cr: '#8A99C7',
  Cu: '#C88033',
  Se: '#FFA100',
  Mo: '#54B5B5',
  W: '#2194D6',
  Ti: '#BFC2C7',
};

const AVAILABLE_ELEMENTS = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl', 'P', 'Br'];

export default function InferencePage() {
  const [molecules, setMolecules] = useState<GeneratedMolecule[]>([]);
  const [selectedMolecule, setSelectedMolecule] = useState<GeneratedMolecule | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'fallback'>('checking');
  
  const [params, setParams] = useState<GenerationParams>({
    numSamples: 3,
    numAtoms: 12,
    temperature: 1.0,
    numDiffusionSteps: 100,
    seed: null,
    elementTypes: ['C', 'N', 'O', 'H'],
    targetBandGap: null,
    guidanceStrength: 1.0,
  });

  // Check backend status on mount
  useEffect(() => {
    checkBackendStatus();
  }, []);

  const checkBackendStatus = async () => {
    try {
      const response = await fetch('/api/inference');
      if (response.ok) {
        const data = await response.json();
        setBackendStatus(data.status === 'online' ? 'online' : 'fallback');
      } else {
        setBackendStatus('fallback');
      }
    } catch {
      setBackendStatus('fallback');
    }
  };

  const generateMolecules = async () => {
    setIsGenerating(true);
    try {
      const response = await fetch('/api/inference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      
      if (!response.ok) throw new Error('Generation failed');
      
      const data = await response.json();
      
      if (data.molecules && data.molecules.length > 0) {
        setMolecules(data.molecules);
        setSelectedMolecule(data.molecules[0]);
      }
    } catch (error) {
      console.error('Generation error:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const downloadXYZ = (molecule: GeneratedMolecule) => {
    const blob = new Blob([molecule.xyz_content], { type: 'chemical/x-xyz' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${molecule.formula}_${molecule.id}.xyz`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const copyXYZ = (molecule: GeneratedMolecule) => {
    navigator.clipboard.writeText(molecule.xyz_content);
    setCopiedId(molecule.id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const saveToLibrary = async (molecule: GeneratedMolecule) => {
    try {
      const response = await fetch('http://localhost:8000/library/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          molecule_id: molecule.id,
          molecule_data: molecule,
          metadata: {
            temperature: params.temperature,
            num_diffusion_steps: params.numDiffusionSteps,
            guidance_strength: params.guidanceStrength,
          }
        }),
      });
      
      if (response.ok) {
        alert(`✓ ${molecule.formula} saved to library!`);
      } else {
        alert('Failed to save molecule');
      }
    } catch (error) {
      console.error('Failed to save molecule:', error);
      alert('Error saving molecule to library');
    }
  };

  const toggleElement = (element: string) => {
    setParams(prev => ({
      ...prev,
      elementTypes: prev.elementTypes.includes(element)
        ? prev.elementTypes.filter(e => e !== element)
        : [...prev.elementTypes, element]
    }));
  };

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="sticky top-0 z-40 bg-black/60 backdrop-blur-2xl border-b border-white/[0.06]">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/dashboard" className="p-2 rounded-lg hover:bg-white/5 transition-colors">
              <ArrowLeft className="w-5 h-5 text-gray-400" />
            </Link>
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-gradient-to-br from-purple-500 to-pink-600">
                <FlaskConical className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-semibold text-white">Molecule Generator</h1>
                <p className="text-xs text-gray-500">AI-powered structure generation</p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            {/* Backend status indicator */}
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full glass text-xs">
              <div className={`w-2 h-2 rounded-full ${
                backendStatus === 'online' ? 'bg-green-400 animate-pulse' : 
                backendStatus === 'fallback' ? 'bg-yellow-400' : 'bg-gray-400'
              }`} />
              <span className="text-gray-300">
                {backendStatus === 'online' ? 'Python Backend' : 
                 backendStatus === 'fallback' ? 'Mock Mode' : 'Checking...'}
              </span>
            </div>
            
            <button
              onClick={() => setShowSettings(!showSettings)}
              className={`p-2 rounded-lg transition-colors ${
                showSettings ? 'bg-white/10 text-white' : 'hover:bg-white/5 text-gray-400'
              }`}
            >
              <Settings2 className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Settings Panel */}
        {showSettings && (
          <div className="mb-8 p-6 rounded-2xl glass animate-fade-in">
            <h2 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
              <Settings2 className="w-5 h-5 text-purple-400" />
              Generation Settings
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {/* Number of Samples */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">Samples</label>
                <input
                  type="number"
                  min={1}
                  max={20}
                  value={params.numSamples}
                  onChange={e => setParams(p => ({ ...p, numSamples: parseInt(e.target.value) || 1 }))}
                  className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white focus:outline-none focus:border-purple-500/50"
                />
              </div>
              
              {/* Number of Atoms */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">Atoms per molecule</label>
                <input
                  type="number"
                  min={2}
                  max={50}
                  value={params.numAtoms}
                  onChange={e => setParams(p => ({ ...p, numAtoms: parseInt(e.target.value) || 12 }))}
                  className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white focus:outline-none focus:border-purple-500/50"
                />
              </div>
              
              {/* Temperature */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">Temperature</label>
                <input
                  type="number"
                  min={0.1}
                  max={2.0}
                  step={0.1}
                  value={params.temperature}
                  onChange={e => setParams(p => ({ ...p, temperature: parseFloat(e.target.value) || 1.0 }))}
                  className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white focus:outline-none focus:border-purple-500/50"
                />
              </div>
              
              {/* Diffusion Steps */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">Diffusion Steps</label>
                <input
                  type="number"
                  min={10}
                  max={1000}
                  step={10}
                  value={params.numDiffusionSteps}
                  onChange={e => setParams(p => ({ ...p, numDiffusionSteps: parseInt(e.target.value) || 100 }))}
                  className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white focus:outline-none focus:border-purple-500/50"
                />
              </div>
            </div>
            
            {/* Property Targeting */}
            <div className="mt-6 p-4 rounded-xl bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/20">
              <div className="flex items-center gap-2 mb-4">
                <Sparkles className="w-4 h-4 text-purple-400" />
                <h3 className="text-sm font-medium text-purple-300">Property-Guided Generation</h3>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Target Band Gap */}
                <div>
                  <label className="block text-sm text-gray-400 mb-2">
                    Target Band Gap (eV)
                    <span className="text-xs text-gray-500 ml-2">Optional</span>
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="number"
                      min={0}
                      max={10}
                      step={0.1}
                      value={params.targetBandGap ?? ''}
                      onChange={e => setParams(p => ({ 
                        ...p, 
                        targetBandGap: e.target.value ? parseFloat(e.target.value) : null 
                      }))}
                      placeholder="e.g., 2.5"
                      className="flex-1 px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white placeholder:text-gray-600 focus:outline-none focus:border-purple-500/50"
                    />
                    {params.targetBandGap !== null && (
                      <button
                        onClick={() => setParams(p => ({ ...p, targetBandGap: null }))}
                        className="px-3 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 transition-colors"
                        title="Clear target"
                      >
                        ✕
                      </button>
                    )}
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    0 = metal, 1-3 = semiconductor, &gt;3 = insulator
                  </p>
                </div>
                
                {/* Guidance Strength */}
                {params.targetBandGap !== null && (
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">
                      Guidance Strength
                    </label>
                    <input
                      type="range"
                      min={0.1}
                      max={3.0}
                      step={0.1}
                      value={params.guidanceStrength}
                      onChange={e => setParams(p => ({ ...p, guidanceStrength: parseFloat(e.target.value) }))}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>Weak (0.1)</span>
                      <span className="text-purple-400 font-medium">{params.guidanceStrength.toFixed(1)}</span>
                      <span>Strong (3.0)</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
            
            {/* Element Selection */}
            <div className="mt-6">
              <label className="block text-sm text-gray-400 mb-3">Elements</label>
              <div className="flex flex-wrap gap-2">
                {AVAILABLE_ELEMENTS.map(elem => (
                  <button
                    key={elem}
                    onClick={() => toggleElement(elem)}
                    className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                      params.elementTypes.includes(elem)
                        ? 'bg-purple-500/20 border border-purple-500/50 text-purple-300'
                        : 'bg-white/5 border border-white/10 text-gray-400 hover:text-white'
                    }`}
                    style={{
                      borderColor: params.elementTypes.includes(elem) 
                        ? ELEMENT_COLORS[elem] + '80' 
                        : undefined
                    }}
                  >
                    {elem}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Generate Button */}
        <div className="mb-8 flex items-center gap-4">
          <button
            onClick={generateMolecules}
            disabled={isGenerating || params.elementTypes.length === 0}
            className="flex items-center gap-3 px-8 py-4 rounded-2xl bg-gradient-to-r from-purple-600 to-pink-600 text-white font-medium hover:opacity-90 disabled:opacity-50 transition-all btn-shine"
          >
            {isGenerating ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5" />
                Generate Molecules
              </>
            )}
          </button>
          
          {molecules.length > 0 && (
            <button
              onClick={() => { setMolecules([]); setSelectedMolecule(null); }}
              className="p-3 rounded-xl glass hover:bg-white/10 transition-colors"
            >
              <RefreshCw className="w-5 h-5 text-gray-400" />
            </button>
          )}
          
          <span className="text-sm text-gray-500">
            {params.numSamples} sample{params.numSamples !== 1 ? 's' : ''} × {params.numAtoms} atoms
          </span>
        </div>

        {/* Results Grid */}
        {molecules.length > 0 ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Molecule List */}
            <div className="space-y-4">
              <h2 className="text-lg font-medium text-white flex items-center gap-2">
                <Atom className="w-5 h-5 text-purple-400" />
                Generated Molecules
                <span className="text-sm text-gray-500 font-normal">({molecules.length})</span>
              </h2>
              
              <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2">
                {molecules.map(mol => (
                  <div
                    key={mol.id}
                    onClick={() => setSelectedMolecule(mol)}
                    className={`p-4 rounded-xl cursor-pointer transition-all ${
                      selectedMolecule?.id === mol.id
                        ? 'glass-strong ring-2 ring-purple-500/50'
                        : 'glass hover:bg-white/[0.04]'
                    }`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h3 className="text-lg font-semibold text-white">{mol.formula}</h3>
                        <p className="text-xs text-gray-500">{mol.atoms.length} atoms • {mol.bonds.length} bonds</p>
                      </div>
                      <div className="flex gap-1">
                        <button
                          onClick={(e) => { e.stopPropagation(); copyXYZ(mol); }}
                          className="p-1.5 rounded-lg hover:bg-white/10 transition-colors"
                          title="Copy XYZ"
                        >
                          {copiedId === mol.id ? (
                            <Check className="w-4 h-4 text-green-400" />
                          ) : (
                            <Copy className="w-4 h-4 text-gray-400" />
                          )}
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); downloadXYZ(mol); }}
                          className="p-1.5 rounded-lg hover:bg-white/10 transition-colors"
                          title="Download XYZ"
                        >
                          <Download className="w-4 h-4 text-gray-400" />
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); saveToLibrary(mol); }}
                          className="p-1.5 rounded-lg hover:bg-white/10 transition-colors"
                          title="Save to Library"
                        >
                          <Star className="w-4 h-4 text-gray-400" />
                        </button>
                      </div>
                    </div>
                    
                    {/* Mini property badges */}
                    <div className="flex gap-2 flex-wrap">
                      <span className="px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 text-xs">
                        E: {mol.properties.formation_energy.toFixed(2)} eV
                      </span>
                      <span className="px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-400 text-xs">
                        Gap: {mol.properties.band_gap.toFixed(2)} eV
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 3D Viewer */}
            <div className="lg:col-span-2">
              {selectedMolecule ? (
                <Molecule3DViewer molecule={selectedMolecule} />
              ) : (
                <div className="h-[600px] rounded-2xl glass flex items-center justify-center">
                  <p className="text-gray-500">Select a molecule to view</p>
                </div>
              )}
            </div>
          </div>
        ) : (
          /* Empty State */
          <div className="py-20 text-center">
            <div className="inline-flex p-6 rounded-3xl glass mb-6">
              <FlaskConical className="w-16 h-16 text-purple-400/50" />
            </div>
            <h2 className="text-2xl font-semibold text-white mb-3">Ready to Generate</h2>
            <p className="text-gray-500 max-w-md mx-auto mb-6">
              Configure your parameters and click Generate to create new molecular structures 
              using the trained QCMD-ECS diffusion model.
            </p>
            <button
              onClick={() => setShowSettings(true)}
              className="text-purple-400 hover:text-purple-300 text-sm flex items-center gap-2 mx-auto"
            >
              <Settings2 className="w-4 h-4" />
              Adjust settings first
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

// 3D Molecule Viewer Component using Canvas
function Molecule3DViewer({ molecule }: { molecule: GeneratedMolecule }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [rotation, setRotation] = useState({ x: 0.3, y: 0.5 });
  const [zoom, setZoom] = useState(1);
  const [autoRotate, setAutoRotate] = useState(true);
  const [viewMode, setViewMode] = useState<'ball-stick' | 'space-fill' | 'wireframe'>('ball-stick');
  const isDragging = useRef(false);
  const lastMouse = useRef({ x: 0, y: 0 });
  const animationRef = useRef<number>();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    // Calculate molecule center and scale
    const atoms = molecule.atoms;
    const center = {
      x: atoms.reduce((sum, a) => sum + a.x, 0) / atoms.length,
      y: atoms.reduce((sum, a) => sum + a.y, 0) / atoms.length,
      z: atoms.reduce((sum, a) => sum + a.z, 0) / atoms.length,
    };

    // Find max extent for scaling
    let maxDist = 0;
    atoms.forEach(a => {
      const dist = Math.sqrt(
        Math.pow(a.x - center.x, 2) +
        Math.pow(a.y - center.y, 2) +
        Math.pow(a.z - center.z, 2)
      );
      maxDist = Math.max(maxDist, dist);
    });
    const scale = (Math.min(rect.width, rect.height) * 0.35 * zoom) / (maxDist || 1);

    const draw = () => {
      ctx.clearRect(0, 0, rect.width, rect.height);
      
      const cosX = Math.cos(rotation.x);
      const sinX = Math.sin(rotation.x);
      const cosY = Math.cos(rotation.y);
      const sinY = Math.sin(rotation.y);

      // Transform atoms to screen coordinates
      const transformed = atoms.map(atom => {
        let x = atom.x - center.x;
        let y = atom.y - center.y;
        let z = atom.z - center.z;

        // Rotate around Y axis
        const x1 = x * cosY - z * sinY;
        const z1 = x * sinY + z * cosY;
        x = x1;
        z = z1;

        // Rotate around X axis
        const y1 = y * cosX - z * sinX;
        const z2 = y * sinX + z * cosX;
        y = y1;
        z = z2;

        return {
          ...atom,
          screenX: rect.width / 2 + x * scale,
          screenY: rect.height / 2 + y * scale,
          depth: z,
        };
      });

      // Sort by depth for proper rendering
      const sorted = [...transformed].sort((a, b) => a.depth - b.depth);

      // Draw bonds first
      if (viewMode !== 'space-fill') {
        molecule.bonds.forEach(bond => {
          const a1 = transformed[bond.atom1];
          const a2 = transformed[bond.atom2];
          
          const gradient = ctx.createLinearGradient(a1.screenX, a1.screenY, a2.screenX, a2.screenY);
          gradient.addColorStop(0, a1.color + 'CC');
          gradient.addColorStop(1, a2.color + 'CC');
          
          ctx.beginPath();
          ctx.moveTo(a1.screenX, a1.screenY);
          ctx.lineTo(a2.screenX, a2.screenY);
          ctx.strokeStyle = gradient;
          ctx.lineWidth = viewMode === 'wireframe' ? 1 : (bond.order === 2 ? 4 : bond.order === 3 ? 6 : 3);
          ctx.lineCap = 'round';
          ctx.stroke();
          
          // Double/triple bond lines
          if (bond.order >= 2 && viewMode === 'ball-stick') {
            const dx = a2.screenX - a1.screenX;
            const dy = a2.screenY - a1.screenY;
            const len = Math.sqrt(dx*dx + dy*dy);
            const nx = -dy / len * 4;
            const ny = dx / len * 4;
            
            ctx.beginPath();
            ctx.moveTo(a1.screenX + nx, a1.screenY + ny);
            ctx.lineTo(a2.screenX + nx, a2.screenY + ny);
            ctx.strokeStyle = gradient;
            ctx.lineWidth = 2;
            ctx.stroke();
          }
        });
      }

      // Draw atoms
      sorted.forEach(atom => {
        const depthFactor = 0.5 + (atom.depth + maxDist) / (maxDist * 2) * 0.5;
        let radius = viewMode === 'space-fill' 
          ? atom.radius * scale * 0.8 
          : viewMode === 'wireframe' 
            ? 4 
            : atom.radius * scale * 0.4;
        radius *= depthFactor;

        // Glow
        const glow = ctx.createRadialGradient(
          atom.screenX, atom.screenY, 0,
          atom.screenX, atom.screenY, radius * 2
        );
        glow.addColorStop(0, atom.color + '40');
        glow.addColorStop(1, 'transparent');
        ctx.beginPath();
        ctx.arc(atom.screenX, atom.screenY, radius * 2, 0, Math.PI * 2);
        ctx.fillStyle = glow;
        ctx.fill();

        // Main sphere
        const gradient = ctx.createRadialGradient(
          atom.screenX - radius * 0.3,
          atom.screenY - radius * 0.3,
          0,
          atom.screenX,
          atom.screenY,
          radius
        );
        gradient.addColorStop(0, '#FFFFFF');
        gradient.addColorStop(0.3, atom.color);
        gradient.addColorStop(1, adjustColor(atom.color, -50));

        ctx.beginPath();
        ctx.arc(atom.screenX, atom.screenY, radius, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();

        // Element label
        if (viewMode !== 'wireframe' && radius > 15) {
          ctx.font = `bold ${Math.max(10, radius * 0.6)}px Inter, system-ui`;
          ctx.fillStyle = getLabelColor(atom.color);
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(atom.element, atom.screenX, atom.screenY);
        }
      });

      // Auto-rotate
      if (autoRotate && !isDragging.current) {
        setRotation(r => ({ ...r, y: r.y + 0.005 }));
      }

      animationRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [molecule, rotation, zoom, autoRotate, viewMode]);

  // Mouse handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    isDragging.current = true;
    lastMouse.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging.current) return;
    
    const dx = e.clientX - lastMouse.current.x;
    const dy = e.clientY - lastMouse.current.y;
    
    setRotation(r => ({
      x: r.x + dy * 0.01,
      y: r.y + dx * 0.01,
    }));
    
    lastMouse.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    setZoom(z => Math.max(0.5, Math.min(3, z - e.deltaY * 0.001)));
  };

  return (
    <div className="space-y-4">
      {/* Viewer Controls */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-medium text-white flex items-center gap-2">
          <Box className="w-5 h-5 text-purple-400" />
          3D Structure
        </h2>
        
        <div className="flex items-center gap-2">
          {/* View Mode */}
          <div className="flex rounded-lg glass overflow-hidden">
            {(['ball-stick', 'space-fill', 'wireframe'] as const).map(mode => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                className={`px-3 py-1.5 text-xs font-medium transition-colors ${
                  viewMode === mode ? 'bg-purple-500/20 text-purple-300' : 'text-gray-400 hover:text-white'
                }`}
              >
                {mode.split('-').map(w => w[0].toUpperCase() + w.slice(1)).join(' ')}
              </button>
            ))}
          </div>
          
          {/* Auto-rotate toggle */}
          <button
            onClick={() => setAutoRotate(!autoRotate)}
            className={`p-2 rounded-lg transition-colors ${
              autoRotate ? 'bg-purple-500/20 text-purple-300' : 'glass text-gray-400'
            }`}
            title="Auto-rotate"
          >
            <RefreshCw className={`w-4 h-4 ${autoRotate ? 'animate-spin' : ''}`} style={{ animationDuration: '3s' }} />
          </button>
          
          {/* Zoom controls */}
          <div className="flex items-center gap-1 glass rounded-lg px-2 py-1">
            <button onClick={() => setZoom(z => Math.max(0.5, z - 0.2))} className="p-1 text-gray-400 hover:text-white">
              −
            </button>
            <span className="text-xs text-gray-400 w-12 text-center">{Math.round(zoom * 100)}%</span>
            <button onClick={() => setZoom(z => Math.min(3, z + 0.2))} className="p-1 text-gray-400 hover:text-white">
              +
            </button>
          </div>
        </div>
      </div>

      {/* Canvas */}
      <div className="relative rounded-2xl glass overflow-hidden" style={{ height: '500px' }}>
        <canvas
          ref={canvasRef}
          className="w-full h-full cursor-grab active:cursor-grabbing"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
        />
        
        {/* Formula overlay */}
        <div className="absolute top-4 left-4 px-4 py-2 rounded-xl bg-black/60 backdrop-blur-lg">
          <span className="text-2xl font-bold text-white">{molecule.formula}</span>
        </div>
        
        {/* Controls hint */}
        <div className="absolute bottom-4 right-4 text-xs text-gray-500">
          Drag to rotate • Scroll to zoom
        </div>
      </div>

      {/* Properties Panel */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <PropertyCard 
          label="Formation Energy" 
          value={`${molecule.properties.formation_energy.toFixed(3)} eV`}
          icon={<Zap className="w-4 h-4" />}
          color="blue"
        />
        <PropertyCard 
          label="Band Gap" 
          value={`${molecule.properties.band_gap.toFixed(3)} eV`}
          icon={<Box className="w-4 h-4" />}
          color="purple"
          subtitle={
            molecule.properties.targeted_gap 
              ? `Target: ${molecule.properties.targeted_gap.toFixed(1)} eV | Error: ${Math.abs(molecule.properties.band_gap - molecule.properties.targeted_gap).toFixed(3)} eV`
              : undefined
          }
        />
        <PropertyCard 
          label="Dipole Moment" 
          value={`${molecule.properties.dipole_moment.toFixed(3)} D`}
          icon={<Atom className="w-4 h-4" />}
          color="pink"
        />
        <PropertyCard 
          label="Total Energy" 
          value={`${molecule.properties.total_energy.toFixed(3)} eV`}
          icon={<Sparkles className="w-4 h-4" />}
          color="cyan"
        />
      </div>

      {/* XYZ Content */}
      <details className="glass rounded-xl">
        <summary className="px-4 py-3 cursor-pointer text-sm text-gray-400 hover:text-white flex items-center justify-between">
          <span className="flex items-center gap-2">
            <ChevronDown className="w-4 h-4" />
            XYZ Coordinates
          </span>
        </summary>
        <pre className="px-4 pb-4 text-xs text-gray-500 font-mono overflow-x-auto">
          {molecule.xyz_content}
        </pre>
      </details>
    </div>
  );
}

function PropertyCard({ 
  label, 
  value, 
  icon, 
  color,
  subtitle
}: { 
  label: string; 
  value: string; 
  icon: React.ReactNode;
  color: string;
  subtitle?: string;
}) {
  const colorClasses: Record<string, string> = {
    blue: 'from-blue-500/20 to-blue-600/10 text-blue-400',
    purple: 'from-purple-500/20 to-purple-600/10 text-purple-400',
    pink: 'from-pink-500/20 to-pink-600/10 text-pink-400',
    cyan: 'from-cyan-500/20 to-cyan-600/10 text-cyan-400',
  };
  
  return (
    <div className="p-4 rounded-xl glass">
      <div className={`inline-flex p-2 rounded-lg bg-gradient-to-br ${colorClasses[color]} mb-2`}>
        {icon}
      </div>
      <p className="text-lg font-semibold text-white">{value}</p>
      <p className="text-xs text-gray-500">{label}</p>
      {subtitle && (
        <p className="text-[10px] text-gray-600 mt-1">{subtitle}</p>
      )}
    </div>
  );
}

// Utility functions
function adjustColor(hex: string, amount: number): string {
  const num = parseInt(hex.replace('#', ''), 16);
  const r = Math.min(255, Math.max(0, (num >> 16) + amount));
  const g = Math.min(255, Math.max(0, ((num >> 8) & 0x00FF) + amount));
  const b = Math.min(255, Math.max(0, (num & 0x0000FF) + amount));
  return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;
}

function getLabelColor(bgColor: string): string {
  const num = parseInt(bgColor.replace('#', ''), 16);
  const r = num >> 16;
  const g = (num >> 8) & 0x00FF;
  const b = num & 0x0000FF;
  const brightness = (r * 299 + g * 587 + b * 114) / 1000;
  return brightness > 128 ? '#000000' : '#FFFFFF';
}
