'use client';

import { useRef, useEffect, useState } from 'react';
import { RotateCcw, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';

interface Atom {
  element: string;
  x: number;
  y: number;
  z: number;
}

interface MoleculeViewerProps {
  atoms: Atom[];
  className?: string;
}

const elementColors: Record<string, string> = {
  H: '#ffffff',
  C: '#909090',
  N: '#3050f8',
  O: '#ff0d0d',
  F: '#90e050',
  S: '#ffff30',
  P: '#ff8000',
  Cl: '#1ff01f',
  Br: '#a62929',
  I: '#940094',
  Fe: '#e06633',
  Cu: '#c88033',
  Zn: '#7d80b0',
  Se: '#ffa100',
  Cr: '#8a99c7',
  Mo: '#54b5b5',
  default: '#ff1493',
};

const elementSizes: Record<string, number> = {
  H: 0.31,
  C: 0.77,
  N: 0.71,
  O: 0.66,
  S: 1.05,
  default: 0.8,
};

export function MoleculeViewer({ atoms, className = '' }: MoleculeViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [rotation, setRotation] = useState({ x: 0.5, y: 0.5 });
  const [zoom, setZoom] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const [lastPos, setLastPos] = useState({ x: 0, y: 0 });
  const [autoRotate, setAutoRotate] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationId: number;
    let currentRotation = rotation;

    const render = () => {
      if (!ctx || !canvas) return;

      const width = canvas.width;
      const height = canvas.height;
      const centerX = width / 2;
      const centerY = height / 2;

      // Clear with gradient background
      const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, Math.max(width, height) / 2);
      gradient.addColorStop(0, '#1a1a2e');
      gradient.addColorStop(1, '#0a0a0f');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);

      if (atoms.length === 0) {
        ctx.fillStyle = 'rgba(255,255,255,0.3)';
        ctx.font = '14px system-ui';
        ctx.textAlign = 'center';
        ctx.fillText('No structure data', centerX, centerY);
        return;
      }

      // Calculate center of mass
      const com = atoms.reduce(
        (acc, atom) => ({ x: acc.x + atom.x, y: acc.y + atom.y, z: acc.z + atom.z }),
        { x: 0, y: 0, z: 0 }
      );
      com.x /= atoms.length;
      com.y /= atoms.length;
      com.z /= atoms.length;

      // Project and sort atoms
      const scale = Math.min(width, height) * 0.15 * zoom;
      const cosX = Math.cos(currentRotation.x);
      const sinX = Math.sin(currentRotation.x);
      const cosY = Math.cos(currentRotation.y);
      const sinY = Math.sin(currentRotation.y);

      const projectedAtoms = atoms.map((atom) => {
        let x = atom.x - com.x;
        let y = atom.y - com.y;
        let z = atom.z - com.z;

        // Rotate around Y axis
        const tempX = x * cosY - z * sinY;
        z = x * sinY + z * cosY;
        x = tempX;

        // Rotate around X axis
        const tempY = y * cosX - z * sinX;
        z = y * sinX + z * cosX;
        y = tempY;

        return {
          element: atom.element,
          screenX: centerX + x * scale,
          screenY: centerY + y * scale,
          z,
          size: (elementSizes[atom.element] || elementSizes.default) * scale * 0.5,
        };
      });

      // Sort by z for proper depth rendering
      projectedAtoms.sort((a, b) => a.z - b.z);

      // Draw atoms
      projectedAtoms.forEach((atom) => {
        const color = elementColors[atom.element] || elementColors.default;
        const depthFactor = 0.5 + (atom.z / 10 + 0.5) * 0.5;

        // Glow effect
        const glowGradient = ctx.createRadialGradient(
          atom.screenX, atom.screenY, 0,
          atom.screenX, atom.screenY, atom.size * 2
        );
        glowGradient.addColorStop(0, color + '40');
        glowGradient.addColorStop(1, 'transparent');
        ctx.fillStyle = glowGradient;
        ctx.beginPath();
        ctx.arc(atom.screenX, atom.screenY, atom.size * 2, 0, Math.PI * 2);
        ctx.fill();

        // Main atom sphere
        const sphereGradient = ctx.createRadialGradient(
          atom.screenX - atom.size * 0.3,
          atom.screenY - atom.size * 0.3,
          0,
          atom.screenX,
          atom.screenY,
          atom.size
        );
        sphereGradient.addColorStop(0, '#ffffff');
        sphereGradient.addColorStop(0.3, color);
        sphereGradient.addColorStop(1, shadeColor(color, -30));

        ctx.fillStyle = sphereGradient;
        ctx.globalAlpha = depthFactor;
        ctx.beginPath();
        ctx.arc(atom.screenX, atom.screenY, atom.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
      });

      if (autoRotate && !isDragging) {
        currentRotation.y += 0.01;
        setRotation({ ...currentRotation });
      }

      animationId = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationId) cancelAnimationFrame(animationId);
    };
  }, [atoms, rotation, zoom, isDragging, autoRotate]);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setAutoRotate(false);
    setLastPos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    const dx = e.clientX - lastPos.x;
    const dy = e.clientY - lastPos.y;
    setRotation((prev) => ({
      x: prev.x + dy * 0.01,
      y: prev.y + dx * 0.01,
    }));
    setLastPos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    setZoom((prev) => Math.max(0.5, Math.min(3, prev - e.deltaY * 0.001)));
  };

  return (
    <div className={`relative group ${className}`}>
      <canvas
        ref={canvasRef}
        width={400}
        height={300}
        className="w-full h-full rounded-xl cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      />
      
      {/* Controls */}
      <div className="absolute bottom-3 right-3 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={() => setAutoRotate(!autoRotate)}
          className={`p-2 rounded-lg backdrop-blur-sm transition-colors ${
            autoRotate ? 'bg-blue-500/20 text-blue-400' : 'bg-white/10 text-white/60 hover:text-white'
          }`}
          title="Toggle auto-rotate"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
        <button
          onClick={() => setZoom((z) => Math.min(3, z + 0.2))}
          className="p-2 rounded-lg bg-white/10 text-white/60 hover:text-white backdrop-blur-sm transition-colors"
          title="Zoom in"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={() => setZoom((z) => Math.max(0.5, z - 0.2))}
          className="p-2 rounded-lg bg-white/10 text-white/60 hover:text-white backdrop-blur-sm transition-colors"
          title="Zoom out"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          onClick={() => {
            setRotation({ x: 0.5, y: 0.5 });
            setZoom(1);
          }}
          className="p-2 rounded-lg bg-white/10 text-white/60 hover:text-white backdrop-blur-sm transition-colors"
          title="Reset view"
        >
          <Maximize2 className="w-4 h-4" />
        </button>
      </div>

      {/* Legend */}
      {atoms.length > 0 && (
        <div className="absolute top-3 left-3 flex flex-wrap gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
          {Array.from(new Set(atoms.map((a) => a.element))).map((el) => (
            <div
              key={el}
              className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-black/50 backdrop-blur-sm text-xs"
            >
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: elementColors[el] || elementColors.default }}
              />
              <span className="text-white/80">{el}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function shadeColor(color: string, percent: number): string {
  const num = parseInt(color.replace('#', ''), 16);
  const amt = Math.round(2.55 * percent);
  const R = Math.max(0, Math.min(255, (num >> 16) + amt));
  const G = Math.max(0, Math.min(255, ((num >> 8) & 0x00ff) + amt));
  const B = Math.max(0, Math.min(255, (num & 0x0000ff) + amt));
  return `#${((1 << 24) | (R << 16) | (G << 8) | B).toString(16).slice(1)}`;
}
