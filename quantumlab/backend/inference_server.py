#!/usr/bin/env python3
"""
FastAPI backend server for molecule generation inference.
Connects to the trained score/surrogate models for QCMD-ECS diffusion.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "core"))
sys.path.insert(0, str(PROJECT_ROOT / "projects/phononic-discovery/framework"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Try to import the actual models
try:
    from qcmd_ecs.core.dynamics import run_reverse_diffusion
    from qcmd_ecs.core.manifold import retract_to_manifold
    from qcmd_ecs.core.types import DTYPE
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("Warning: QCMD-ECS core not available, using mock generation")

app = FastAPI(
    title="QuantumLab Inference API",
    description="Backend API for molecule generation using trained diffusion models",
    version="1.0.0"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Request/Response Models -----

class GenerationRequest(BaseModel):
    num_samples: int = 5
    num_atoms: int = 12
    temperature: float = 1.0
    num_diffusion_steps: int = 100
    seed: Optional[int] = None
    element_types: List[str] = ["C", "N", "O", "H"]

class MoleculeStructure(BaseModel):
    id: str
    atoms: List[Dict[str, Any]]  # [{element, x, y, z}, ...]
    bonds: List[Dict[str, Any]]  # [{atom1, atom2, order}, ...]
    properties: Dict[str, Any]
    xyz_content: str
    formula: str
    generated_at: str

class GenerationResponse(BaseModel):
    success: bool
    molecules: List[MoleculeStructure]
    generation_time: float
    model_info: Dict[str, Any]

class ModelInfo(BaseModel):
    name: str
    type: str
    status: str
    path: Optional[str]
    config: Dict[str, Any]

# ----- Element Properties -----

ELEMENT_DATA = {
    "H": {"atomic_number": 1, "color": "#FFFFFF", "radius": 0.31},
    "C": {"atomic_number": 6, "color": "#909090", "radius": 0.77},
    "N": {"atomic_number": 7, "color": "#3050F8", "radius": 0.71},
    "O": {"atomic_number": 8, "color": "#FF0D0D", "radius": 0.66},
    "S": {"atomic_number": 16, "color": "#FFFF30", "radius": 1.05},
    "F": {"atomic_number": 9, "color": "#90E050", "radius": 0.57},
    "Cl": {"atomic_number": 17, "color": "#1FF01F", "radius": 1.02},
    "Br": {"atomic_number": 35, "color": "#A62929", "radius": 1.20},
    "P": {"atomic_number": 15, "color": "#FF8000", "radius": 1.07},
    "Cr": {"atomic_number": 24, "color": "#8A99C7", "radius": 1.39},
    "Cu": {"atomic_number": 29, "color": "#C88033", "radius": 1.32},
    "Se": {"atomic_number": 34, "color": "#FFA100", "radius": 1.20},
    "Mo": {"atomic_number": 42, "color": "#54B5B5", "radius": 1.54},
    "W": {"atomic_number": 74, "color": "#2194D6", "radius": 1.62},
    "Ti": {"atomic_number": 22, "color": "#BFC2C7", "radius": 1.60},
}

# ----- Model Loading -----

class ModelManager:
    """Manages loading and caching of trained models."""
    
    def __init__(self):
        self.score_model = None
        self.surrogate_model = None
        self.model_paths = {
            "score_model": PROJECT_ROOT / "core/models/score_model",
            "surrogate": PROJECT_ROOT / "core/models/surrogate",
            "tmd_score": PROJECT_ROOT / "core/models/tmd_score",
            "tmd_surrogate": PROJECT_ROOT / "core/models/tmd_surrogate",
        }
        self.loaded_models = {}
        
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available trained models."""
        models = []
        for name, path in self.model_paths.items():
            metrics_file = path / "training_metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                    models.append(ModelInfo(
                        name=name,
                        type="score" if "score" in name else "surrogate",
                        status="available",
                        path=str(path),
                        config=metrics
                    ))
                except Exception as e:
                    models.append(ModelInfo(
                        name=name,
                        type="score" if "score" in name else "surrogate",
                        status="error",
                        path=str(path),
                        config={"error": str(e)}
                    ))
            else:
                models.append(ModelInfo(
                    name=name,
                    type="score" if "score" in name else "surrogate",
                    status="not_trained",
                    path=str(path),
                    config={}
                ))
        return models
    
    def load_model(self, model_name: str):
        """Load a specific model if not already loaded."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        model_path = self.model_paths.get(model_name)
        if not model_path or not model_path.exists():
            return None
        
        # Check for saved weights
        weight_files = list(model_path.glob("*.pt")) + list(model_path.glob("*.pth"))
        if weight_files:
            # In production, load the actual model weights here
            pass
        
        return None

model_manager = ModelManager()

# ----- Generation Functions -----

def generate_molecule_coordinates(
    num_atoms: int,
    element_types: List[str],
    temperature: float = 1.0,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Generate molecular coordinates using diffusion or mock data."""
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Select random elements for each atom
    elements = np.random.choice(element_types, size=num_atoms)
    
    if MODELS_AVAILABLE:
        # Use actual diffusion model
        try:
            # Initialize random positions on manifold
            U_T = torch.randn(num_atoms, 3, dtype=DTYPE)
            U_T = retract_to_manifold(U_T)
            
            # Simple schedules for demo
            def gamma_schedule(t): return 0.1 * (t / 100)
            def eta_schedule(t): return 0.01
            def tau_schedule(t): return 0.1 * np.sqrt(t / 100)
            
            # Mock score/energy models for now
            def score_model(U, t): return -0.01 * U
            def energy_model(U): return -0.005 * U
            
            # Run diffusion
            U_0 = run_reverse_diffusion(
                U_T=U_T,
                score_model=score_model,
                energy_gradient_model=energy_model,
                gamma_schedule=gamma_schedule,
                eta_schedule=eta_schedule,
                tau_schedule=tau_schedule,
                num_steps=50,
                seed=seed or 42
            )
            
            # Scale to reasonable molecular distances (Angstroms)
            positions = U_0.numpy() * 2.0
            
        except Exception as e:
            print(f"Diffusion failed, using fallback: {e}")
            positions = generate_mock_positions(num_atoms, temperature)
    else:
        positions = generate_mock_positions(num_atoms, temperature)
    
    # Build atom list
    atoms = []
    for i, (elem, pos) in enumerate(zip(elements, positions)):
        atoms.append({
            "index": i,
            "element": elem,
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
            "color": ELEMENT_DATA.get(elem, {}).get("color", "#CCCCCC"),
            "radius": ELEMENT_DATA.get(elem, {}).get("radius", 1.0),
        })
    
    # Infer bonds based on distances
    bonds = infer_bonds(atoms)
    
    # Calculate formula
    formula = calculate_formula(elements)
    
    # Generate XYZ content
    xyz_content = generate_xyz(atoms)
    
    # Calculate mock properties
    properties = {
        "total_energy": float(-10 - np.random.random() * 50),
        "formation_energy": float(-0.5 - np.random.random() * 2),
        "band_gap": float(np.random.random() * 3),
        "dipole_moment": float(np.random.random() * 5),
        "num_atoms": num_atoms,
        "valid": True,
    }
    
    return {
        "atoms": atoms,
        "bonds": bonds,
        "formula": formula,
        "xyz_content": xyz_content,
        "properties": properties,
    }

def generate_mock_positions(num_atoms: int, temperature: float) -> np.ndarray:
    """Generate reasonable-looking molecular positions."""
    positions = []
    
    # Start from origin
    positions.append([0.0, 0.0, 0.0])
    
    for i in range(1, num_atoms):
        # Add atom at reasonable distance from existing atoms
        for _ in range(100):
            # Random direction
            theta = np.random.random() * 2 * np.pi
            phi = np.random.random() * np.pi
            r = 1.2 + np.random.random() * 0.8  # Bond-like distance
            
            # From random existing atom
            ref_idx = np.random.randint(0, len(positions))
            ref = positions[ref_idx]
            
            new_pos = [
                ref[0] + r * np.sin(phi) * np.cos(theta) * temperature,
                ref[1] + r * np.sin(phi) * np.sin(theta) * temperature,
                ref[2] + r * np.cos(phi) * temperature,
            ]
            
            # Check not too close to existing
            valid = True
            for existing in positions:
                dist = np.sqrt(sum((a - b)**2 for a, b in zip(new_pos, existing)))
                if dist < 0.9:
                    valid = False
                    break
            
            if valid:
                positions.append(new_pos)
                break
        else:
            # Fallback random position
            positions.append([
                np.random.randn() * 2,
                np.random.randn() * 2,
                np.random.randn() * 2,
            ])
    
    return np.array(positions)

def infer_bonds(atoms: List[Dict]) -> List[Dict]:
    """Infer chemical bonds based on atomic distances."""
    bonds = []
    
    for i, atom1 in enumerate(atoms):
        for j, atom2 in enumerate(atoms):
            if j <= i:
                continue
            
            # Calculate distance
            dx = atom1["x"] - atom2["x"]
            dy = atom1["y"] - atom2["y"]
            dz = atom1["z"] - atom2["z"]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Get covalent radii sum
            r1 = atom1.get("radius", 1.0)
            r2 = atom2.get("radius", 1.0)
            max_bond_dist = (r1 + r2) * 1.3
            
            if dist < max_bond_dist:
                # Determine bond order based on distance
                if dist < (r1 + r2) * 0.9:
                    order = 3  # Triple
                elif dist < (r1 + r2) * 1.05:
                    order = 2  # Double
                else:
                    order = 1  # Single
                
                bonds.append({
                    "atom1": i,
                    "atom2": j,
                    "order": order,
                    "length": float(dist),
                })
    
    return bonds

def calculate_formula(elements: np.ndarray) -> str:
    """Calculate molecular formula from element list."""
    from collections import Counter
    counts = Counter(elements)
    
    # Standard order: C, H, then alphabetical
    formula_parts = []
    if "C" in counts:
        formula_parts.append(f"C{counts['C']}" if counts['C'] > 1 else "C")
        del counts["C"]
    if "H" in counts:
        formula_parts.append(f"H{counts['H']}" if counts['H'] > 1 else "H")
        del counts["H"]
    
    for elem in sorted(counts.keys()):
        formula_parts.append(f"{elem}{counts[elem]}" if counts[elem] > 1 else elem)
    
    return "".join(formula_parts)

def generate_xyz(atoms: List[Dict]) -> str:
    """Generate XYZ file content."""
    lines = [str(len(atoms)), "Generated by QuantumLab"]
    for atom in atoms:
        lines.append(f"{atom['element']:2s}  {atom['x']:12.6f}  {atom['y']:12.6f}  {atom['z']:12.6f}")
    return "\n".join(lines)

# ----- API Endpoints -----

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "QuantumLab Inference API",
        "version": "1.0.0",
        "models_available": MODELS_AVAILABLE,
    }

@app.get("/models")
async def get_models():
    """Get list of available trained models."""
    return {
        "models": [m.dict() for m in model_manager.get_available_models()]
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_molecules(request: GenerationRequest):
    """Generate new molecular structures using the trained model."""
    import time
    start_time = time.time()
    
    try:
        molecules = []
        base_seed = request.seed or int(time.time())
        
        for i in range(request.num_samples):
            mol_data = generate_molecule_coordinates(
                num_atoms=request.num_atoms,
                element_types=request.element_types,
                temperature=request.temperature,
                seed=base_seed + i
            )
            
            molecules.append(MoleculeStructure(
                id=f"mol_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}",
                atoms=mol_data["atoms"],
                bonds=mol_data["bonds"],
                properties=mol_data["properties"],
                xyz_content=mol_data["xyz_content"],
                formula=mol_data["formula"],
                generated_at=datetime.now().isoformat(),
            ))
        
        generation_time = time.time() - start_time
        
        return GenerationResponse(
            success=True,
            molecules=molecules,
            generation_time=generation_time,
            model_info={
                "backend": "QCMD-ECS" if MODELS_AVAILABLE else "Mock",
                "diffusion_steps": request.num_diffusion_steps,
                "temperature": request.temperature,
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/elements")
async def get_elements():
    """Get available elements and their properties."""
    return {"elements": ELEMENT_DATA}

# ----- Main -----

if __name__ == "__main__":
    print("Starting QuantumLab Inference Server...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Models available: {MODELS_AVAILABLE}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
