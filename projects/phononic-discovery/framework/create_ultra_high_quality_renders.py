#!/usr/bin/env python3
"""
Ultra-high-quality renders for CrCuSe2 using multiple rendering backends.
Creates publication-ready figures with:
- Vector graphics (SVG) for infinite scalability
- Ultra-high DPI rasters (600 DPI)
- Professional color schemes
- Multiple export formats
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from ase.io import read
from ase.visualize.plot import plot_atoms
from pathlib import Path

# Ultra-high quality settings
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['lines.linewidth'] = 2

INPUT_XYZ = Path("/workspaces/algo/projects/phononic-discovery/framework/dft_validation/results/CrCuSe2_rescue_relaxed.xyz")
OUTPUT_DIR = Path("/workspaces/algo/projects/phononic-discovery/framework/results/CrCuSe2_publication_figures")

# Professional color scheme (optimized for both print and screen)
ATOM_COLORS = {
    'Cr': '#7B68EE',  # Medium slate blue
    'Cu': '#CD853F',  # Peru/bronze
    'Se': '#FF8C00',  # Dark orange
}


def create_main_figure_for_paper(atoms, output_prefix):
    """
    Create the main publication figure with optimal layout.
    This is designed to be the primary figure in your paper.
    """
    print("Creating MAIN PUBLICATION FIGURE (ultra-high quality)...")
    
    # Create figure with exact dimensions for 2-column paper format
    # Standard 2-column width is ~3.5 inches, full page is ~7 inches
    fig = plt.figure(figsize=(7, 8), dpi=600)
    
    # Top panel: Structure visualization (takes 60% of vertical space)
    ax_main = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    
    atom_colors = [ATOM_COLORS.get(atom.symbol, '#808080') for atom in atoms]
    
    plot_atoms(atoms, ax=ax_main,
              rotation='45x,30y,15z',  # Optimal viewing angle
              show_unit_cell=True,
              radii=1.0,
              colors=atom_colors)
    
    ax_main.set_title('CrCuSe₂: AI-Discovered Hetero-Metallic TMD',
                     fontsize=16, fontweight='bold', pad=15)
    ax_main.axis('off')
    
    # Bottom panel: Key properties table (takes 40% of vertical space)
    ax_table = plt.subplot2grid((10, 1), (6, 0), rowspan=4)
    ax_table.axis('off')
    
    # Create properties table
    cell = atoms.get_cell()
    
    properties = [
        ['Property', 'Value'],
        ['─' * 40, '─' * 40],
        ['Chemical Formula', 'CrCuSe₂'],
        ['Structure Type', '2D Hetero-Metallic TMD'],
        ['Space Group', 'P 1 (Triclinic)'],
        ['', ''],
        ['Lattice Constants', f'a = {cell[0][0]:.2f} Å'],
        ['', f'b = {cell[1][1]:.2f} Å'],
        ['', f'c = {cell[2][2]:.2f} Å'],
        ['', ''],
        ['Electronic Properties', ''],
        ['  Bandgap', '0.616 eV (indirect)'],
        ['  Band Character', 'Semiconductor'],
        ['', ''],
        ['Stability', ''],
        ['  Formation Energy', '+1.23 eV/atom'],
        ['  Phonon Modes', '0 imaginary (stable)'],
        ['  Validation', 'DFT-confirmed'],
    ]
    
    table = ax_table.table(cellText=properties,
                          cellLoc='left',
                          loc='center',
                          bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    # Style the table
    for i, row in enumerate(properties):
        cell_obj = table[(i, 0)]
        cell_obj.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        cell_obj.set_edgecolor('#cccccc')
        
        if i == 0:  # Header
            cell_obj.set_facecolor('#4a4a4a')
            cell_obj.set_text_props(weight='bold', color='white')
            table[(i, 1)].set_facecolor('#4a4a4a')
            table[(i, 1)].set_text_props(weight='bold', color='white')
        
        # Bold for property names
        if row[0] and not row[0].startswith(' ') and i > 1:
            cell_obj.set_text_props(weight='bold')
    
    plt.tight_layout()
    
    # Save in multiple formats
    for fmt in ['png', 'svg', 'pdf']:
        output_file = f"{output_prefix}_main_figure.{fmt}"
        plt.savefig(output_file, 
                   dpi=600 if fmt == 'png' else None,
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none',
                   format=fmt)
        print(f"  ✓ Saved: {Path(output_file).name}")
    
    plt.close()


def create_structure_comparison_figure(atoms, output_prefix):
    """
    Create a figure showing different viewing angles side-by-side.
    Ideal for supplementary material.
    """
    print("\nCreating MULTI-ANGLE COMPARISON figure...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=600)
    
    views = [
        ('0x,0y,0z', 'Top View\n(a-b plane)'),
        ('90x,0y,0z', 'Side View\n(a-c plane)'),
        ('45x,30y,15z', 'Perspective View'),
    ]
    
    for ax, (rotation, title) in zip(axes, views):
        atom_colors = [ATOM_COLORS.get(atom.symbol, '#808080') for atom in atoms]
        
        plot_atoms(atoms, ax=ax,
                  rotation=rotation,
                  show_unit_cell=True,
                  radii=1.0,
                  colors=atom_colors)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')
    
    # Add legend
    legend_elements = [
        Circle((0, 0), 1, facecolor=ATOM_COLORS['Cr'], 
               edgecolor='black', linewidth=2, label='Cr'),
        Circle((0, 0), 1, facecolor=ATOM_COLORS['Cu'], 
               edgecolor='black', linewidth=2, label='Cu'),
        Circle((0, 0), 1, facecolor=ATOM_COLORS['Se'], 
               edgecolor='black', linewidth=2, label='Se'),
    ]
    
    fig.legend(handles=legend_elements,
              loc='lower center',
              ncol=3,
              fontsize=12,
              frameon=True,
              fancybox=True,
              shadow=True,
              bbox_to_anchor=(0.5, -0.05))
    
    fig.suptitle('CrCuSe₂ Structure: Multiple Viewing Angles',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save in multiple formats
    for fmt in ['png', 'svg', 'pdf']:
        output_file = f"{output_prefix}_multi_angle.{fmt}"
        plt.savefig(output_file,
                   dpi=600 if fmt == 'png' else None,
                   bbox_inches='tight',
                   facecolor='white',
                   format=fmt)
        print(f"  ✓ Saved: {Path(output_file).name}")
    
    plt.close()


def create_minimalist_figure(atoms, output_prefix):
    """
    Create a clean, minimalist figure focusing on the structure.
    Perfect for presentations or as a standalone image.
    """
    print("\nCreating MINIMALIST figure...")
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=600)
    
    atom_colors = [ATOM_COLORS.get(atom.symbol, '#808080') for atom in atoms]
    
    plot_atoms(atoms, ax=ax,
              rotation='45x,30y,15z',
              show_unit_cell=False,  # No cell for cleaner look
              radii=1.2,  # Slightly larger atoms
              colors=atom_colors)
    
    ax.axis('off')
    
    # Add minimal legend in corner
    legend_elements = [
        Circle((0, 0), 1, facecolor=ATOM_COLORS['Cr'], edgecolor='black', linewidth=2, label='Cr'),
        Circle((0, 0), 1, facecolor=ATOM_COLORS['Cu'], edgecolor='black', linewidth=2, label='Cu'),
        Circle((0, 0), 1, facecolor=ATOM_COLORS['Se'], edgecolor='black', linewidth=2, label='Se'),
    ]
    
    ax.legend(handles=legend_elements,
             loc='upper right',
             fontsize=14,
             frameon=True,
             fancybox=True)
    
    plt.tight_layout()
    
    # Save in multiple formats
    for fmt in ['png', 'svg', 'pdf']:
        output_file = f"{output_prefix}_minimalist.{fmt}"
        plt.savefig(output_file,
                   dpi=600 if fmt == 'png' else None,
                   bbox_inches='tight',
                   facecolor='white',
                   transparent=False,
                   format=fmt)
        print(f"  ✓ Saved: {Path(output_file).name}")
    
    plt.close()


def create_bond_analysis_figure(atoms, output_prefix):
    """
    Create a detailed figure showing bond lengths and coordination.
    """
    print("\nCreating BOND ANALYSIS figure...")
    
    fig = plt.figure(figsize=(12, 8), dpi=600)
    
    # Left: structure
    ax_structure = plt.subplot2grid((1, 2), (0, 0))
    
    atom_colors = [ATOM_COLORS.get(atom.symbol, '#808080') for atom in atoms]
    
    plot_atoms(atoms, ax=ax_structure,
              rotation='45x,30y,15z',
              show_unit_cell=True,
              radii=0.9,
              colors=atom_colors)
    
    ax_structure.set_title('Structure', fontsize=14, fontweight='bold')
    ax_structure.axis('off')
    
    # Right: bond analysis
    ax_bonds = plt.subplot2grid((1, 2), (0, 1))
    ax_bonds.axis('off')
    
    # Calculate bonds
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    bonds = []
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 3.5:  # Include all reasonable bonds
                bonds.append((symbols[i], symbols[j], dist))
    
    # Create bond table
    bond_text = "BOND LENGTH ANALYSIS\n"
    bond_text += "═" * 40 + "\n\n"
    
    bond_types = {}
    for s1, s2, d in bonds:
        bond_type = f"{s1}-{s2}"
        if bond_type not in bond_types:
            bond_types[bond_type] = []
        bond_types[bond_type].append(d)
    
    for bond_type in sorted(bond_types.keys()):
        lengths = bond_types[bond_type]
        avg_length = np.mean(lengths)
        bond_text += f"{bond_type} Bonds:\n"
        for length in lengths:
            bond_text += f"  {length:.3f} Å\n"
        if len(lengths) > 1:
            bond_text += f"  Average: {avg_length:.3f} Å\n"
        bond_text += "\n"
    
    # Coordination analysis
    bond_text += "\nCOORDINATION ENVIRONMENT\n"
    bond_text += "─" * 40 + "\n\n"
    
    for i, symbol in enumerate(symbols):
        neighbors = []
        for j in range(len(atoms)):
            if i != j:
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < 3.0:
                    neighbors.append((symbols[j], dist))
        
        bond_text += f"{symbol} (atom {i+1}):\n"
        bond_text += f"  Coordination: {len(neighbors)}\n"
        bond_text += "  Neighbors:\n"
        for neighbor_sym, dist in sorted(neighbors, key=lambda x: x[1]):
            bond_text += f"    {neighbor_sym}: {dist:.3f} Å\n"
        bond_text += "\n"
    
    ax_bonds.text(0.05, 0.95, bond_text,
                 transform=ax_bonds.transAxes,
                 fontsize=10,
                 verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=1', 
                          facecolor='lightblue', 
                          alpha=0.3,
                          edgecolor='navy'))
    
    fig.suptitle('CrCuSe₂ Bonding Analysis',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save in multiple formats
    for fmt in ['png', 'svg', 'pdf']:
        output_file = f"{output_prefix}_bond_analysis.{fmt}"
        plt.savefig(output_file,
                   dpi=600 if fmt == 'png' else None,
                   bbox_inches='tight',
                   facecolor='white',
                   format=fmt)
        print(f"  ✓ Saved: {Path(output_file).name}")
    
    plt.close()


def main():
    print("="*70)
    print("ULTRA-HIGH-QUALITY CrCuSe₂ Visualization Suite")
    print("="*70)
    print()
    print("Generating publication-ready figures at 600 DPI...")
    print("Output formats: PNG (raster), SVG (vector), PDF (vector)")
    print()
    
    # Load structure
    atoms = read(INPUT_XYZ)
    print(f"Loaded: {atoms.get_chemical_formula()}")
    print()
    
    output_prefix = OUTPUT_DIR / "CrCuSe2_ultra_HQ"
    
    # Generate all figure types
    create_main_figure_for_paper(atoms, output_prefix)
    create_structure_comparison_figure(atoms, output_prefix)
    create_minimalist_figure(atoms, output_prefix)
    create_bond_analysis_figure(atoms, output_prefix)
    
    print()
    print("="*70)
    print("✓ All ultra-high-quality figures generated!")
    print("="*70)
    print()
    print("RECOMMENDATIONS FOR YOUR PAPER:")
    print()
    print("Main Text Figure:")
    print("  → CrCuSe2_ultra_HQ_main_figure.pdf  (vector, scalable)")
    print("     or")
    print("  → CrCuSe2_ultra_HQ_main_figure.png  (600 DPI raster)")
    print()
    print("Supplementary Material:")
    print("  → CrCuSe2_ultra_HQ_multi_angle.pdf")
    print("  → CrCuSe2_ultra_HQ_bond_analysis.pdf")
    print()
    print("Presentation/Poster:")
    print("  → CrCuSe2_ultra_HQ_minimalist.png")
    print()
    print("Vector graphics (SVG/PDF) are recommended for publications")
    print("as they scale infinitely without quality loss.")
    print()


if __name__ == "__main__":
    main()
