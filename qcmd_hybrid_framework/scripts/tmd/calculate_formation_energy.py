#!/usr/bin/env python3
"""
Calculate elemental reference energies and formation energy for CrCuSe2
"""

from ase import Atoms
from ase.io import write
from gpaw import GPAW, PW, FermiDirac, Mixer
import numpy as np

print("=" * 80)
print("🧮 CALCULATING ELEMENTAL REFERENCE ENERGIES")
print("=" * 80)

# Same DFT settings as your CrCuSe2 calculation
calc_params = {
    'mode': PW(340),  # 340 eV cutoff
    'xc': 'PBE',
    'kpts': (8, 8, 8),  # Dense k-points for bulk elements
    'occupations': FermiDirac(0.1),
    'mixer': Mixer(0.05, 5, 50),
    'txt': 'elemental_calc.txt',
    'symmetry': 'off'
}

# 1. Chromium (bcc structure)
print("\n1️⃣  Calculating Cr (bcc)...")
a_Cr = 2.88  # Å, experimental lattice parameter
Cr = Atoms('Cr2',
           scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)],
           cell=[a_Cr, a_Cr, a_Cr],
           pbc=True)

Cr.calc = GPAW(**calc_params, txt='Cr_bcc.txt')
E_Cr_total = Cr.get_potential_energy()
E_Cr_per_atom = E_Cr_total / len(Cr)
print(f"   Cr (bcc): {E_Cr_per_atom:.6f} eV/atom")

# 2. Copper (fcc structure)
print("\n2️⃣  Calculating Cu (fcc)...")
a_Cu = 3.61  # Å
Cu = Atoms('Cu4',
           scaled_positions=[(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
           cell=[a_Cu, a_Cu, a_Cu],
           pbc=True)

Cu.calc = GPAW(**calc_params, txt='Cu_fcc.txt')
E_Cu_total = Cu.get_potential_energy()
E_Cu_per_atom = E_Cu_total / len(Cu)
print(f"   Cu (fcc): {E_Cu_per_atom:.6f} eV/atom")

# 3. Selenium (trigonal structure)
print("\n3️⃣  Calculating Se (trigonal)...")
# Se has helical chains, simplified here
a_Se = 4.36  # Å
c_Se = 4.95  # Å
Se = Atoms('Se3',
           scaled_positions=[(0.217, 0, 0.333), (0, 0.217, 0.667), (0.783, 0.783, 0)],
           cell=[a_Se, a_Se, c_Se],
           pbc=True)

Se.calc = GPAW(**{**calc_params, 'kpts': (6, 6, 6)}, txt='Se_trig.txt')
E_Se_total = Se.get_potential_energy()
E_Se_per_atom = E_Se_total / len(Se)
print(f"   Se (trigonal): {E_Se_per_atom:.6f} eV/atom")

# Calculate formation energy
print("\n" + "=" * 80)
print("🎯 FORMATION ENERGY CALCULATION")
print("=" * 80)

E_CrCuSe2 = -15.288124  # From your DFT

E_form = E_CrCuSe2 - (E_Cr_per_atom + E_Cu_per_atom + 2 * E_Se_per_atom)
E_form_per_atom = E_form / 4

print(f"\nYour CrCuSe₂: {E_CrCuSe2:.6f} eV (4 atoms)")
print(f"\nReferences:")
print(f"  Cr: {E_Cr_per_atom:.6f} eV/atom")
print(f"  Cu: {E_Cu_per_atom:.6f} eV/atom")
print(f"  Se: {E_Se_per_atom:.6f} eV/atom")

print(f"\nFormation Energy:")
print(f"  E_form = {E_CrCuSe2:.6f} - ({E_Cr_per_atom:.6f} + {E_Cu_per_atom:.6f} + 2×{E_Se_per_atom:.6f})")
print(f"  E_form = {E_form:.6f} eV")
print(f"  E_form/atom = {E_form_per_atom:.6f} eV/atom")

print(f"\n📊 COMPARISON:")
print(f"  Your structure: {E_form_per_atom:.3f} eV/atom")
print(f"  mp-568587:      -0.368 eV/atom")

if E_form_per_atom < -0.368:
    print(f"\n  🎉🎉🎉 YOUR STRUCTURE IS MORE STABLE by {-0.368 - E_form_per_atom:.3f} eV/atom!")
elif E_form_per_atom < 0:
    print(f"\n  ✅ Your structure is stable (E_form < 0)")
    print(f"  ⚠️  But mp-568587 is {E_form_per_atom + 0.368:.3f} eV/atom more stable")
else:
    print(f"\n  ❌ Your structure is UNSTABLE (E_form > 0)")

# Save results
with open('formation_energy_results.txt', 'w') as f:
    f.write(f"Elemental Reference Energies (PBE, consistent settings):\n")
    f.write(f"  Cr (bcc): {E_Cr_per_atom:.6f} eV/atom\n")
    f.write(f"  Cu (fcc): {E_Cu_per_atom:.6f} eV/atom\n")
    f.write(f"  Se (trig): {E_Se_per_atom:.6f} eV/atom\n\n")
    f.write(f"Formation Energy:\n")
    f.write(f"  Total: {E_form:.6f} eV\n")
    f.write(f"  Per atom: {E_form_per_atom:.6f} eV/atom\n\n")
    f.write(f"Comparison to mp-568587: {E_form_per_atom:.6f} vs -0.368 eV/atom\n")

print(f"\n✅ Results saved to formation_energy_results.txt")
