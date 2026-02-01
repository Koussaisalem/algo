"""
Database module for molecule library persistence.
Handles saving, searching, and version control of generated molecules.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

# Database path
DB_PATH = Path(__file__).parent / "molecule_library.db"


class MoleculeDatabase:
    """Manages persistent storage of generated molecules."""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Main molecules table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS molecules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    molecule_id TEXT UNIQUE NOT NULL,
                    formula TEXT NOT NULL,
                    num_atoms INTEGER NOT NULL,
                    xyz_content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    tags TEXT,
                    notes TEXT,
                    favorite BOOLEAN DEFAULT 0,
                    version INTEGER DEFAULT 1
                )
            """)
            
            # Properties table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS molecule_properties (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    molecule_id TEXT NOT NULL,
                    formation_energy REAL,
                    band_gap REAL,
                    dipole_moment REAL,
                    total_energy REAL,
                    targeted_gap REAL,
                    valid BOOLEAN,
                    FOREIGN KEY (molecule_id) REFERENCES molecules(molecule_id)
                )
            """)
            
            # Atoms table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS atoms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    molecule_id TEXT NOT NULL,
                    atom_index INTEGER NOT NULL,
                    element TEXT NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    z REAL NOT NULL,
                    FOREIGN KEY (molecule_id) REFERENCES molecules(molecule_id)
                )
            """)
            
            # Generation metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generation_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    molecule_id TEXT NOT NULL,
                    temperature REAL,
                    num_diffusion_steps INTEGER,
                    guidance_strength REAL,
                    seed INTEGER,
                    model_version TEXT,
                    FOREIGN KEY (molecule_id) REFERENCES molecules(molecule_id)
                )
            """)
            
            # Datasets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    molecule_count INTEGER DEFAULT 0
                )
            """)
            
            # Dataset members table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dataset_members (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id INTEGER NOT NULL,
                    molecule_id TEXT NOT NULL,
                    added_at TEXT NOT NULL,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(id),
                    FOREIGN KEY (molecule_id) REFERENCES molecules(molecule_id),
                    UNIQUE(dataset_id, molecule_id)
                )
            """)
            
            # Version history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS version_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    molecule_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    change_type TEXT NOT NULL,
                    changed_at TEXT NOT NULL,
                    changes TEXT,
                    FOREIGN KEY (molecule_id) REFERENCES molecules(molecule_id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_formula ON molecules(formula)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_band_gap ON molecule_properties(band_gap)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_element ON atoms(element)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_favorite ON molecules(favorite)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON molecules(created_at)")
    
    def save_molecule(self, molecule_data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save a molecule to the library."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            molecule_id = molecule_data.get("id", f"mol_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Check if molecule exists
            cursor.execute("SELECT id, version FROM molecules WHERE molecule_id = ?", (molecule_id,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing molecule
                version = existing['version'] + 1
                cursor.execute("""
                    UPDATE molecules 
                    SET formula = ?, num_atoms = ?, xyz_content = ?, updated_at = ?, version = ?
                    WHERE molecule_id = ?
                """, (
                    molecule_data["formula"],
                    molecule_data["properties"]["num_atoms"],
                    molecule_data["xyz_content"],
                    now,
                    version,
                    molecule_id
                ))
                
                # Log version change
                cursor.execute("""
                    INSERT INTO version_history (molecule_id, version, change_type, changed_at, changes)
                    VALUES (?, ?, ?, ?, ?)
                """, (molecule_id, version, "update", now, json.dumps(molecule_data)))
            else:
                # Insert new molecule
                cursor.execute("""
                    INSERT INTO molecules (molecule_id, formula, num_atoms, xyz_content, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    molecule_id,
                    molecule_data["formula"],
                    molecule_data["properties"]["num_atoms"],
                    molecule_data["xyz_content"],
                    now,
                    now
                ))
                
                # Log creation
                cursor.execute("""
                    INSERT INTO version_history (molecule_id, version, change_type, changed_at, changes)
                    VALUES (?, ?, ?, ?, ?)
                """, (molecule_id, 1, "create", now, json.dumps(molecule_data)))
            
            # Save properties
            cursor.execute("DELETE FROM molecule_properties WHERE molecule_id = ?", (molecule_id,))
            props = molecule_data["properties"]
            cursor.execute("""
                INSERT INTO molecule_properties 
                (molecule_id, formation_energy, band_gap, dipole_moment, total_energy, targeted_gap, valid)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                molecule_id,
                props.get("formation_energy"),
                props.get("band_gap"),
                props.get("dipole_moment"),
                props.get("total_energy"),
                props.get("targeted_gap"),
                props.get("valid", True)
            ))
            
            # Save atoms
            cursor.execute("DELETE FROM atoms WHERE molecule_id = ?", (molecule_id,))
            for atom in molecule_data["atoms"]:
                cursor.execute("""
                    INSERT INTO atoms (molecule_id, atom_index, element, x, y, z)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (molecule_id, atom["index"], atom["element"], atom["x"], atom["y"], atom["z"]))
            
            # Save generation metadata
            if metadata:
                cursor.execute("DELETE FROM generation_metadata WHERE molecule_id = ?", (molecule_id,))
                cursor.execute("""
                    INSERT INTO generation_metadata 
                    (molecule_id, temperature, num_diffusion_steps, guidance_strength, seed, model_version)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    molecule_id,
                    metadata.get("temperature"),
                    metadata.get("num_diffusion_steps"),
                    metadata.get("guidance_strength"),
                    metadata.get("seed"),
                    metadata.get("model_version", "qcmd-ecs-v1")
                ))
            
            return molecule_id
    
    def get_molecule(self, molecule_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a molecule by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get molecule
            cursor.execute("SELECT * FROM molecules WHERE molecule_id = ?", (molecule_id,))
            mol = cursor.fetchone()
            if not mol:
                return None
            
            # Get properties
            cursor.execute("SELECT * FROM molecule_properties WHERE molecule_id = ?", (molecule_id,))
            props = cursor.fetchone()
            
            # Get atoms
            cursor.execute("SELECT * FROM atoms WHERE molecule_id = ? ORDER BY atom_index", (molecule_id,))
            atoms = cursor.fetchall()
            
            # Get metadata
            cursor.execute("SELECT * FROM generation_metadata WHERE molecule_id = ?", (molecule_id,))
            metadata = cursor.fetchone()
            
            return {
                "id": mol["molecule_id"],
                "formula": mol["formula"],
                "num_atoms": mol["num_atoms"],
                "xyz_content": mol["xyz_content"],
                "created_at": mol["created_at"],
                "updated_at": mol["updated_at"],
                "tags": json.loads(mol["tags"]) if mol["tags"] else [],
                "notes": mol["notes"],
                "favorite": bool(mol["favorite"]),
                "version": mol["version"],
                "properties": dict(props) if props else {},
                "atoms": [dict(atom) for atom in atoms],
                "metadata": dict(metadata) if metadata else {}
            }
    
    def search_molecules(
        self,
        formula: Optional[str] = None,
        elements: Optional[List[str]] = None,
        min_band_gap: Optional[float] = None,
        max_band_gap: Optional[float] = None,
        favorite_only: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Search molecules with filters."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT DISTINCT m.*, p.band_gap, p.formation_energy
                FROM molecules m
                LEFT JOIN molecule_properties p ON m.molecule_id = p.molecule_id
                WHERE 1=1
            """
            params = []
            
            if formula:
                query += " AND m.formula LIKE ?"
                params.append(f"%{formula}%")
            
            if elements:
                element_conditions = " AND ".join(
                    ["EXISTS (SELECT 1 FROM atoms WHERE molecule_id = m.molecule_id AND element = ?)"] * len(elements)
                )
                query += f" AND ({element_conditions})"
                params.extend(elements)
            
            if min_band_gap is not None:
                query += " AND p.band_gap >= ?"
                params.append(min_band_gap)
            
            if max_band_gap is not None:
                query += " AND p.band_gap <= ?"
                params.append(max_band_gap)
            
            if favorite_only:
                query += " AND m.favorite = 1"
            
            query += " ORDER BY m.created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            return [dict(row) for row in results]
    
    def update_molecule_tags(self, molecule_id: str, tags: List[str]):
        """Update molecule tags."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE molecules SET tags = ?, updated_at = ? WHERE molecule_id = ?",
                (json.dumps(tags), datetime.now().isoformat(), molecule_id)
            )
    
    def toggle_favorite(self, molecule_id: str) -> bool:
        """Toggle favorite status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT favorite FROM molecules WHERE molecule_id = ?", (molecule_id,))
            current = cursor.fetchone()
            if current:
                new_status = not current['favorite']
                cursor.execute(
                    "UPDATE molecules SET favorite = ?, updated_at = ? WHERE molecule_id = ?",
                    (new_status, datetime.now().isoformat(), molecule_id)
                )
                return new_status
            return False
    
    def delete_molecule(self, molecule_id: str):
        """Delete a molecule and all related data."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM atoms WHERE molecule_id = ?", (molecule_id,))
            cursor.execute("DELETE FROM molecule_properties WHERE molecule_id = ?", (molecule_id,))
            cursor.execute("DELETE FROM generation_metadata WHERE molecule_id = ?", (molecule_id,))
            cursor.execute("DELETE FROM dataset_members WHERE molecule_id = ?", (molecule_id,))
            cursor.execute("DELETE FROM version_history WHERE molecule_id = ?", (molecule_id,))
            cursor.execute("DELETE FROM molecules WHERE molecule_id = ?", (molecule_id,))
    
    def create_dataset(self, name: str, description: str = "") -> int:
        """Create a new dataset."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO datasets (name, description, created_at)
                VALUES (?, ?, ?)
            """, (name, description, datetime.now().isoformat()))
            return cursor.lastrowid
    
    def add_to_dataset(self, dataset_id: int, molecule_id: str):
        """Add molecule to dataset."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO dataset_members (dataset_id, molecule_id, added_at)
                    VALUES (?, ?, ?)
                """, (dataset_id, molecule_id, datetime.now().isoformat()))
                
                # Update count
                cursor.execute("""
                    UPDATE datasets 
                    SET molecule_count = (SELECT COUNT(*) FROM dataset_members WHERE dataset_id = ?)
                    WHERE id = ?
                """, (dataset_id, dataset_id))
            except sqlite3.IntegrityError:
                pass  # Already in dataset
    
    def export_dataset(self, dataset_id: int, format: str = "xyz") -> str:
        """Export dataset to specified format."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get dataset info
            cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
            dataset = cursor.fetchone()
            
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            # Get molecules
            cursor.execute("""
                SELECT m.* 
                FROM molecules m
                JOIN dataset_members dm ON m.molecule_id = dm.molecule_id
                WHERE dm.dataset_id = ?
            """, (dataset_id,))
            molecules = cursor.fetchall()
            
            if format == "xyz":
                output = []
                for mol in molecules:
                    output.append(mol["xyz_content"])
                    output.append("")  # Blank line between molecules
                return "\n".join(output)
            
            elif format == "json":
                data = {
                    "dataset_name": dataset["name"],
                    "description": dataset["description"],
                    "created_at": dataset["created_at"],
                    "molecule_count": len(molecules),
                    "molecules": []
                }
                
                for mol in molecules:
                    mol_data = self.get_molecule(mol["molecule_id"])
                    if mol_data:
                        data["molecules"].append(mol_data)
                
                return json.dumps(data, indent=2)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) as total FROM molecules")
            total = cursor.fetchone()['total']
            
            cursor.execute("SELECT COUNT(*) as favorites FROM molecules WHERE favorite = 1")
            favorites = cursor.fetchone()['favorites']
            
            cursor.execute("SELECT COUNT(DISTINCT formula) as unique_formulas FROM molecules")
            unique_formulas = cursor.fetchone()['unique_formulas']
            
            cursor.execute("SELECT AVG(band_gap) as avg_gap FROM molecule_properties WHERE band_gap IS NOT NULL")
            avg_gap = cursor.fetchone()['avg_gap']
            
            cursor.execute("SELECT COUNT(*) as datasets FROM datasets")
            datasets = cursor.fetchone()['datasets']
            
            return {
                "total_molecules": total,
                "favorites": favorites,
                "unique_formulas": unique_formulas,
                "average_band_gap": avg_gap,
                "datasets": datasets
            }


# Global database instance
db = MoleculeDatabase()
