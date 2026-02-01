'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { 
  ArrowLeft, 
  Search, 
  Star, 
  Trash2, 
  Download,
  Filter,
  Plus,
  X,
  Package,
  BarChart3,
  Sparkles
} from 'lucide-react';

interface Molecule {
  molecule_id: string;
  formula: string;
  num_atoms: number;
  band_gap?: number;
  formation_energy?: number;
  favorite?: number;
  created_at: string;
  tags?: string;
}

interface LibraryStats {
  total_molecules: number;
  favorites: number;
  unique_formulas: number;
  average_band_gap: number;
  datasets: number;
}

export default function LibraryPage() {
  const [molecules, setMolecules] = useState<Molecule[]>([]);
  const [stats, setStats] = useState<LibraryStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterFavorites, setFilterFavorites] = useState(false);
  const [minBandGap, setMinBandGap] = useState<string>('');
  const [maxBandGap, setMaxBandGap] = useState<string>('');
  const [selectedElements, setSelectedElements] = useState<string[]>([]);
  const [showFilters, setShowFilters] = useState(false);

  const elements = ['H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br'];

  useEffect(() => {
    loadMolecules();
    loadStats();
  }, []);

  const loadMolecules = async () => {
    try {
      const response = await fetch('http://localhost:8000/library/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          formula: searchTerm || null,
          elements: selectedElements.length > 0 ? selectedElements : null,
          min_band_gap: minBandGap ? parseFloat(minBandGap) : null,
          max_band_gap: maxBandGap ? parseFloat(maxBandGap) : null,
          favorite_only: filterFavorites,
        }),
      });
      const data = await response.json();
      if (data.success) {
        setMolecules(data.molecules);
      }
    } catch (error) {
      console.error('Failed to load molecules:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadStats = async () => {
    try {
      const response = await fetch('http://localhost:8000/library/stats');
      const data = await response.json();
      if (data.success) {
        setStats(data.statistics);
      }
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  const toggleFavorite = async (moleculeId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/library/favorite/${moleculeId}`, {
        method: 'POST',
      });
      if (response.ok) {
        loadMolecules();
        loadStats();
      }
    } catch (error) {
      console.error('Failed to toggle favorite:', error);
    }
  };

  const deleteMolecule = async (moleculeId: string) => {
    if (!confirm('Are you sure you want to delete this molecule?')) return;
    
    try {
      const response = await fetch(`http://localhost:8000/library/molecule/${moleculeId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        loadMolecules();
        loadStats();
      }
    } catch (error) {
      console.error('Failed to delete molecule:', error);
    }
  };

  const toggleElement = (element: string) => {
    setSelectedElements(prev =>
      prev.includes(element)
        ? prev.filter(e => e !== element)
        : [...prev, element]
    );
  };

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white">
      {/* Header */}
      <nav className="border-b border-white/10 bg-black/40 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/dashboard" className="p-2 hover:bg-white/5 rounded-lg transition-colors">
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <div>
              <h1 className="text-xl font-semibold">Molecule Library</h1>
              <p className="text-xs text-gray-500">Browse and manage saved molecules</p>
            </div>
          </div>
          
          <Link
            href="/inference"
            className="px-4 py-2 rounded-full bg-gradient-to-r from-purple-600 to-pink-600 hover:opacity-90 transition-opacity text-sm font-medium flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Generate New
          </Link>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Statistics */}
        {stats && (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
            <StatCard label="Total Molecules" value={stats.total_molecules} icon={<Package className="w-4 h-4" />} />
            <StatCard label="Favorites" value={stats.favorites} icon={<Star className="w-4 h-4" />} />
            <StatCard label="Unique Formulas" value={stats.unique_formulas} icon={<Sparkles className="w-4 h-4" />} />
            <StatCard label="Avg Band Gap" value={`${stats.average_band_gap?.toFixed(2) || 0} eV`} icon={<BarChart3 className="w-4 h-4" />} />
            <StatCard label="Datasets" value={stats.datasets} icon={<Download className="w-4 h-4" />} />
          </div>
        )}

        {/* Search and Filters */}
        <div className="glass rounded-2xl p-6 mb-6">
          <div className="flex gap-4 mb-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
              <input
                type="text"
                placeholder="Search by formula (e.g., C6H12O2)..."
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder:text-gray-600 focus:outline-none focus:border-purple-500/50"
              />
            </div>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={`px-6 py-3 rounded-xl flex items-center gap-2 transition-colors ${
                showFilters ? 'bg-purple-500/20 text-purple-300' : 'glass text-gray-400 hover:text-white'
              }`}
            >
              <Filter className="w-5 h-5" />
              Filters
            </button>
            <button
              onClick={loadMolecules}
              className="px-6 py-3 rounded-xl bg-purple-600 hover:bg-purple-700 transition-colors font-medium"
            >
              Search
            </button>
          </div>

          {/* Filters Panel */}
          {showFilters && (
            <div className="pt-4 border-t border-white/10 space-y-4">
              {/* Favorites Filter */}
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={filterFavorites}
                  onChange={e => setFilterFavorites(e.target.checked)}
                  className="w-4 h-4 rounded accent-purple-600"
                />
                <span className="text-sm text-gray-300">Show favorites only</span>
              </label>

              {/* Band Gap Range */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Min Band Gap (eV)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={minBandGap}
                    onChange={e => setMinBandGap(e.target.value)}
                    placeholder="e.g., 1.0"
                    className="w-full px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white placeholder:text-gray-600 focus:outline-none focus:border-purple-500/50"
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Max Band Gap (eV)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={maxBandGap}
                    onChange={e => setMaxBandGap(e.target.value)}
                    placeholder="e.g., 5.0"
                    className="w-full px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white placeholder:text-gray-600 focus:outline-none focus:border-purple-500/50"
                  />
                </div>
              </div>

              {/* Element Filter */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">Elements (AND logic)</label>
                <div className="flex flex-wrap gap-2">
                  {elements.map(element => (
                    <button
                      key={element}
                      onClick={() => toggleElement(element)}
                      className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                        selectedElements.includes(element)
                          ? 'bg-purple-500/30 text-purple-300 border border-purple-500/50'
                          : 'glass text-gray-400 hover:text-white'
                      }`}
                    >
                      {element}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Molecules Grid */}
        {loading ? (
          <div className="text-center py-20">
            <div className="inline-block w-8 h-8 border-4 border-purple-500/30 border-t-purple-500 rounded-full animate-spin" />
            <p className="text-gray-500 mt-4">Loading library...</p>
          </div>
        ) : molecules.length === 0 ? (
          <div className="text-center py-20 glass rounded-2xl">
            <Package className="w-16 h-16 text-gray-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2">No molecules found</h3>
            <p className="text-gray-500 mb-6">Start by generating some molecules</p>
            <Link
              href="/inference"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-full bg-gradient-to-r from-purple-600 to-pink-600 hover:opacity-90 transition-opacity"
            >
              <Plus className="w-4 h-4" />
              Generate Molecules
            </Link>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {molecules.map(molecule => (
              <MoleculeCard
                key={molecule.molecule_id}
                molecule={molecule}
                onToggleFavorite={toggleFavorite}
                onDelete={deleteMolecule}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value, icon }: { label: string; value: string | number; icon: React.ReactNode }) {
  return (
    <div className="glass rounded-xl p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-gray-500">{label}</span>
        <div className="text-purple-400">{icon}</div>
      </div>
      <p className="text-2xl font-semibold text-white">{value}</p>
    </div>
  );
}

function MoleculeCard({ 
  molecule, 
  onToggleFavorite, 
  onDelete 
}: { 
  molecule: Molecule; 
  onToggleFavorite: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  return (
    <div className="glass rounded-xl p-4 hover:bg-white/[0.08] transition-colors">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="text-lg font-semibold text-white mb-1">{molecule.formula}</h3>
          <p className="text-xs text-gray-500">{molecule.num_atoms} atoms</p>
        </div>
        <div className="flex gap-1">
          <button
            onClick={() => onToggleFavorite(molecule.molecule_id)}
            className={`p-2 rounded-lg transition-colors ${
              molecule.favorite ? 'text-yellow-400 bg-yellow-400/10' : 'text-gray-500 hover:text-yellow-400 hover:bg-yellow-400/10'
            }`}
          >
            <Star className="w-4 h-4" fill={molecule.favorite ? 'currentColor' : 'none'} />
          </button>
          <button
            onClick={() => onDelete(molecule.molecule_id)}
            className="p-2 rounded-lg text-gray-500 hover:text-red-400 hover:bg-red-400/10 transition-colors"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {(molecule.band_gap !== undefined || molecule.formation_energy !== undefined) && (
        <div className="space-y-2 mb-3">
          {molecule.band_gap !== undefined && (
            <div className="flex justify-between text-sm">
              <span className="text-gray-500">Band Gap</span>
              <span className="text-purple-400 font-medium">{molecule.band_gap.toFixed(3)} eV</span>
            </div>
          )}
          {molecule.formation_energy !== undefined && (
            <div className="flex justify-between text-sm">
              <span className="text-gray-500">Formation Energy</span>
              <span className="text-blue-400 font-medium">{molecule.formation_energy.toFixed(3)} eV</span>
            </div>
          )}
        </div>
      )}

      <div className="flex items-center justify-between text-xs text-gray-600">
        <span>{new Date(molecule.created_at).toLocaleDateString()}</span>
        <Link
          href={`/library/${molecule.molecule_id}`}
          className="text-purple-400 hover:text-purple-300 transition-colors"
        >
          View Details →
        </Link>
      </div>
    </div>
  );
}
