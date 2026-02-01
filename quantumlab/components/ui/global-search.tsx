'use client';

import { useState, useEffect, useRef } from 'react';
import { Search, X, Clock, ArrowRight } from 'lucide-react';
import Link from 'next/link';
import { useLocalStorage } from '@/hooks/use-utils';

interface SearchResult {
  id: string;
  type: 'dataset' | 'model' | 'result' | 'page';
  title: string;
  subtitle?: string;
  href: string;
}

interface GlobalSearchProps {
  className?: string;
}

export function GlobalSearch({ className = '' }: GlobalSearchProps) {
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [recentSearches, setRecentSearches] = useLocalStorage<string[]>('recent-searches', []);
  const inputRef = useRef<HTMLInputElement>(null);

  // Mock search results - in production this would query an API
  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      return;
    }

    const allItems: SearchResult[] = [
      { id: '1', type: 'dataset', title: 'QM9 Micro', subtitle: '5,000 molecules', href: '/datasets/ds-001' },
      { id: '2', type: 'model', title: 'SchNet Surrogate', subtitle: 'Trained model', href: '/models/model-001' },
      { id: '3', type: 'result', title: 'CrCuSe₂', subtitle: 'Novel discovery', href: '/results/result-001' },
      { id: '4', type: 'page', title: 'Dashboard', subtitle: 'Overview', href: '/dashboard' },
      { id: '5', type: 'page', title: 'Settings', subtitle: 'Preferences', href: '/settings' },
    ];

    const searchResults = allItems.filter(
      (r) =>
        r.title.toLowerCase().includes(query.toLowerCase()) ||
        r.subtitle?.toLowerCase().includes(query.toLowerCase())
    );

    setResults(searchResults);
  }, [query]);

  const handleSearch = (searchQuery: string) => {
    setQuery(searchQuery);
    if (searchQuery.trim() && !recentSearches.includes(searchQuery)) {
      setRecentSearches([searchQuery, ...recentSearches.slice(0, 4)]);
    }
  };

  const clearRecentSearches = () => {
    setRecentSearches([]);
  };

  const typeColors = {
    dataset: 'bg-blue-500/10 text-blue-400',
    model: 'bg-purple-500/10 text-purple-400',
    result: 'bg-emerald-500/10 text-emerald-400',
    page: 'bg-neutral-500/10 text-neutral-400',
  };

  return (
    <div className={`relative ${className}`}>
      <div className="relative">
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-400" />
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => handleSearch(e.target.value)}
          onFocus={() => setIsOpen(true)}
          placeholder="Search datasets, models, results..."
          className="w-full pl-12 pr-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-neutral-500 focus:outline-none focus:border-blue-500/50 focus:bg-white/[0.07] transition-all"
        />
        {query && (
          <button
            onClick={() => setQuery('')}
            className="absolute right-4 top-1/2 -translate-y-1/2 p-1 hover:bg-white/10 rounded transition-colors"
          >
            <X className="w-4 h-4 text-neutral-400" />
          </button>
        )}
      </div>

      {isOpen && (query || recentSearches.length > 0) && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute top-full left-0 right-0 mt-2 glass rounded-xl border border-white/10 shadow-2xl z-50 overflow-hidden animate-slide-up">
            {results.length > 0 ? (
              <div className="py-2">
                <div className="px-4 py-2 text-xs font-medium text-neutral-500 uppercase">
                  Results
                </div>
                {results.map((result) => (
                  <Link
                    key={result.id}
                    href={result.href}
                    onClick={() => setIsOpen(false)}
                    className="flex items-center gap-3 px-4 py-3 hover:bg-white/5 transition-colors"
                  >
                    <span
                      className={`px-2 py-1 rounded text-xs font-medium ${typeColors[result.type]}`}
                    >
                      {result.type}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-white truncate">{result.title}</p>
                      {result.subtitle && (
                        <p className="text-xs text-neutral-500 truncate">
                          {result.subtitle}
                        </p>
                      )}
                    </div>
                    <ArrowRight className="w-4 h-4 text-neutral-500" />
                  </Link>
                ))}
              </div>
            ) : query ? (
              <div className="px-4 py-8 text-center text-neutral-500">
                No results for &quot;{query}&quot;
              </div>
            ) : null}

            {!query && recentSearches.length > 0 && (
              <div className="py-2">
                <div className="flex items-center justify-between px-4 py-2">
                  <span className="text-xs font-medium text-neutral-500 uppercase">
                    Recent Searches
                  </span>
                  <button
                    onClick={clearRecentSearches}
                    className="text-xs text-neutral-500 hover:text-white transition-colors"
                  >
                    Clear
                  </button>
                </div>
                {recentSearches.map((search, i) => (
                  <button
                    key={i}
                    onClick={() => handleSearch(search)}
                    className="flex items-center gap-3 w-full px-4 py-2 hover:bg-white/5 transition-colors text-left"
                  >
                    <Clock className="w-4 h-4 text-neutral-500" />
                    <span className="text-sm text-neutral-400">{search}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
