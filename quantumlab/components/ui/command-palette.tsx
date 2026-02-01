'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { useRouter } from 'next/navigation';
import {
  Search,
  Database,
  Brain,
  Cpu,
  BarChart3,
  Settings,
  FileText,
  Plus,
  Play,
  Upload,
  Home,
  Command,
} from 'lucide-react';

interface CommandItem {
  id: string;
  label: string;
  icon: React.ElementType;
  category: string;
  action: () => void;
  shortcut?: string;
}

export function CommandPalette() {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  const commands: CommandItem[] = [
    // Navigation
    { id: 'nav-home', label: 'Go to Dashboard', icon: Home, category: 'Navigation', action: () => router.push('/dashboard'), shortcut: 'G D' },
    { id: 'nav-datasets', label: 'Go to Datasets', icon: Database, category: 'Navigation', action: () => router.push('/datasets'), shortcut: 'G S' },
    { id: 'nav-models', label: 'Go to Models', icon: Brain, category: 'Navigation', action: () => router.push('/models'), shortcut: 'G M' },
    { id: 'nav-compute', label: 'Go to Compute', icon: Cpu, category: 'Navigation', action: () => router.push('/compute'), shortcut: 'G C' },
    { id: 'nav-results', label: 'Go to Results', icon: BarChart3, category: 'Navigation', action: () => router.push('/results'), shortcut: 'G R' },
    { id: 'nav-docs', label: 'Go to Documentation', icon: FileText, category: 'Navigation', action: () => router.push('/docs') },
    { id: 'nav-settings', label: 'Go to Settings', icon: Settings, category: 'Navigation', action: () => router.push('/settings') },
    // Actions
    { id: 'action-upload', label: 'Upload Dataset', icon: Upload, category: 'Actions', action: () => { router.push('/datasets'); } },
    { id: 'action-train', label: 'Train New Model', icon: Play, category: 'Actions', action: () => { router.push('/models'); } },
    { id: 'action-compute', label: 'Start Computation', icon: Cpu, category: 'Actions', action: () => { router.push('/compute'); } },
    { id: 'action-new', label: 'Create New Project', icon: Plus, category: 'Actions', action: () => {} },
  ];

  const filteredCommands = commands.filter((cmd) =>
    cmd.label.toLowerCase().includes(query.toLowerCase()) ||
    cmd.category.toLowerCase().includes(query.toLowerCase())
  );

  const groupedCommands = filteredCommands.reduce((acc, cmd) => {
    if (!acc[cmd.category]) acc[cmd.category] = [];
    acc[cmd.category].push(cmd);
    return acc;
  }, {} as Record<string, CommandItem[]>);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    // Open with Cmd/Ctrl + K
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      setIsOpen((prev) => !prev);
      setQuery('');
      setSelectedIndex(0);
    }

    if (!isOpen) return;

    if (e.key === 'Escape') {
      setIsOpen(false);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) => Math.min(prev + 1, filteredCommands.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => Math.max(prev - 1, 0));
    } else if (e.key === 'Enter' && filteredCommands[selectedIndex]) {
      e.preventDefault();
      filteredCommands[selectedIndex].action();
      setIsOpen(false);
    }
  }, [isOpen, filteredCommands, selectedIndex]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  if (!isOpen) return null;

  let flatIndex = 0;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh]">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fade-in"
        onClick={() => setIsOpen(false)}
      />

      {/* Palette */}
      <div className="relative w-full max-w-xl mx-4 glass rounded-2xl border border-white/10 shadow-2xl animate-scale-in overflow-hidden">
        {/* Search Input */}
        <div className="flex items-center gap-3 px-4 py-4 border-b border-white/10">
          <Search className="w-5 h-5 text-white/40" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search commands..."
            className="flex-1 bg-transparent text-white placeholder-white/40 outline-none text-lg"
          />
          <kbd className="px-2 py-1 text-xs text-white/40 bg-white/5 rounded-md border border-white/10">
            ESC
          </kbd>
        </div>

        {/* Commands List */}
        <div className="max-h-80 overflow-y-auto py-2">
          {Object.entries(groupedCommands).map(([category, cmds]) => (
            <div key={category}>
              <div className="px-4 py-2 text-xs font-medium text-white/40 uppercase tracking-wider">
                {category}
              </div>
              {cmds.map((cmd) => {
                const currentIndex = flatIndex++;
                const isSelected = currentIndex === selectedIndex;
                const Icon = cmd.icon;

                return (
                  <button
                    key={cmd.id}
                    onClick={() => {
                      cmd.action();
                      setIsOpen(false);
                    }}
                    className={`w-full flex items-center gap-3 px-4 py-3 text-left transition-colors ${
                      isSelected ? 'bg-white/10' : 'hover:bg-white/5'
                    }`}
                  >
                    <Icon className="w-5 h-5 text-white/60" />
                    <span className="flex-1 text-white">{cmd.label}</span>
                    {cmd.shortcut && (
                      <kbd className="px-2 py-1 text-xs text-white/40 bg-white/5 rounded-md">
                        {cmd.shortcut}
                      </kbd>
                    )}
                  </button>
                );
              })}
            </div>
          ))}

          {filteredCommands.length === 0 && (
            <div className="px-4 py-8 text-center text-white/40">
              No commands found for &quot;{query}&quot;
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-3 border-t border-white/10 text-xs text-white/40">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 bg-white/5 rounded">↑↓</kbd> Navigate
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 bg-white/5 rounded">↵</kbd> Select
            </span>
          </div>
          <div className="flex items-center gap-1">
            <Command className="w-3 h-3" />
            <span>K to open</span>
          </div>
        </div>
      </div>
    </div>
  );
}
