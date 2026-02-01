'use client';

import { useState, useEffect } from 'react';
import { Sun, Moon, Monitor } from 'lucide-react';

type Theme = 'light' | 'dark' | 'system';

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>('dark');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const stored = localStorage.getItem('theme') as Theme;
    if (stored) {
      setTheme(stored);
    }
  }, []);

  useEffect(() => {
    if (!mounted) return;

    const root = document.documentElement;
    localStorage.setItem('theme', theme);

    if (theme === 'system') {
      const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      root.classList.toggle('dark', systemDark);
      root.classList.toggle('light', !systemDark);
    } else {
      root.classList.toggle('dark', theme === 'dark');
      root.classList.toggle('light', theme === 'light');
    }
  }, [theme, mounted]);

  if (!mounted) return null;

  const options = [
    { value: 'light', icon: Sun },
    { value: 'dark', icon: Moon },
    { value: 'system', icon: Monitor },
  ] as const;

  return (
    <div className="flex items-center gap-1 p-1 rounded-xl bg-white/5 border border-white/10">
      {options.map(({ value, icon: Icon }) => (
        <button
          key={value}
          onClick={() => setTheme(value)}
          className={`p-2 rounded-lg transition-all ${
            theme === value
              ? 'bg-white/10 text-white'
              : 'text-white/40 hover:text-white/60'
          }`}
          title={value.charAt(0).toUpperCase() + value.slice(1)}
        >
          <Icon className="w-4 h-4" />
        </button>
      ))}
    </div>
  );
}
