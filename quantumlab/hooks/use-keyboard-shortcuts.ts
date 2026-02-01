'use client';

import { useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';

interface KeyboardShortcut {
  key: string;
  modifiers?: ('ctrl' | 'meta' | 'shift' | 'alt')[];
  action: () => void;
  description?: string;
}

export function useKeyboardShortcuts(shortcuts: KeyboardShortcut[]) {
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      for (const shortcut of shortcuts) {
        const keyMatches = event.key.toLowerCase() === shortcut.key.toLowerCase();
        
        const modifiersMatch = shortcut.modifiers
          ? shortcut.modifiers.every((mod) => {
              switch (mod) {
                case 'ctrl':
                  return event.ctrlKey;
                case 'meta':
                  return event.metaKey;
                case 'shift':
                  return event.shiftKey;
                case 'alt':
                  return event.altKey;
                default:
                  return false;
              }
            })
          : true;

        if (keyMatches && modifiersMatch) {
          event.preventDefault();
          shortcut.action();
          break;
        }
      }
    },
    [shortcuts]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);
}

// Common navigation shortcuts
export function useNavigationShortcuts() {
  const router = useRouter();

  useKeyboardShortcuts([
    { key: 'd', modifiers: ['meta', 'shift'], action: () => router.push('/dashboard') },
    { key: 's', modifiers: ['meta', 'shift'], action: () => router.push('/datasets') },
    { key: 'm', modifiers: ['meta', 'shift'], action: () => router.push('/models') },
    { key: 'c', modifiers: ['meta', 'shift'], action: () => router.push('/compute') },
    { key: 'r', modifiers: ['meta', 'shift'], action: () => router.push('/results') },
  ]);
}
