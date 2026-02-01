'use client';

import { ReactNode } from 'react';
import { ToastProvider } from './toast-provider';
import { CommandPalette } from '@/components/ui/command-palette';

export function ClientProviders({ children }: { children: ReactNode }) {
  return (
    <ToastProvider>
      {children}
      <CommandPalette />
    </ToastProvider>
  );
}
