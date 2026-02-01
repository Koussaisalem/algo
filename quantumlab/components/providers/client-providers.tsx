'use client';

import { ReactNode } from 'react';
import { ToastProvider } from './toast-provider';
import { CommandPalette } from '@/components/ui/command-palette';
import { OnboardingModal } from '@/components/ui/onboarding';

export function ClientProviders({ children }: { children: ReactNode }) {
  return (
    <ToastProvider>
      {children}
      <CommandPalette />
      <OnboardingModal />
    </ToastProvider>
  );
}
