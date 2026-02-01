'use client';

import { ReactNode } from 'react';
import { ToastProvider } from './toast-provider';
import { CommandPalette } from '@/components/ui/command-palette';
import { OnboardingModal } from '@/components/ui/onboarding';
import { PremiumBackground } from '@/components/ui/animated-background';

export function ClientProviders({ children }: { children: ReactNode }) {
  return (
    <ToastProvider>
      <PremiumBackground 
        showMolecules={true}
        showOrbs={true}
        showGrid={true}
        particleCount={45}
      />
      {children}
      <CommandPalette />
      <OnboardingModal />
    </ToastProvider>
  );
}
