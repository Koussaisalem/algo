import { ReactNode } from 'react';

interface KbdProps {
  children: ReactNode;
  className?: string;
}

export function Kbd({ children, className = '' }: KbdProps) {
  return (
    <kbd
      className={`inline-flex items-center justify-center px-2 py-1 text-xs font-medium text-white/60 bg-white/5 border border-white/10 rounded-md ${className}`}
    >
      {children}
    </kbd>
  );
}
