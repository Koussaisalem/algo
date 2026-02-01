import { ReactNode } from 'react';

interface BadgeProps {
  children: ReactNode;
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info';
  size?: 'sm' | 'md';
  dot?: boolean;
  className?: string;
}

const variants = {
  default: 'bg-white/10 text-white/80 border-white/20',
  success: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  warning: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  error: 'bg-red-500/20 text-red-400 border-red-500/30',
  info: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
};

const dotColors = {
  default: 'bg-white/60',
  success: 'bg-emerald-400',
  warning: 'bg-amber-400',
  error: 'bg-red-400',
  info: 'bg-blue-400',
};

export function Badge({
  children,
  variant = 'default',
  size = 'sm',
  dot = false,
  className = '',
}: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 border rounded-full font-medium ${
        variants[variant]
      } ${size === 'sm' ? 'px-2.5 py-0.5 text-xs' : 'px-3 py-1 text-sm'} ${className}`}
    >
      {dot && (
        <span
          className={`w-1.5 h-1.5 rounded-full ${dotColors[variant]} animate-pulse`}
        />
      )}
      {children}
    </span>
  );
}
