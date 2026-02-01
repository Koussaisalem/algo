import { ReactNode } from 'react';

interface AvatarProps {
  src?: string;
  alt?: string;
  fallback?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
}

const sizes = {
  sm: 'w-8 h-8 text-xs',
  md: 'w-10 h-10 text-sm',
  lg: 'w-12 h-12 text-base',
  xl: 'w-16 h-16 text-lg',
};

export function Avatar({
  src,
  alt = '',
  fallback,
  size = 'md',
  className = '',
}: AvatarProps) {
  const initials = fallback
    ? fallback
        .split(' ')
        .map((n) => n[0])
        .join('')
        .slice(0, 2)
        .toUpperCase()
    : '';

  return (
    <div
      className={`relative inline-flex items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-purple-600 text-white font-medium overflow-hidden ${sizes[size]} ${className}`}
    >
      {src ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={src}
          alt={alt}
          className="w-full h-full object-cover"
        />
      ) : (
        <span>{initials}</span>
      )}
    </div>
  );
}

interface AvatarGroupProps {
  children: ReactNode;
  max?: number;
}

export function AvatarGroup({ children, max = 4 }: AvatarGroupProps) {
  const childArray = Array.isArray(children) ? children : [children];
  const visible = childArray.slice(0, max);
  const remaining = childArray.length - max;

  return (
    <div className="flex items-center -space-x-3">
      {visible.map((child, i) => (
        <div
          key={i}
          className="relative ring-2 ring-[#0a0a0f] rounded-full"
        >
          {child}
        </div>
      ))}
      {remaining > 0 && (
        <div className="relative ring-2 ring-[#0a0a0f] rounded-full">
          <div className="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center text-sm font-medium text-white">
            +{remaining}
          </div>
        </div>
      )}
    </div>
  );
}
