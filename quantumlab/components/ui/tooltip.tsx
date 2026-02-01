'use client';

import { ReactNode, useState } from 'react';

interface TooltipProps {
  content: ReactNode;
  children: ReactNode;
  side?: 'top' | 'bottom' | 'left' | 'right';
  className?: string;
}

const positions = {
  top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
  bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
  left: 'right-full top-1/2 -translate-y-1/2 mr-2',
  right: 'left-full top-1/2 -translate-y-1/2 ml-2',
};

const arrows = {
  top: 'top-full left-1/2 -translate-x-1/2 border-t-[#1a1a24] border-x-transparent border-b-transparent',
  bottom: 'bottom-full left-1/2 -translate-x-1/2 border-b-[#1a1a24] border-x-transparent border-t-transparent',
  left: 'left-full top-1/2 -translate-y-1/2 border-l-[#1a1a24] border-y-transparent border-r-transparent',
  right: 'right-full top-1/2 -translate-y-1/2 border-r-[#1a1a24] border-y-transparent border-l-transparent',
};

export function Tooltip({
  content,
  children,
  side = 'top',
  className = '',
}: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <div
      className="relative inline-flex"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children}
      {isVisible && (
        <div
          className={`absolute z-50 ${positions[side]} animate-fade-in pointer-events-none`}
        >
          <div
            className={`px-3 py-2 text-sm text-white bg-[#1a1a24] border border-white/10 rounded-lg shadow-xl whitespace-nowrap ${className}`}
          >
            {content}
          </div>
          <div
            className={`absolute w-0 h-0 border-4 ${arrows[side]}`}
          />
        </div>
      )}
    </div>
  );
}
