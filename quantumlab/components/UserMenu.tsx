'use client';

import { signOut, useSession } from 'next-auth/react';
import { useState, useRef, useEffect } from 'react';
import { User, LogOut, Settings, CreditCard, HelpCircle, ChevronDown } from 'lucide-react';

export function UserMenu() {
  const { data: session } = useSession();
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  if (!session?.user) return null;

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map(n => n[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  return (
    <div className="relative" ref={menuRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-white/5 transition-colors"
      >
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white text-sm font-semibold">
          {session.user.image ? (
            <img
              src={session.user.image}
              alt={session.user.name || ''}
              className="w-full h-full rounded-full"
            />
          ) : (
            getInitials(session.user.name || session.user.email || 'U')
          )}
        </div>
        <div className="hidden md:block text-left">
          <div className="text-sm font-medium text-white">
            {session.user.name || 'User'}
          </div>
          <div className="text-xs text-gray-400">
            {session.user.email}
          </div>
        </div>
        <ChevronDown className="w-4 h-4 text-gray-400" />
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-56 bg-gray-900 border border-white/10 rounded-lg shadow-xl py-1 z-50">
          <div className="px-4 py-3 border-b border-white/10">
            <div className="text-sm font-medium text-white">
              {session.user.name}
            </div>
            <div className="text-xs text-gray-400 mt-1">
              {session.user.email}
            </div>
          </div>

          <button
            onClick={() => setIsOpen(false)}
            className="w-full px-4 py-2 text-sm text-gray-300 hover:bg-white/5 flex items-center gap-3 transition-colors"
          >
            <User className="w-4 h-4" />
            Profile
          </button>

          <button
            onClick={() => setIsOpen(false)}
            className="w-full px-4 py-2 text-sm text-gray-300 hover:bg-white/5 flex items-center gap-3 transition-colors"
          >
            <Settings className="w-4 h-4" />
            Settings
          </button>

          <button
            onClick={() => setIsOpen(false)}
            className="w-full px-4 py-2 text-sm text-gray-300 hover:bg-white/5 flex items-center gap-3 transition-colors"
          >
            <CreditCard className="w-4 h-4" />
            Billing
          </button>

          <button
            onClick={() => setIsOpen(false)}
            className="w-full px-4 py-2 text-sm text-gray-300 hover:bg-white/5 flex items-center gap-3 transition-colors"
          >
            <HelpCircle className="w-4 h-4" />
            Help & Support
          </button>

          <div className="border-t border-white/10 my-1" />

          <button
            onClick={() => signOut({ callbackUrl: '/auth/login' })}
            className="w-full px-4 py-2 text-sm text-red-400 hover:bg-red-500/10 flex items-center gap-3 transition-colors"
          >
            <LogOut className="w-4 h-4" />
            Sign Out
          </button>
        </div>
      )}
    </div>
  );
}
