'use client';

import { useState, useEffect } from 'react';
import { 
  Upload, 
  Cpu, 
  Zap, 
  Sparkles, 
  CheckCircle2, 
  XCircle, 
  Clock,
  ArrowRight
} from 'lucide-react';
import Link from 'next/link';

interface ActivityItem {
  id: string;
  type: 'upload' | 'train' | 'compute' | 'discover' | 'error';
  title: string;
  description: string;
  timestamp: Date;
  href?: string;
}

const icons = {
  upload: Upload,
  train: Cpu,
  compute: Zap,
  discover: Sparkles,
  error: XCircle,
};

const colors = {
  upload: 'bg-blue-500/10 text-blue-400',
  train: 'bg-purple-500/10 text-purple-400',
  compute: 'bg-emerald-500/10 text-emerald-400',
  discover: 'bg-amber-500/10 text-amber-400',
  error: 'bg-red-500/10 text-red-400',
};

function timeAgo(date: Date): string {
  const seconds = Math.floor((new Date().getTime() - date.getTime()) / 1000);
  
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

export function ActivityFeed({ className = '' }: { className?: string }) {
  const [activities, setActivities] = useState<ActivityItem[]>([
    {
      id: '1',
      type: 'discover',
      title: 'New material discovered',
      description: 'CrCuSe₂ validated with DFT',
      timestamp: new Date(Date.now() - 1000 * 60 * 5),
      href: '/results',
    },
    {
      id: '2',
      type: 'compute',
      title: 'xTB calculation completed',
      description: '12 structures optimized',
      timestamp: new Date(Date.now() - 1000 * 60 * 30),
      href: '/compute',
    },
    {
      id: '3',
      type: 'train',
      title: 'Model training finished',
      description: 'SchNet surrogate - 95.2% accuracy',
      timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2),
      href: '/models',
    },
    {
      id: '4',
      type: 'upload',
      title: 'Dataset uploaded',
      description: 'QM9 subset - 5000 molecules',
      timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24),
      href: '/datasets',
    },
  ]);

  return (
    <div className={`glass rounded-2xl p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-white">Activity</h2>
        <span className="px-2 py-1 rounded-full bg-emerald-500/10 text-emerald-400 text-xs">
          Live
        </span>
      </div>

      <div className="space-y-1">
        {activities.map((activity) => {
          const Icon = icons[activity.type];
          return (
            <div
              key={activity.id}
              className="group flex items-start gap-4 p-3 rounded-xl hover:bg-white/5 transition-colors"
            >
              <div className={`p-2 rounded-lg ${colors[activity.type]}`}>
                <Icon className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-white truncate">
                  {activity.title}
                </p>
                <p className="text-xs text-neutral-500 truncate">
                  {activity.description}
                </p>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <span className="text-xs text-neutral-500">
                  {timeAgo(activity.timestamp)}
                </span>
                {activity.href && (
                  <Link
                    href={activity.href}
                    className="p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-white/10 transition-all"
                  >
                    <ArrowRight className="w-3 h-3 text-neutral-400" />
                  </Link>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <button className="mt-4 w-full py-2 text-sm text-neutral-500 hover:text-white transition-colors">
        View all activity
      </button>
    </div>
  );
}
