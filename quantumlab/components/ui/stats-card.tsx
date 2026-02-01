import { LucideIcon, TrendingUp, TrendingDown } from 'lucide-react';

interface StatsCardProps {
  title: string;
  value: string | number;
  change?: number;
  changeLabel?: string;
  icon: LucideIcon;
  iconColor?: string;
  gradient?: string;
}

export function StatsCard({
  title,
  value,
  change,
  changeLabel,
  icon: Icon,
  iconColor = 'text-blue-400',
  gradient = 'from-blue-500/20 to-cyan-500/20',
}: StatsCardProps) {
  const isPositive = change && change > 0;
  const isNegative = change && change < 0;

  return (
    <div className="glass rounded-2xl p-6 border border-white/10 card-hover group">
      <div className="flex items-start justify-between mb-4">
        <div className={`p-3 rounded-xl bg-gradient-to-br ${gradient}`}>
          <Icon className={`w-6 h-6 ${iconColor}`} />
        </div>
        {change !== undefined && (
          <div
            className={`flex items-center gap-1 text-sm font-medium ${
              isPositive ? 'text-emerald-400' : isNegative ? 'text-red-400' : 'text-white/60'
            }`}
          >
            {isPositive ? (
              <TrendingUp className="w-4 h-4" />
            ) : isNegative ? (
              <TrendingDown className="w-4 h-4" />
            ) : null}
            <span>{Math.abs(change)}%</span>
          </div>
        )}
      </div>

      <div className="space-y-1">
        <p className="text-3xl font-bold text-white group-hover:text-gradient transition-all">
          {value}
        </p>
        <div className="flex items-center justify-between">
          <p className="text-sm text-white/60">{title}</p>
          {changeLabel && (
            <p className="text-xs text-white/40">{changeLabel}</p>
          )}
        </div>
      </div>
    </div>
  );
}
