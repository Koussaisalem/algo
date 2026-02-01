'use client';

import { useMemo } from 'react';

interface DataPoint {
  label: string;
  value: number;
}

interface BarChartProps {
  data: DataPoint[];
  title?: string;
  color?: string;
  height?: number;
}

export function BarChart({ data, title, color = 'blue', height = 200 }: BarChartProps) {
  const maxValue = useMemo(() => Math.max(...data.map((d) => d.value)), [data]);

  const gradients: Record<string, string> = {
    blue: 'from-blue-500 to-cyan-400',
    purple: 'from-purple-500 to-pink-400',
    green: 'from-emerald-500 to-teal-400',
    orange: 'from-orange-500 to-amber-400',
  };

  return (
    <div className="glass rounded-2xl p-6 border border-white/10">
      {title && (
        <h3 className="text-lg font-semibold text-white mb-6">{title}</h3>
      )}
      <div className="flex items-end gap-2" style={{ height }}>
        {data.map((point, index) => {
          const heightPercent = (point.value / maxValue) * 100;
          return (
            <div
              key={index}
              className="flex-1 flex flex-col items-center gap-2 group"
            >
              <div className="relative w-full flex flex-col items-center">
                <span className="absolute -top-6 text-xs text-white/60 opacity-0 group-hover:opacity-100 transition-opacity">
                  {point.value}
                </span>
                <div
                  className={`w-full rounded-t-lg bg-gradient-to-t ${gradients[color]} transition-all duration-500 ease-out hover:opacity-80`}
                  style={{
                    height: `${heightPercent}%`,
                    minHeight: 4,
                    animationDelay: `${index * 50}ms`,
                  }}
                />
              </div>
              <span className="text-xs text-white/40 truncate max-w-full">
                {point.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

interface LineChartProps {
  data: DataPoint[];
  title?: string;
  color?: string;
  height?: number;
}

export function LineChart({ data, title, color = 'blue', height = 200 }: LineChartProps) {
  const maxValue = useMemo(() => Math.max(...data.map((d) => d.value)), [data]);
  const minValue = useMemo(() => Math.min(...data.map((d) => d.value)), [data]);
  const range = maxValue - minValue || 1;

  const colors: Record<string, { stroke: string; fill: string }> = {
    blue: { stroke: '#3b82f6', fill: 'rgba(59, 130, 246, 0.1)' },
    purple: { stroke: '#a855f7', fill: 'rgba(168, 85, 247, 0.1)' },
    green: { stroke: '#10b981', fill: 'rgba(16, 185, 129, 0.1)' },
    orange: { stroke: '#f97316', fill: 'rgba(249, 115, 22, 0.1)' },
  };

  const points = useMemo(() => {
    const width = 100;
    const stepX = width / (data.length - 1 || 1);
    return data.map((d, i) => ({
      x: i * stepX,
      y: 100 - ((d.value - minValue) / range) * 100,
      ...d,
    }));
  }, [data, minValue, range]);

  const pathD = useMemo(() => {
    if (points.length === 0) return '';
    return points
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`)
      .join(' ');
  }, [points]);

  const areaD = useMemo(() => {
    if (points.length === 0) return '';
    return `${pathD} L ${points[points.length - 1].x} 100 L 0 100 Z`;
  }, [pathD, points]);

  return (
    <div className="glass rounded-2xl p-6 border border-white/10">
      {title && (
        <h3 className="text-lg font-semibold text-white mb-6">{title}</h3>
      )}
      <div style={{ height }} className="relative">
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="w-full h-full">
          {/* Grid lines */}
          {[0, 25, 50, 75, 100].map((y) => (
            <line
              key={y}
              x1="0"
              y1={y}
              x2="100"
              y2={y}
              stroke="rgba(255,255,255,0.05)"
              strokeWidth="0.5"
            />
          ))}
          
          {/* Area fill */}
          <path
            d={areaD}
            fill={colors[color].fill}
            className="transition-all duration-500"
          />
          
          {/* Line */}
          <path
            d={pathD}
            fill="none"
            stroke={colors[color].stroke}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="transition-all duration-500"
            style={{
              filter: `drop-shadow(0 0 8px ${colors[color].stroke})`,
            }}
          />
          
          {/* Points */}
          {points.map((p, i) => (
            <circle
              key={i}
              cx={p.x}
              cy={p.y}
              r="2"
              fill={colors[color].stroke}
              className="transition-all duration-300 hover:r-4"
            />
          ))}
        </svg>
        
        {/* X-axis labels */}
        <div className="absolute bottom-0 left-0 right-0 flex justify-between text-xs text-white/40 -mb-6">
          {data.filter((_, i) => i % Math.ceil(data.length / 6) === 0).map((d, i) => (
            <span key={i}>{d.label}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

interface DonutChartProps {
  data: { label: string; value: number; color: string }[];
  title?: string;
  size?: number;
}

export function DonutChart({ data, title, size = 200 }: DonutChartProps) {
  const total = useMemo(() => data.reduce((sum, d) => sum + d.value, 0), [data]);

  const segments = useMemo(() => {
    let currentAngle = -90; // Start from top
    return data.map((d) => {
      const angle = (d.value / total) * 360;
      const startAngle = currentAngle;
      currentAngle += angle;
      return { ...d, startAngle, angle };
    });
  }, [data, total]);

  const polarToCartesian = (angle: number, radius: number) => {
    const rad = (angle * Math.PI) / 180;
    return {
      x: 50 + radius * Math.cos(rad),
      y: 50 + radius * Math.sin(rad),
    };
  };

  const describeArc = (startAngle: number, endAngle: number, innerR: number, outerR: number) => {
    const start = polarToCartesian(startAngle, outerR);
    const end = polarToCartesian(endAngle, outerR);
    const innerStart = polarToCartesian(endAngle, innerR);
    const innerEnd = polarToCartesian(startAngle, innerR);
    const largeArc = endAngle - startAngle > 180 ? 1 : 0;

    return [
      `M ${start.x} ${start.y}`,
      `A ${outerR} ${outerR} 0 ${largeArc} 1 ${end.x} ${end.y}`,
      `L ${innerStart.x} ${innerStart.y}`,
      `A ${innerR} ${innerR} 0 ${largeArc} 0 ${innerEnd.x} ${innerEnd.y}`,
      'Z',
    ].join(' ');
  };

  return (
    <div className="glass rounded-2xl p-6 border border-white/10">
      {title && (
        <h3 className="text-lg font-semibold text-white mb-6 text-center">{title}</h3>
      )}
      <div className="flex flex-col items-center gap-4">
        <div style={{ width: size, height: size }} className="relative">
          <svg viewBox="0 0 100 100" className="w-full h-full">
            {segments.map((seg, i) => (
              <path
                key={i}
                d={describeArc(seg.startAngle, seg.startAngle + seg.angle - 1, 25, 40)}
                fill={seg.color}
                className="transition-all duration-300 hover:opacity-80"
                style={{
                  filter: `drop-shadow(0 0 4px ${seg.color})`,
                }}
              />
            ))}
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="text-2xl font-bold text-white">{total}</div>
              <div className="text-xs text-white/60">Total</div>
            </div>
          </div>
        </div>
        <div className="flex flex-wrap justify-center gap-4">
          {data.map((d, i) => (
            <div key={i} className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: d.color }}
              />
              <span className="text-sm text-white/60">{d.label}</span>
              <span className="text-sm font-medium text-white">{d.value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

interface ProgressRingProps {
  value: number;
  max?: number;
  size?: number;
  strokeWidth?: number;
  color?: string;
  label?: string;
}

export function ProgressRing({
  value,
  max = 100,
  size = 120,
  strokeWidth = 8,
  color = '#3b82f6',
  label,
}: ProgressRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.min(value / max, 1);
  const offset = circumference * (1 - progress);

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={size} height={size} className="-rotate-90">
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth={strokeWidth}
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-500 ease-out"
          style={{
            filter: `drop-shadow(0 0 8px ${color})`,
          }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-xl font-bold text-white">
          {Math.round(progress * 100)}%
        </span>
        {label && (
          <span className="text-xs text-white/60">{label}</span>
        )}
      </div>
    </div>
  );
}
