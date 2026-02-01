interface ProgressProps {
  value: number;
  max?: number;
  size?: 'sm' | 'md' | 'lg';
  color?: 'blue' | 'green' | 'purple' | 'orange';
  showLabel?: boolean;
  animated?: boolean;
  className?: string;
}

const sizes = {
  sm: 'h-1.5',
  md: 'h-2.5',
  lg: 'h-4',
};

const colors = {
  blue: 'bg-gradient-to-r from-blue-500 to-cyan-400',
  green: 'bg-gradient-to-r from-emerald-500 to-teal-400',
  purple: 'bg-gradient-to-r from-purple-500 to-pink-400',
  orange: 'bg-gradient-to-r from-orange-500 to-amber-400',
};

export function Progress({
  value,
  max = 100,
  size = 'md',
  color = 'blue',
  showLabel = false,
  animated = true,
  className = '',
}: ProgressProps) {
  const percent = Math.min(100, Math.max(0, (value / max) * 100));

  return (
    <div className={className}>
      <div className={`w-full bg-white/10 rounded-full overflow-hidden ${sizes[size]}`}>
        <div
          className={`h-full rounded-full ${colors[color]} ${
            animated ? 'transition-all duration-500 ease-out' : ''
          }`}
          style={{ width: `${percent}%` }}
        >
          {animated && (
            <div className="w-full h-full bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
          )}
        </div>
      </div>
      {showLabel && (
        <div className="flex justify-between mt-1 text-sm text-white/60">
          <span>{value}</span>
          <span>{percent.toFixed(0)}%</span>
        </div>
      )}
    </div>
  );
}

interface ProgressStepsProps {
  steps: string[];
  currentStep: number;
  className?: string;
}

export function ProgressSteps({ steps, currentStep, className = '' }: ProgressStepsProps) {
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {steps.map((step, i) => {
        const isComplete = i < currentStep;
        const isCurrent = i === currentStep;

        return (
          <div key={i} className="flex items-center gap-2 flex-1">
            <div
              className={`flex items-center justify-center w-8 h-8 rounded-full text-sm font-medium transition-all ${
                isComplete
                  ? 'bg-emerald-500 text-white'
                  : isCurrent
                  ? 'bg-blue-500 text-white ring-4 ring-blue-500/20'
                  : 'bg-white/10 text-white/40'
              }`}
            >
              {isComplete ? '✓' : i + 1}
            </div>
            <span
              className={`text-sm hidden sm:block ${
                isCurrent ? 'text-white font-medium' : 'text-white/40'
              }`}
            >
              {step}
            </span>
            {i < steps.length - 1 && (
              <div
                className={`flex-1 h-0.5 ${
                  isComplete ? 'bg-emerald-500' : 'bg-white/10'
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
