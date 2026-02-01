'use client';

import { useState, useEffect } from 'react';
import { X, ArrowRight, Upload, Cpu, Zap, Sparkles, Check } from 'lucide-react';
import { Button } from './button';

const steps = [
  {
    id: 1,
    title: 'Upload Your Dataset',
    description: 'Start by uploading molecular structures in common formats like XYZ, SDF, or CIF.',
    icon: Upload,
    action: '/datasets',
  },
  {
    id: 2,
    title: 'Train AI Models',
    description: 'Train surrogate models to predict material properties with high accuracy.',
    icon: Cpu,
    action: '/models',
  },
  {
    id: 3,
    title: 'Run Computations',
    description: 'Perform xTB and DFT calculations on your structures for validation.',
    icon: Zap,
    action: '/compute',
  },
  {
    id: 4,
    title: 'Discover Materials',
    description: 'Explore generated structures and validate novel material discoveries.',
    icon: Sparkles,
    action: '/results',
  },
];

export function OnboardingModal() {
  const [isOpen, setIsOpen] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    const hasSeenOnboarding = localStorage.getItem('onboarding-complete');
    if (!hasSeenOnboarding) {
      // Delay showing to let the page load
      const timer = setTimeout(() => setIsOpen(true), 1000);
      return () => clearTimeout(timer);
    }
  }, []);

  const handleComplete = () => {
    localStorage.setItem('onboarding-complete', 'true');
    setIsOpen(false);
  };

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep((s) => s + 1);
    } else {
      handleComplete();
    }
  };

  if (!isOpen) return null;

  const step = steps[currentStep];
  const Icon = step.icon;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm animate-fade-in"
        onClick={handleComplete}
      />

      {/* Modal */}
      <div className="relative w-full max-w-lg mx-4 glass rounded-3xl border border-white/10 shadow-2xl animate-scale-in overflow-hidden">
        {/* Close button */}
        <button
          onClick={handleComplete}
          className="absolute top-4 right-4 p-2 rounded-lg hover:bg-white/10 transition-colors z-10"
        >
          <X className="w-5 h-5 text-white/60" />
        </button>

        {/* Progress */}
        <div className="px-8 pt-8">
          <div className="flex items-center gap-2 mb-6">
            {steps.map((s, i) => (
              <div
                key={s.id}
                className={`h-1 flex-1 rounded-full transition-all ${
                  i <= currentStep ? 'bg-blue-500' : 'bg-white/10'
                }`}
              />
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="px-8 pb-8">
          <div className="flex flex-col items-center text-center mb-8">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 flex items-center justify-center mb-6">
              <Icon className="w-10 h-10 text-blue-400" />
            </div>
            <h2 className="text-2xl font-bold text-white mb-3">{step.title}</h2>
            <p className="text-neutral-400 max-w-sm">{step.description}</p>
          </div>

          {/* Step indicators */}
          <div className="flex justify-center gap-4 mb-8">
            {steps.map((s, i) => (
              <button
                key={s.id}
                onClick={() => setCurrentStep(i)}
                className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium transition-all ${
                  i === currentStep
                    ? 'bg-blue-500 text-white scale-110'
                    : i < currentStep
                    ? 'bg-emerald-500/20 text-emerald-400'
                    : 'bg-white/5 text-white/40'
                }`}
              >
                {i < currentStep ? <Check className="w-4 h-4" /> : i + 1}
              </button>
            ))}
          </div>

          {/* Actions */}
          <div className="flex items-center justify-between">
            <button
              onClick={handleComplete}
              className="text-sm text-neutral-400 hover:text-white transition-colors"
            >
              Skip tour
            </button>
            <Button onClick={handleNext} className="btn-shine px-6">
              {currentStep < steps.length - 1 ? (
                <>
                  Next <ArrowRight className="w-4 h-4 ml-2" />
                </>
              ) : (
                'Get Started'
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

export function ResetOnboarding() {
  const handleReset = () => {
    localStorage.removeItem('onboarding-complete');
    window.location.reload();
  };

  return (
    <button
      onClick={handleReset}
      className="text-sm text-neutral-500 hover:text-white transition-colors"
    >
      Restart tour
    </button>
  );
}
