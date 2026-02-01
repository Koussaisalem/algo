'use client';

import Link from "next/link"
import { ArrowRight, Atom, Cpu, LineChart, FlaskConical, Layers, ChevronRight, Sparkles, Zap, Shield } from "lucide-react"
import { MoleculeBackground, FloatingOrbs, GridPattern } from "@/components/ui/animated-background"

export default function Home() {
  return (
    <main className="relative min-h-screen">
      {/* Animated Background */}
      <div className="fixed inset-0 z-0 bg-[#050508]">
        <div className="absolute inset-0 bg-gradient-to-b from-blue-950/20 via-transparent to-purple-950/20" />
        <GridPattern />
        <FloatingOrbs />
        <MoleculeBackground particleCount={60} connectionDistance={140} speed={0.4} />
      </div>

      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-black/60 backdrop-blur-2xl border-b border-white/[0.06]">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-1.5 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600">
              <Atom className="w-5 h-5 text-white" />
            </div>
            <span className="text-lg font-semibold text-white tracking-tight">QuantumLab</span>
          </div>
          <div className="hidden md:flex items-center gap-8">
            <Link href="/dashboard" className="text-sm text-gray-400 hover:text-white transition-colors">
              Dashboard
            </Link>
            <Link href="/datasets" className="text-sm text-gray-400 hover:text-white transition-colors">
              Datasets
            </Link>
            <Link href="/models" className="text-sm text-gray-400 hover:text-white transition-colors">
              Models
            </Link>
            <Link href="/docs" className="text-sm text-gray-400 hover:text-white transition-colors">
              Docs
            </Link>
          </div>
          <Link 
            href="/dashboard"
            className="px-4 py-2 rounded-full bg-white text-black text-sm font-medium hover:bg-gray-100 transition-all"
          >
            Launch App
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-40 pb-32 px-6 overflow-hidden z-10">
        
        <div className="max-w-5xl mx-auto text-center relative z-10">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass mb-8">
            <Sparkles className="w-3.5 h-3.5 text-purple-400" />
            <span className="text-xs font-medium text-gray-300">
              Powered by Stiefel Manifold Diffusion
            </span>
          </div>

          <h1 className="text-6xl md:text-8xl font-semibold text-white mb-8 tracking-tight leading-[0.9]">
            Quantum Materials
            <br />
            <span className="text-gradient">
              Discovery Platform
            </span>
          </h1>

          <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-12 leading-relaxed font-light">
            A professional-grade platform for end-to-end materials discovery. 
            Combine generative AI with quantum chemistry.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Link 
              href="/inference"
              className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-full text-base font-medium hover:opacity-90 transition-all btn-shine"
            >
              <Sparkles className="w-4 h-4" />
              Generate Molecules
            </Link>
            <Link 
              href="/dashboard"
              className="inline-flex items-center gap-2 px-8 py-4 bg-white text-black rounded-full text-base font-medium hover:bg-gray-100 transition-all"
            >
              Open Dashboard
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link 
              href="/docs"
              className="inline-flex items-center gap-2 px-8 py-4 glass rounded-full text-base font-medium text-white hover:bg-white/[0.08] transition-all"
            >
              Read Documentation
            </Link>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-32 px-6 relative z-10">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-20">
            <h2 className="text-4xl font-semibold text-white mb-5 tracking-tight">Complete Discovery Pipeline</h2>
            <p className="text-lg text-gray-500 max-w-xl mx-auto font-light">
              Everything you need to go from raw data to validated discoveries
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            <FeatureCard
              icon={<Layers className="w-5 h-5" />}
              title="Dataset Management"
              description="Upload, organize, and preprocess molecular datasets with built-in validation"
              href="/datasets"
              color="blue"
            />
            <FeatureCard
              icon={<Cpu className="w-5 h-5" />}
              title="Model Training"
              description="Train custom generative and surrogate models with real-time monitoring"
              href="/models"
              color="purple"
            />
            <FeatureCard
              icon={<FlaskConical className="w-5 h-5" />}
              title="DFT Computation"
              description="Run xTB or full DFT calculations with configurable parameters"
              href="/compute"
              color="emerald"
            />
            <FeatureCard
              icon={<Atom className="w-5 h-5" />}
              title="Structure Generation"
              description="Generate novel molecular structures using manifold diffusion"
              href="/compute"
              color="amber"
            />
            <FeatureCard
              icon={<LineChart className="w-5 h-5" />}
              title="Results Analysis"
              description="Visualize and analyze results with interactive charts and 3D viewers"
              href="/results"
              color="rose"
            />
            <FeatureCard
              icon={<Zap className="w-5 h-5" />}
              title="Pipeline Automation"
              description="Chain workflows together for fully automated discovery pipelines"
              href="/compute"
              color="cyan"
            />
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-32 px-6 z-10 relative">
        <div className="max-w-5xl mx-auto">
          <div className="glass rounded-3xl p-12">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-10 text-center">
              <StatCard value="95%" label="Valid Structures" />
              <StatCard value="1000x" label="Faster than DFT" />
              <StatCard value="87%" label="xTB Convergence" />
              <StatCard value="34%" label="DFT Stability" />
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-32 px-6 z-10 relative">
        <div className="max-w-3xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass mb-8">
            <Shield className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-xs font-medium text-gray-300">
              Research-grade validation
            </span>
          </div>
          <h2 className="text-4xl font-semibold text-white mb-5 tracking-tight">Ready to discover?</h2>
          <p className="text-lg text-gray-500 mb-10 font-light">
            Start exploring the platform and accelerate your materials research
          </p>
          <Link 
            href="/dashboard"
            className="inline-flex items-center gap-2 px-8 py-4 bg-white text-black rounded-full text-base font-medium hover:bg-gray-100 transition-all btn-shine"
          >
            Get Started
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-10 px-6 border-t border-white/[0.06] z-10 relative">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-1.5 rounded-lg bg-white/5">
              <Atom className="w-4 h-4 text-gray-500" />
            </div>
            <span className="text-sm text-gray-500 font-medium">QuantumLab</span>
          </div>
          <p className="text-sm text-gray-600">
            Part of the Quantum Materials Discovery Platform
          </p>
        </div>
      </footer>
    </main>
  )
}

function FeatureCard({
  icon,
  title,
  description,
  href,
  color,
}: {
  icon: React.ReactNode
  title: string
  description: string
  href: string
  color: string
}) {
  const colors: Record<string, string> = {
    blue: "from-blue-500/20 to-blue-600/10 border-blue-500/20 group-hover:border-blue-500/40",
    purple: "from-purple-500/20 to-purple-600/10 border-purple-500/20 group-hover:border-purple-500/40",
    emerald: "from-emerald-500/20 to-emerald-600/10 border-emerald-500/20 group-hover:border-emerald-500/40",
    amber: "from-amber-500/20 to-amber-600/10 border-amber-500/20 group-hover:border-amber-500/40",
    rose: "from-rose-500/20 to-rose-600/10 border-rose-500/20 group-hover:border-rose-500/40",
    cyan: "from-cyan-500/20 to-cyan-600/10 border-cyan-500/20 group-hover:border-cyan-500/40",
  }
  
  const iconColors: Record<string, string> = {
    blue: "text-blue-400",
    purple: "text-purple-400",
    emerald: "text-emerald-400",
    amber: "text-amber-400",
    rose: "text-rose-400",
    cyan: "text-cyan-400",
  }
  
  return (
    <Link href={href}>
      <div className="group p-6 rounded-2xl glass card-hover h-full">
        <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${colors[color]} border flex items-center justify-center mb-5 transition-colors`}>
          <span className={iconColors[color]}>{icon}</span>
        </div>
        <h3 className="text-lg font-medium text-white mb-2">{title}</h3>
        <p className="text-sm text-gray-500 leading-relaxed">{description}</p>
      </div>
    </Link>
  )
}

function StatCard({ value, label }: { value: string; label: string }) {
  return (
    <div>
      <div className="text-4xl font-semibold text-white mb-2">{value}</div>
      <div className="text-sm text-gray-500">{label}</div>
    </div>
  )
}
