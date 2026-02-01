import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ArrowRight, Sparkles, Zap, Database } from "lucide-react"

export default function Home() {
  return (
    <main className="relative min-h-screen overflow-hidden">
      {/* Hero Section */}
      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4 sm:px-6 lg:px-8">
        {/* Animated Background */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-1/2 -left-1/4 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse"></div>
          <div className="absolute -bottom-1/2 -right-1/4 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse animation-delay-2000"></div>
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse animation-delay-4000"></div>
        </div>

        {/* Content */}
        <div className="relative z-20 text-center max-w-5xl mx-auto">
          {/* Logo/Badge */}
          <div className="inline-flex items-center gap-2 glass-card px-4 py-2 mb-8 animate-fade-in">
            <Sparkles className="w-4 h-4 text-blue-400" />
            <span className="text-sm font-medium text-gray-300">
              Powered by Stiefel Manifold Diffusion
            </span>
          </div>

          {/* Main Heading */}
          <h1 className="text-6xl sm:text-7xl lg:text-8xl font-bold mb-6 animate-slide-up">
            <span className="text-gradient">QuantumLab</span>
          </h1>

          <p className="text-xl sm:text-2xl text-gray-300 mb-4 max-w-3xl mx-auto animate-slide-up animation-delay-200">
            Discover Novel Quantum Materials with AI
          </p>

          <p className="text-lg text-gray-400 mb-12 max-w-2xl mx-auto animate-slide-up animation-delay-400">
            End-to-end platform for materials discovery combining generative AI,
            quantum chemistry, and intuitive visualization
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-16 animate-slide-up animation-delay-600">
            <Link href="/dashboard">
              <Button size="lg" className="glass-button text-lg group">
                Get Started
                <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Button>
            </Link>
            <Link href="/docs">
              <Button size="lg" variant="outline" className="glass-button text-lg">
                Documentation
              </Button>
            </Link>
          </div>

          {/* Feature Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto animate-slide-up animation-delay-800">
            <FeatureCard
              icon={<Database className="w-8 h-8 text-blue-400" />}
              title="Custom Datasets"
              description="Upload and manage your molecular datasets with ease"
            />
            <FeatureCard
              icon={<Zap className="w-8 h-8 text-purple-400" />}
              title="AI Models"
              description="Train custom generative models with your data"
            />
            <FeatureCard
              icon={<Sparkles className="w-8 h-8 text-pink-400" />}
              title="DFT Validation"
              description="Validate discoveries with xTB or full DFT calculations"
            />
          </div>
        </div>
      </div>
    </main>
  )
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode
  title: string
  description: string
}) {
  return (
    <div className="glass-card p-6 hover:scale-105 transition-transform duration-300 cursor-pointer group">
      <div className="flex flex-col items-center text-center">
        <div className="mb-4 p-3 rounded-xl bg-gradient-to-br from-blue-500/10 to-purple-500/10 group-hover:from-blue-500/20 group-hover:to-purple-500/20 transition-colors">
          {icon}
        </div>
        <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
        <p className="text-sm text-gray-400">{description}</p>
      </div>
    </div>
  )
}
