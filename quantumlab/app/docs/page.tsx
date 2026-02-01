import Link from "next/link"
import { Atom, Book, Code, FileText, Terminal, Zap, ArrowLeft, Search, ChevronRight } from "lucide-react"

export default function DocsPage() {
  return (
    <div className="min-h-screen bg-gray-950">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-gray-950/80 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <Link href="/" className="flex items-center gap-2">
              <Atom className="w-6 h-6 text-blue-400" />
              <span className="text-xl font-semibold text-white">QuantumLab</span>
            </Link>
            <span className="text-gray-500">/</span>
            <span className="text-gray-400">Documentation</span>
          </div>
          <div className="flex items-center gap-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500" />
              <input
                type="text"
                placeholder="Search docs..."
                className="w-64 pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 transition-colors text-sm"
              />
            </div>
            <Link href="/dashboard" className="text-sm text-gray-400 hover:text-white transition-colors">
              Dashboard
            </Link>
          </div>
        </div>
      </nav>

      <div className="flex pt-16">
        {/* Sidebar */}
        <aside className="fixed left-0 top-16 bottom-0 w-64 border-r border-white/5 p-6 overflow-y-auto">
          <nav className="space-y-6">
            <DocSection title="Getting Started">
              <DocLink href="/docs/introduction" label="Introduction" />
              <DocLink href="/docs/installation" label="Installation" />
              <DocLink href="/docs/quickstart" label="Quick Start" />
            </DocSection>

            <DocSection title="Datasets">
              <DocLink href="/docs/datasets/upload" label="Uploading Data" />
              <DocLink href="/docs/datasets/formats" label="Supported Formats" />
              <DocLink href="/docs/datasets/preprocessing" label="Preprocessing" />
            </DocSection>

            <DocSection title="Models">
              <DocLink href="/docs/models/surrogate" label="Surrogate Models" />
              <DocLink href="/docs/models/score" label="Score Models" />
              <DocLink href="/docs/models/training" label="Training Guide" />
            </DocSection>

            <DocSection title="Compute">
              <DocLink href="/docs/compute/xtb" label="xTB Calculations" />
              <DocLink href="/docs/compute/dft" label="DFT Calculations" />
              <DocLink href="/docs/compute/optimization" label="Optimization" />
            </DocSection>

            <DocSection title="API Reference">
              <DocLink href="/docs/api/rest" label="REST API" />
              <DocLink href="/docs/api/python" label="Python Client" />
            </DocSection>
          </nav>
        </aside>

        {/* Main Content */}
        <main className="ml-64 flex-1 p-12">
          <div className="max-w-3xl">
            <Link href="/" className="inline-flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors mb-8">
              <ArrowLeft className="w-4 h-4" />
              Back to Home
            </Link>

            <h1 className="text-4xl font-bold text-white mb-4">Documentation</h1>
            <p className="text-lg text-gray-400 mb-12">
              Learn how to use QuantumLab for quantum materials discovery
            </p>

            {/* Quick Links */}
            <div className="grid grid-cols-2 gap-6 mb-12">
              <QuickLinkCard
                icon={<Zap className="w-5 h-5" />}
                title="Quick Start"
                description="Get up and running in 5 minutes"
                href="/docs/quickstart"
              />
              <QuickLinkCard
                icon={<Book className="w-5 h-5" />}
                title="User Guide"
                description="Complete guide to all features"
                href="/docs/guide"
              />
              <QuickLinkCard
                icon={<Code className="w-5 h-5" />}
                title="API Reference"
                description="Full API documentation"
                href="/docs/api"
              />
              <QuickLinkCard
                icon={<Terminal className="w-5 h-5" />}
                title="CLI Reference"
                description="Command line interface docs"
                href="/docs/cli"
              />
            </div>

            {/* Introduction Content */}
            <div className="prose prose-invert max-w-none">
              <h2 className="text-2xl font-bold text-white mb-4">Introduction</h2>
              <p className="text-gray-400 mb-6">
                QuantumLab is a professional-grade platform for end-to-end quantum materials discovery. 
                It combines state-of-the-art generative AI with quantum chemistry calculations to accelerate 
                the discovery of novel materials.
              </p>

              <h3 className="text-xl font-semibold text-white mb-3">Key Features</h3>
              <ul className="space-y-2 text-gray-400 mb-6">
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 mt-1 text-blue-400 flex-shrink-0" />
                  <span>Upload and manage molecular datasets with built-in validation</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 mt-1 text-blue-400 flex-shrink-0" />
                  <span>Train custom surrogate and generative models</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 mt-1 text-blue-400 flex-shrink-0" />
                  <span>Run xTB and DFT calculations with real-time monitoring</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 mt-1 text-blue-400 flex-shrink-0" />
                  <span>Visualize results with interactive 3D molecule viewers</span>
                </li>
              </ul>

              <h3 className="text-xl font-semibold text-white mb-3">Architecture</h3>
              <p className="text-gray-400 mb-6">
                The platform is built on the QCMD-ECS framework, which implements Stiefel manifold 
                diffusion for generating physically valid molecular structures. The web interface 
                connects to a Python backend that handles all computational tasks.
              </p>

              <div className="bg-gray-900/50 rounded-xl border border-white/5 p-6">
                <h4 className="text-lg font-semibold text-white mb-3">Next Steps</h4>
                <div className="space-y-2">
                  <Link href="/docs/installation" className="flex items-center justify-between p-3 rounded-lg hover:bg-white/5 transition-colors">
                    <span className="text-white">Installation Guide</span>
                    <ChevronRight className="w-4 h-4 text-gray-500" />
                  </Link>
                  <Link href="/docs/quickstart" className="flex items-center justify-between p-3 rounded-lg hover:bg-white/5 transition-colors">
                    <span className="text-white">Quick Start Tutorial</span>
                    <ChevronRight className="w-4 h-4 text-gray-500" />
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}

function DocSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">{title}</h3>
      <div className="space-y-1">{children}</div>
    </div>
  )
}

function DocLink({ href, label, active }: { href: string; label: string; active?: boolean }) {
  return (
    <Link href={href}>
      <div className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
        active ? "bg-blue-500/20 text-blue-400" : "text-gray-400 hover:text-white hover:bg-white/5"
      }`}>
        {label}
      </div>
    </Link>
  )
}

function QuickLinkCard({ icon, title, description, href }: {
  icon: React.ReactNode
  title: string
  description: string
  href: string
}) {
  return (
    <Link href={href}>
      <div className="bg-gray-900/50 rounded-xl border border-white/5 p-6 hover:border-white/20 transition-colors">
        <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center text-blue-400 mb-4">
          {icon}
        </div>
        <h3 className="text-lg font-semibold text-white mb-1">{title}</h3>
        <p className="text-sm text-gray-400">{description}</p>
      </div>
    </Link>
  )
}
