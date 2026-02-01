import Link from "next/link"
import { Atom, Upload, Cpu, FlaskConical, LineChart, Settings, User, Bell, Shield, Database, Palette, Code } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function SettingsPage() {
  return (
    <div className="min-h-screen bg-gray-950">
      {/* Sidebar */}
      <aside className="fixed left-0 top-0 bottom-0 w-64 bg-gray-900/50 border-r border-white/5 p-6">
        <div className="flex items-center gap-2 mb-8">
          <Atom className="w-6 h-6 text-blue-400" />
          <span className="text-lg font-semibold text-white">QuantumLab</span>
        </div>

        <nav className="space-y-1">
          <NavItem href="/dashboard" icon={<LineChart className="w-4 h-4" />} label="Overview" />
          <NavItem href="/datasets" icon={<Upload className="w-4 h-4" />} label="Datasets" />
          <NavItem href="/models" icon={<Cpu className="w-4 h-4" />} label="Models" />
          <NavItem href="/compute" icon={<FlaskConical className="w-4 h-4" />} label="Compute" />
          <NavItem href="/results" icon={<LineChart className="w-4 h-4" />} label="Results" />
        </nav>

        <div className="absolute bottom-6 left-6 right-6">
          <NavItem href="/settings" icon={<Settings className="w-4 h-4" />} label="Settings" active />
        </div>
      </aside>

      {/* Main Content */}
      <main className="ml-64 p-8">
        <div className="max-w-4xl">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-2xl font-bold text-white mb-1">Settings</h1>
            <p className="text-gray-500 text-sm">Manage your account and preferences</p>
          </div>

          {/* Settings Sections */}
          <div className="space-y-6">
            <SettingsSection
              icon={<User className="w-5 h-5" />}
              title="Profile"
              description="Manage your account information"
            >
              <div className="space-y-4">
                <InputField label="Name" value="Koussai Salem" />
                <InputField label="Email" value="koussai@example.com" />
                <InputField label="Organization" value="Research Lab" />
              </div>
            </SettingsSection>

            <SettingsSection
              icon={<Database className="w-5 h-5" />}
              title="Storage"
              description="Manage data storage and quotas"
            >
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-400">Storage Used</span>
                    <span className="text-sm text-white">4.2 GB / 10 GB</span>
                  </div>
                  <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-500 w-[42%]" />
                  </div>
                </div>
                <div className="flex gap-3">
                  <Button size="sm" variant="outline" className="border-white/10 text-white hover:bg-white/10">
                    Clear Cache
                  </Button>
                  <Button size="sm" variant="outline" className="border-white/10 text-white hover:bg-white/10">
                    Export Data
                  </Button>
                </div>
              </div>
            </SettingsSection>

            <SettingsSection
              icon={<Code className="w-5 h-5" />}
              title="API Configuration"
              description="Configure backend connections"
            >
              <div className="space-y-4">
                <InputField label="Python Backend URL" value="http://localhost:8000" />
                <InputField label="xTB Binary Path" value="/usr/local/bin/xtb" />
                <InputField label="GPAW Config" value="default" />
              </div>
            </SettingsSection>

            <SettingsSection
              icon={<Bell className="w-5 h-5" />}
              title="Notifications"
              description="Configure notification preferences"
            >
              <div className="space-y-3">
                <ToggleOption label="Email notifications" description="Receive email when jobs complete" enabled />
                <ToggleOption label="Browser notifications" description="Show desktop notifications" enabled={false} />
                <ToggleOption label="Weekly digest" description="Receive weekly summary of activity" enabled />
              </div>
            </SettingsSection>

            <SettingsSection
              icon={<Palette className="w-5 h-5" />}
              title="Appearance"
              description="Customize the interface"
            >
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">Theme</label>
                  <div className="flex gap-3">
                    <ThemeOption label="Dark" active />
                    <ThemeOption label="Light" />
                    <ThemeOption label="System" />
                  </div>
                </div>
              </div>
            </SettingsSection>

            <SettingsSection
              icon={<Shield className="w-5 h-5" />}
              title="Security"
              description="Manage security settings"
            >
              <div className="space-y-3">
                <Button variant="outline" className="border-white/10 text-white hover:bg-white/10">
                  Change Password
                </Button>
                <Button variant="outline" className="border-white/10 text-white hover:bg-white/10">
                  Enable Two-Factor Authentication
                </Button>
              </div>
            </SettingsSection>
          </div>
        </div>
      </main>
    </div>
  )
}

function NavItem({ href, icon, label, active }: { href: string; icon: React.ReactNode; label: string; active?: boolean }) {
  return (
    <Link href={href}>
      <div className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${
        active 
          ? "bg-white/10 text-white" 
          : "text-gray-400 hover:text-white hover:bg-white/5"
      }`}>
        {icon}
        <span className="text-sm font-medium">{label}</span>
      </div>
    </Link>
  )
}

function SettingsSection({ icon, title, description, children }: { 
  icon: React.ReactNode
  title: string
  description: string
  children: React.ReactNode
}) {
  return (
    <div className="bg-gray-900/50 rounded-xl border border-white/5 p-6">
      <div className="flex items-start gap-4 mb-6">
        <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center text-blue-400">
          {icon}
        </div>
        <div>
          <h2 className="text-lg font-semibold text-white">{title}</h2>
          <p className="text-sm text-gray-500">{description}</p>
        </div>
      </div>
      {children}
    </div>
  )
}

function InputField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-400 mb-2">{label}</label>
      <input
        type="text"
        defaultValue={value}
        className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 transition-colors"
      />
    </div>
  )
}

function ToggleOption({ label, description, enabled }: { label: string; description: string; enabled: boolean }) {
  return (
    <div className="flex items-center justify-between py-2">
      <div>
        <p className="text-sm font-medium text-white">{label}</p>
        <p className="text-xs text-gray-500">{description}</p>
      </div>
      <button className={`w-10 h-6 rounded-full transition-colors ${enabled ? "bg-blue-500" : "bg-white/10"}`}>
        <div className={`w-4 h-4 rounded-full bg-white transform transition-transform ${enabled ? "translate-x-5" : "translate-x-1"}`} />
      </button>
    </div>
  )
}

function ThemeOption({ label, active }: { label: string; active?: boolean }) {
  return (
    <button className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
      active 
        ? "bg-blue-500 text-white" 
        : "bg-white/5 text-gray-400 hover:bg-white/10 hover:text-white"
    }`}>
      {label}
    </button>
  )
}
