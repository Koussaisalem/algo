'use client'

import { useState } from 'react'
import Link from "next/link"
import { Atom, Upload, Cpu, FlaskConical, LineChart, Settings, User, Bell, Shield, Database, Palette, Code, Check, Keyboard } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ThemeToggle } from "@/components/ui/theme-toggle"
import { Breadcrumb } from "@/components/ui/breadcrumb"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { useToast } from "@/components/providers/toast-provider"
import { Kbd } from "@/components/ui/kbd"

export default function SettingsPage() {
  const { addToast } = useToast()
  const [saved, setSaved] = useState(false)

  const handleSave = () => {
    setSaved(true)
    addToast({
      type: 'success',
      title: 'Settings saved',
      message: 'Your preferences have been updated successfully.'
    })
    setTimeout(() => setSaved(false), 2000)
  }

  return (
    <div className="min-h-screen bg-[#0a0a0f]">
      {/* Ambient background */}
      <div className="fixed inset-0 bg-gradient-to-br from-blue-950/20 via-transparent to-purple-950/20" />
      <div className="fixed inset-0 bg-grid opacity-30" />

      {/* Sidebar */}
      <aside className="fixed left-0 top-0 bottom-0 w-72 glass-subtle p-6 z-50">
        <Link href="/" className="flex items-center gap-3 mb-10 px-2">
          <div className="p-2 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 shadow-lg shadow-blue-500/20">
            <Atom className="w-5 h-5 text-white" />
          </div>
          <span className="text-lg font-semibold text-white tracking-tight">QuantumLab</span>
        </Link>

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
      <main className="relative z-10 ml-72 p-10">
        <div className="max-w-4xl">
          <Breadcrumb items={[{ label: 'Settings' }]} />
          
          {/* Header */}
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-3xl font-bold text-gradient">Settings</h1>
              <p className="text-neutral-400 mt-2">Manage your account and preferences</p>
            </div>
            <Button onClick={handleSave} className="btn-shine">
              {saved ? <><Check className="w-4 h-4 mr-2" /> Saved</> : 'Save Changes'}
            </Button>
          </div>

          <Tabs defaultValue="general" className="space-y-6">
            <TabsList>
              <TabsTrigger value="general">General</TabsTrigger>
              <TabsTrigger value="compute">Compute</TabsTrigger>
              <TabsTrigger value="notifications">Notifications</TabsTrigger>
              <TabsTrigger value="shortcuts">Shortcuts</TabsTrigger>
            </TabsList>

            <TabsContent value="general" className="space-y-6">
              <SettingsSection
                icon={<User className="w-5 h-5" />}
                title="Profile"
                description="Manage your account information"
              >
                <div className="space-y-4">
                  <InputField label="Name" defaultValue="Koussai Salem" />
                  <InputField label="Email" defaultValue="koussai@example.com" />
                  <InputField label="Organization" defaultValue="Research Lab" />
                </div>
              </SettingsSection>

              <SettingsSection
                icon={<Palette className="w-5 h-5" />}
                title="Appearance"
                description="Customize the interface"
              >
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-neutral-400 mb-3">Theme</label>
                    <ThemeToggle />
                  </div>
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
                      <span className="text-sm text-neutral-400">Storage Used</span>
                      <span className="text-sm text-white">4.2 GB / 10 GB</span>
                    </div>
                    <Progress value={42} color="blue" showLabel={false} />
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
            </TabsContent>

            <TabsContent value="compute" className="space-y-6">
              <SettingsSection
                icon={<Code className="w-5 h-5" />}
                title="Backend Configuration"
                description="Configure compute backend connections"
              >
                <div className="space-y-4">
                  <InputField label="Python Backend URL" defaultValue="http://localhost:8000" />
                  <InputField label="xTB Binary Path" defaultValue="/usr/local/bin/xtb" />
                  <InputField label="GPAW Config" defaultValue="default" />
                </div>
              </SettingsSection>

              <SettingsSection
                icon={<Cpu className="w-5 h-5" />}
                title="GPU Settings"
                description="Configure GPU acceleration"
              >
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/10">
                    <div>
                      <p className="text-white font-medium">NVIDIA RTX 4090</p>
                      <p className="text-sm text-neutral-400">24GB VRAM</p>
                    </div>
                    <Badge variant="success" dot>Active</Badge>
                  </div>
                  <ToggleOption label="Enable CUDA acceleration" description="Use GPU for computations" enabled />
                  <ToggleOption label="Mixed precision training" description="FP16 for faster training" enabled />
                </div>
              </SettingsSection>
            </TabsContent>

            <TabsContent value="notifications" className="space-y-6">
              <SettingsSection
                icon={<Bell className="w-5 h-5" />}
                title="Notification Preferences"
                description="Configure how you receive notifications"
              >
                <div className="space-y-3">
                  <ToggleOption label="Email notifications" description="Receive email when jobs complete" enabled />
                  <ToggleOption label="Browser notifications" description="Show desktop notifications" enabled={false} />
                  <ToggleOption label="Weekly digest" description="Receive weekly summary of activity" enabled />
                  <ToggleOption label="Error alerts" description="Get notified when jobs fail" enabled />
                </div>
              </SettingsSection>
            </TabsContent>

            <TabsContent value="shortcuts" className="space-y-6">
              <SettingsSection
                icon={<Keyboard className="w-5 h-5" />}
                title="Keyboard Shortcuts"
                description="Quick actions for power users"
              >
                <div className="space-y-3">
                  <ShortcutRow action="Open command palette" keys={['⌘', 'K']} />
                  <ShortcutRow action="Go to Dashboard" keys={['G', 'D']} />
                  <ShortcutRow action="Go to Datasets" keys={['G', 'S']} />
                  <ShortcutRow action="Go to Models" keys={['G', 'M']} />
                  <ShortcutRow action="Go to Compute" keys={['G', 'C']} />
                  <ShortcutRow action="Go to Results" keys={['G', 'R']} />
                  <ShortcutRow action="Save settings" keys={['⌘', 'S']} />
                </div>
              </SettingsSection>
            </TabsContent>
          </Tabs>

          {/* Security Section - always visible */}
          <div className="mt-6">
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
      <div className={`flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all ${
        active 
          ? "bg-white/10 text-white" 
          : "text-neutral-400 hover:text-white hover:bg-white/5"
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
    <div className="glass rounded-2xl p-6">
      <div className="flex items-start gap-4 mb-6">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 flex items-center justify-center text-blue-400">
          {icon}
        </div>
        <div>
          <h2 className="text-lg font-semibold text-white">{title}</h2>
          <p className="text-sm text-neutral-500">{description}</p>
        </div>
      </div>
      {children}
    </div>
  )
}

function InputField({ label, defaultValue }: { label: string; defaultValue: string }) {
  return (
    <div>
      <label className="block text-sm font-medium text-neutral-400 mb-2">{label}</label>
      <input
        type="text"
        defaultValue={defaultValue}
        className="w-full px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white placeholder-neutral-500 focus:outline-none focus:border-blue-500/50 focus:bg-white/[0.07] transition-all"
      />
    </div>
  )
}

function ToggleOption({ label, description, enabled }: { label: string; description: string; enabled: boolean }) {
  const [isEnabled, setIsEnabled] = useState(enabled)
  
  return (
    <div className="flex items-center justify-between py-3 px-4 rounded-xl hover:bg-white/5 transition-colors">
      <div>
        <p className="text-sm font-medium text-white">{label}</p>
        <p className="text-xs text-neutral-500">{description}</p>
      </div>
      <button 
        onClick={() => setIsEnabled(!isEnabled)}
        className={`w-11 h-6 rounded-full transition-all ${isEnabled ? "bg-blue-500 shadow-lg shadow-blue-500/30" : "bg-white/10"}`}
      >
        <div className={`w-5 h-5 rounded-full bg-white shadow transform transition-transform ${isEnabled ? "translate-x-5" : "translate-x-0.5"}`} />
      </button>
    </div>
  )
}

function ShortcutRow({ action, keys }: { action: string; keys: string[] }) {
  return (
    <div className="flex items-center justify-between py-3 px-4 rounded-xl hover:bg-white/5 transition-colors">
      <span className="text-sm text-white">{action}</span>
      <div className="flex items-center gap-1">
        {keys.map((key, i) => (
          <Kbd key={i}>{key}</Kbd>
        ))}
      </div>
    </div>
  )
}
