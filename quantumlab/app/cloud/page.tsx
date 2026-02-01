'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { 
  ArrowLeft, 
  Server, 
  Plus, 
  Key, 
  Cloud, 
  Terminal,
  Play,
  Square,
  Trash2,
  Eye,
  EyeOff,
  Check,
  Loader2,
  Activity,
  Cpu,
  HardDrive,
  Zap,
  Shield,
  Clock
} from 'lucide-react';

interface SSHCredential {
  id: number;
  name: string;
  host: string;
  port: number;
  username: string;
  auth_type: string;
  created_at: string;
  last_used_at?: string;
}

interface VM {
  id: number;
  name: string;
  provider: string;
  instance_type: string;
  status: string;
  public_ip?: string;
  region?: string;
  ssh_credential_id?: number;
  created_at: string;
}

interface TrainingSession {
  id: number;
  name: string;
  vm_instance_id: number;
  model_type: string;
  status: string;
  progress: number;
  start_time?: string;
  created_at: string;
}

export default function CloudPage() {
  const [activeTab, setActiveTab] = useState<'vms' | 'credentials' | 'training'>('vms');
  const [vms, setVMs] = useState<VM[]>([]);
  const [credentials, setCredentials] = useState<SSHCredential[]>([]);
  const [sessions, setSessions] = useState<TrainingSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCredentialModal, setShowCredentialModal] = useState(false);
  const [showVMModal, setShowVMModal] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [vmsRes, credsRes, sessionsRes] = await Promise.all([
        fetch('http://localhost:8000/cloud/vms'),
        fetch('http://localhost:8000/cloud/credentials/ssh'),
        fetch('http://localhost:8000/cloud/training')
      ]);

      const [vmsData, credsData, sessionsData] = await Promise.all([
        vmsRes.json(),
        credsRes.json(),
        sessionsRes.json()
      ]);

      if (vmsData.success) setVMs(vmsData.vms);
      if (credsData.success) setCredentials(credsData.credentials);
      if (sessionsData.success) setSessions(sessionsData.sessions);
    } catch (error) {
      console.error('Failed to load data:', error);
    } finally {
      setLoading(false);
    }
  };

  const openSSHTerminal = (vmId: number) => {
    // Open SSH terminal in new window or modal
    window.open(`/cloud/terminal/${vmId}`, '_blank', 'width=1000,height=600');
  };

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white">
      {/* Header */}
      <nav className="border-b border-white/10 bg-black/40 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/dashboard" className="p-2 hover:bg-white/5 rounded-lg transition-colors">
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <div>
              <h1 className="text-xl font-semibold">Cloud Training</h1>
              <p className="text-xs text-gray-500">Manage VMs, credentials, and training sessions</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowCredentialModal(true)}
              className="px-4 py-2 rounded-full glass hover:bg-white/10 transition-colors text-sm font-medium flex items-center gap-2"
            >
              <Key className="w-4 h-4" />
              Add Credentials
            </button>
            <button
              onClick={() => setShowVMModal(true)}
              className="px-4 py-2 rounded-full bg-gradient-to-r from-purple-600 to-pink-600 hover:opacity-90 transition-opacity text-sm font-medium flex items-center gap-2"
            >
              <Plus className="w-4 h-4" />
              Register VM
            </button>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <StatCard 
            label="Active VMs" 
            value={vms.filter(v => v.status === 'running').length} 
            total={vms.length}
            icon={<Server className="w-4 h-4" />}
            color="purple"
          />
          <StatCard 
            label="Saved Credentials" 
            value={credentials.length}
            icon={<Shield className="w-4 h-4" />}
            color="blue"
          />
          <StatCard 
            label="Training Sessions" 
            value={sessions.filter(s => s.status === 'running').length}
            total={sessions.length}
            icon={<Activity className="w-4 h-4" />}
            color="green"
          />
          <StatCard 
            label="Total GPU Hours" 
            value="127"
            icon={<Zap className="w-4 h-4" />}
            color="amber"
          />
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6 glass rounded-xl p-1">
          <TabButton 
            active={activeTab === 'vms'} 
            onClick={() => setActiveTab('vms')}
            icon={<Server className="w-4 h-4" />}
            label="Virtual Machines"
            badge={vms.length}
          />
          <TabButton 
            active={activeTab === 'credentials'} 
            onClick={() => setActiveTab('credentials')}
            icon={<Key className="w-4 h-4" />}
            label="SSH Credentials"
            badge={credentials.length}
          />
          <TabButton 
            active={activeTab === 'training'} 
            onClick={() => setActiveTab('training')}
            icon={<Activity className="w-4 h-4" />}
            label="Training Sessions"
            badge={sessions.length}
          />
        </div>

        {/* Content */}
        {loading ? (
          <div className="text-center py-20">
            <Loader2 className="w-8 h-8 animate-spin text-purple-500 mx-auto mb-4" />
            <p className="text-gray-500">Loading...</p>
          </div>
        ) : (
          <>
            {activeTab === 'vms' && (
              <VMList vms={vms} onOpenTerminal={openSSHTerminal} onRefresh={loadData} />
            )}
            {activeTab === 'credentials' && (
              <CredentialsList credentials={credentials} onRefresh={loadData} />
            )}
            {activeTab === 'training' && (
              <TrainingSessionsList sessions={sessions} vms={vms} onRefresh={loadData} />
            )}
          </>
        )}
      </div>

      {/* Modals */}
      {showCredentialModal && (
        <AddCredentialModal onClose={() => setShowCredentialModal(false)} onSuccess={loadData} />
      )}
      {showVMModal && (
        <AddVMModal credentials={credentials} onClose={() => setShowVMModal(false)} onSuccess={loadData} />
      )}
    </div>
  );
}

function StatCard({ label, value, total, icon, color }: any) {
  const colors = {
    purple: 'from-purple-500/20 to-purple-600/10 text-purple-400',
    blue: 'from-blue-500/20 to-blue-600/10 text-blue-400',
    green: 'from-green-500/20 to-green-600/10 text-green-400',
    amber: 'from-amber-500/20 to-amber-600/10 text-amber-400',
  };

  return (
    <div className="glass rounded-xl p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-gray-500">{label}</span>
        <div className={`p-2 rounded-lg bg-gradient-to-br ${colors[color]}`}>
          {icon}
        </div>
      </div>
      <div className="flex items-baseline gap-2">
        <p className="text-2xl font-semibold text-white">{value}</p>
        {total !== undefined && <p className="text-sm text-gray-500">/ {total}</p>}
      </div>
    </div>
  );
}

function TabButton({ active, onClick, icon, label, badge }: any) {
  return (
    <button
      onClick={onClick}
      className={`flex-1 px-4 py-2.5 rounded-lg transition-all flex items-center justify-center gap-2 ${
        active 
          ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-lg' 
          : 'text-gray-400 hover:text-white hover:bg-white/5'
      }`}
    >
      {icon}
      <span className="font-medium text-sm">{label}</span>
      {badge !== undefined && (
        <span className={`px-2 py-0.5 rounded-full text-xs ${
          active ? 'bg-white/20' : 'bg-white/10'
        }`}>
          {badge}
        </span>
      )}
    </button>
  );
}

function VMList({ vms, onOpenTerminal, onRefresh }: any) {
  if (vms.length === 0) {
    return (
      <div className="text-center py-20 glass rounded-2xl">
        <Server className="w-16 h-16 text-gray-600 mx-auto mb-4" />
        <h3 className="text-xl font-semibold mb-2">No VMs registered</h3>
        <p className="text-gray-500 mb-6">Register a VM to start training large models</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {vms.map(vm => (
        <div key={vm.id} className="glass rounded-xl p-5">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h3 className="text-lg font-semibold text-white mb-1">{vm.name}</h3>
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <Cloud className="w-3 h-3" />
                <span>{vm.provider}</span>
                <span>•</span>
                <span>{vm.instance_type}</span>
              </div>
            </div>
            <StatusBadge status={vm.status} />
          </div>

          {vm.public_ip && (
            <div className="mb-4 px-3 py-2 rounded-lg bg-white/5 text-sm text-gray-300 font-mono">
              {vm.public_ip}
            </div>
          )}

          <div className="flex gap-2">
            <button
              onClick={() => onOpenTerminal(vm.id)}
              disabled={vm.status !== 'running'}
              className="flex-1 px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm font-medium flex items-center justify-center gap-2"
            >
              <Terminal className="w-4 h-4" />
              Open SSH
            </button>
            <button
              className="px-4 py-2 rounded-lg glass hover:bg-white/10 transition-colors"
              title="VM Details"
            >
              <Eye className="w-4 h-4" />
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

function CredentialsList({ credentials, onRefresh }: any) {
  const [showPassword, setShowPassword] = useState<{[key: number]: boolean}>({});

  if (credentials.length === 0) {
    return (
      <div className="text-center py-20 glass rounded-2xl">
        <Key className="w-16 h-16 text-gray-600 mx-auto mb-4" />
        <h3 className="text-xl font-semibold mb-2">No credentials saved</h3>
        <p className="text-gray-500">Add SSH credentials to connect to VMs</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {credentials.map(cred => (
        <div key={cred.id} className="glass rounded-xl p-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-lg bg-gradient-to-br from-blue-500/20 to-blue-600/10">
              <Shield className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <h3 className="font-semibold text-white">{cred.name}</h3>
              <p className="text-sm text-gray-500">
                {cred.username}@{cred.host}:{cred.port}
              </p>
              <p className="text-xs text-gray-600 mt-1">
                Type: {cred.auth_type === 'key' ? 'Private Key' : 'Password'}
                {cred.last_used_at && ` • Last used: ${new Date(cred.last_used_at).toLocaleDateString()}`}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button className="p-2 rounded-lg glass hover:bg-white/10 transition-colors">
              <Eye className="w-4 h-4 text-gray-400" />
            </button>
            <button className="p-2 rounded-lg glass hover:bg-red-500/10 hover:text-red-400 transition-colors">
              <Trash2 className="w-4 h-4 text-gray-400" />
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

function TrainingSessionsList({ sessions, vms, onRefresh }: any) {
  if (sessions.length === 0) {
    return (
      <div className="text-center py-20 glass rounded-2xl">
        <Activity className="w-16 h-16 text-gray-600 mx-auto mb-4" />
        <h3 className="text-xl font-semibold mb-2">No training sessions</h3>
        <p className="text-gray-500">Start a training session on a VM</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {sessions.map(session => {
        const vm = vms.find((v: VM) => v.id === session.vm_instance_id);
        return (
          <div key={session.id} className="glass rounded-xl p-5">
            <div className="flex items-start justify-between mb-3">
              <div>
                <h3 className="font-semibold text-white mb-1">{session.name}</h3>
                <p className="text-sm text-gray-500">{session.model_type} • {vm?.name || 'Unknown VM'}</p>
              </div>
              <StatusBadge status={session.status} />
            </div>

            {/* Progress bar */}
            {session.status === 'running' && (
              <div className="mb-3">
                <div className="flex justify-between text-xs text-gray-500 mb-1">
                  <span>Progress</span>
                  <span>{(session.progress * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-purple-600 to-pink-600 transition-all duration-300"
                    style={{ width: `${session.progress * 100}%` }}
                  />
                </div>
              </div>
            )}

            <div className="flex items-center gap-4 text-xs text-gray-600">
              <div className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                Started {new Date(session.created_at).toLocaleString()}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const styles = {
    running: 'bg-green-500/20 text-green-400 border-green-500/30',
    stopped: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
    pending: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    failed: 'bg-red-500/20 text-red-400 border-red-500/30',
    completed: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  };

  return (
    <span className={`px-3 py-1 rounded-full text-xs font-medium border ${styles[status as keyof typeof styles] || styles.stopped}`}>
      {status}
    </span>
  );
}

function AddCredentialModal({ onClose, onSuccess }: any) {
  const [formData, setFormData] = useState({
    name: '',
    host: '',
    port: 22,
    username: '',
    auth_type: 'password',
    password: '',
    private_key: '',
    passphrase: ''
  });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/cloud/credentials/ssh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      if (response.ok) {
        onSuccess();
        onClose();
      }
    } catch (error) {
      console.error('Failed to save credentials:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="glass rounded-2xl p-6 max-w-md w-full">
        <h2 className="text-xl font-semibold mb-4">Add SSH Credentials</h2>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm text-gray-400 mb-2">Name</label>
            <input
              type="text"
              value={formData.name}
              onChange={e => setFormData({...formData, name: e.target.value})}
              className="w-full px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white"
              required
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">Host</label>
              <input
                type="text"
                value={formData.host}
                onChange={e => setFormData({...formData, host: e.target.value})}
                className="w-full px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white"
                required
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">Port</label>
              <input
                type="number"
                value={formData.port}
                onChange={e => setFormData({...formData, port: parseInt(e.target.value)})}
                className="w-full px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white"
                required
              />
            </div>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">Username</label>
            <input
              type="text"
              value={formData.username}
              onChange={e => setFormData({...formData, username: e.target.value})}
              className="w-full px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white"
              required
            />
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">Authentication Type</label>
            <select
              value={formData.auth_type}
              onChange={e => setFormData({...formData, auth_type: e.target.value})}
              className="w-full px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white"
            >
              <option value="password">Password</option>
              <option value="key">Private Key</option>
            </select>
          </div>

          {formData.auth_type === 'password' ? (
            <div>
              <label className="block text-sm text-gray-400 mb-2">Password</label>
              <input
                type="password"
                value={formData.password}
                onChange={e => setFormData({...formData, password: e.target.value})}
                className="w-full px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white"
                required
              />
            </div>
          ) : (
            <div>
              <label className="block text-sm text-gray-400 mb-2">Private Key</label>
              <textarea
                value={formData.private_key}
                onChange={e => setFormData({...formData, private_key: e.target.value})}
                className="w-full px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white font-mono text-xs"
                rows={4}
                placeholder="-----BEGIN RSA PRIVATE KEY-----"
                required
              />
            </div>
          )}

          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 rounded-xl glass hover:bg-white/10 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="flex-1 px-4 py-2 rounded-xl bg-gradient-to-r from-purple-600 to-pink-600 hover:opacity-90 transition-opacity disabled:opacity-50"
            >
              {loading ? 'Saving...' : 'Save'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function AddVMModal({ credentials, onClose, onSuccess }: any) {
  const [formData, setFormData] = useState({
    name: '',
    provider: 'custom',
    instance_type: '',
    ssh_credential_id: '',
    public_ip: '',
    region: ''
  });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/cloud/vms/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...formData,
          ssh_credential_id: formData.ssh_credential_id ? parseInt(formData.ssh_credential_id) : null
        })
      });

      if (response.ok) {
        onSuccess();
        onClose();
      }
    } catch (error) {
      console.error('Failed to register VM:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="glass rounded-2xl p-6 max-w-md w-full">
        <h2 className="text-xl font-semibold mb-4">Register VM Instance</h2>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm text-gray-400 mb-2">VM Name</label>
            <input
              type="text"
              value={formData.name}
              onChange={e => setFormData({...formData, name: e.target.value})}
              placeholder="e.g., gpu-training-01"
              className="w-full px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white"
              required
            />
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">Provider</label>
            <select
              value={formData.provider}
              onChange={e => setFormData({...formData, provider: e.target.value})}
              className="w-full px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white"
            >
              <option value="aws">AWS</option>
              <option value="gcp">Google Cloud</option>
              <option value="azure">Azure</option>
              <option value="custom">Custom / On-Premise</option>
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">Instance Type</label>
              <input
                type="text"
                value={formData.instance_type}
                onChange={e => setFormData({...formData, instance_type: e.target.value})}
                placeholder="e.g., p3.2xlarge"
                className="w-full px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white"
                required
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">Region</label>
              <input
                type="text"
                value={formData.region}
                onChange={e => setFormData({...formData, region: e.target.value})}
                placeholder="e.g., us-east-1"
                className="w-full px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">Public IP</label>
            <input
              type="text"
              value={formData.public_ip}
              onChange={e => setFormData({...formData, public_ip: e.target.value})}
              placeholder="e.g., 54.123.45.67"
              className="w-full px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white"
            />
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">SSH Credentials</label>
            <select
              value={formData.ssh_credential_id}
              onChange={e => setFormData({...formData, ssh_credential_id: e.target.value})}
              className="w-full px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white"
            >
              <option value="">Select credentials...</option>
              {credentials.map((cred: any) => (
                <option key={cred.id} value={cred.id}>
                  {cred.name} ({cred.username}@{cred.host})
                </option>
              ))}
            </select>
          </div>

          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 rounded-xl glass hover:bg-white/10 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="flex-1 px-4 py-2 rounded-xl bg-gradient-to-r from-purple-600 to-pink-600 hover:opacity-90 transition-opacity disabled:opacity-50"
            >
              {loading ? 'Registering...' : 'Register'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
