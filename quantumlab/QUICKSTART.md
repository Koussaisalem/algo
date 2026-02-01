# QuantumLab Platform

> **Quantum Chemical Molecular Design with Stiefel Manifold Optimization**

A cutting-edge web platform for AI-driven molecular generation with property-guided design, real-time inference, and cloud training capabilities.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Node 20+](https://img.shields.io/badge/node-20+-green.svg)](https://nodejs.org/)
[![PostgreSQL 16+](https://img.shields.io/badge/postgresql-16+-blue.svg)](https://www.postgresql.org/)

---

## Quick Start (3 Commands)

```bash
# Clone the repository
git clone https://github.com/Koussaisalem/algo.git
cd algo/quantumlab

# Run automated setup (installs everything)
chmod +x setup.sh && ./setup.sh

# Start the platform
./start_all.sh
```

**That's it!** Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Docker Quick Start (Even Faster!)

```bash
# Clone and start with Docker Compose
git clone https://github.com/Koussaisalem/algo.git
cd algo/quantumlab

# Create environment file
cp .env.example .env.local
# Edit .env.local with your secrets (or use the generated ones)

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

Access at [http://localhost:3000](http://localhost:3000) • API at [http://localhost:8000](http://localhost:8000)

---

## Features

### **Molecular Generation**
- **Stiefel Manifold Optimization**: Mathematically rigorous geometry-preserving diffusion
- **Property-Guided Design**: Target specific properties (band gap, energy, etc.)
- **Real-time Inference**: Generate molecules on-demand with trained models
- **3D Visualization**: Interactive molecule viewer with WebGL rendering

### **Molecule Library**
- Save generated molecules with metadata
- Advanced search & filtering
- Export to SDF, MOL, XYZ formats
- Batch operations & favorites

### **Cloud Training**
- **VM Management**: Register and manage cloud compute instances
- **SSH Terminal**: Built-in web terminal with xterm.js
- **Secure Vault**: Encrypted credential storage with Fernet encryption
- **Training Sessions**: Track long-running training jobs

### **System Monitoring**
- Auto-detect runtime environment (Docker/Codespace/AWS/GCP/Azure/Local)
- Real-time GPU/CPU/RAM monitoring
- Storage & network statistics
- Smart resource recommendations

### **Authentication**
- NextAuth.js with PostgreSQL backend
- Email/password authentication
- OAuth ready (GitHub, Google)
- Protected routes & session management

---

## Requirements

### System Requirements
- **OS**: Linux, macOS, or Windows (WSL2)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **GPU**: Optional (NVIDIA/AMD for faster training)

### Software Dependencies
- **Node.js**: 20.x or higher
- **Python**: 3.12 or higher
- **PostgreSQL**: 16.x or higher
- **npm**: 10.x or higher

### Python Packages (auto-installed)
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
torch>=2.0.0
schnetpack>=2.0.0
psutil>=5.9.0
paramiko>=3.3.0
cryptography>=41.0.0
sqlalchemy>=2.0.0
asyncpg>=0.29.0
pg8000>=1.30.0
```

### Node.js Packages (auto-installed)
```
next@14.x
react@18.x
typescript@5.x
tailwindcss@3.x
next-auth@4.x
pg@8.x
bcrypt@5.x
```

---

## Manual Installation

If you prefer manual setup or the script doesn't work:

### 1. Clone Repository
```bash
git clone https://github.com/Koussaisalem/algo.git
cd algo/quantumlab
```

### 2. Setup Environment
```bash
cp .env.example .env.local
# Edit .env.local with your configuration
```

### 3. Install PostgreSQL
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install postgresql-16

# macOS (Homebrew)
brew install postgresql@16

# Start service
sudo service postgresql start # Linux
brew services start postgresql@16 # macOS
```

### 4. Create Database
```bash
sudo -u postgres psql -c "CREATE DATABASE quantumlab;"
sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD 'your_password';"
```

### 5. Install Frontend Dependencies
```bash
npm install --legacy-peer-deps
```

### 6. Install Backend Dependencies
```bash
cd backend
python3 -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

### 7. Initialize Database
```bash
node -e "
const { Pool } = require('pg');
const pool = new Pool({ connectionString: process.env.DATABASE_URL });
(async () => {
 await pool.query(\`
 CREATE TABLE users (
 id SERIAL PRIMARY KEY,
 email VARCHAR(255) UNIQUE NOT NULL,
 name VARCHAR(255),
 password VARCHAR(255) NOT NULL,
 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
 );
 CREATE TABLE sessions (
 id SERIAL PRIMARY KEY,
 user_id INTEGER REFERENCES users(id),
 session_token VARCHAR(255) UNIQUE NOT NULL,
 expires TIMESTAMP NOT NULL
 );
 \`);
 await pool.end();
})();
"
```

### 8. Start Services
```bash
# Terminal 1: Backend
cd backend && source venv/bin/activate && python inference_server.py

# Terminal 2: Frontend
npm run dev
```

---

## Usage Guide

### First Time Setup
1. Navigate to [http://localhost:3000](http://localhost:3000)
2. Click **"Get Started"**
3. **Sign up** with email/password
4. You'll be auto-logged in and redirected to the dashboard

### Generate Molecules
1. Go to **Inference** tab
2. Select a trained model (or use dummy for testing)
3. Set target properties (e.g., band gap: 2.5 eV)
4. Click **"Generate"**
5. View 3D structure and predicted properties
6. Save to library

### Manage Molecule Library
1. Go to **Library** tab
2. Search by name, formula, or properties
3. Filter by date, property ranges
4. Export selected molecules
5. Star favorites

### Cloud Training
1. Go to **Cloud** tab
2. Add SSH credentials (encrypted in vault)
3. Register VM instances
4. Create training session
5. Open SSH terminal to monitor
6. Track progress in dashboard

### System Monitoring
1. Go to **Settings** → **Compute**
2. View real-time GPU/CPU/RAM usage
3. Check storage and network stats
4. Review system recommendations

---

## Architecture

```
quantumlab/
 app/ # Next.js 14 App Router
 auth/ # Login/signup pages
 cloud/ # Cloud training UI
 compute/ # System monitoring
 inference/ # Molecule generation
 library/ # Molecule database
 backend/ # FastAPI Python backend
 inference_server.py
 system_detect.py
 vault.py
 components/ # Reusable React components
 lib/ # Utilities
 auth.ts # NextAuth config
 db.ts # PostgreSQL client
 public/ # Static assets
```

### Tech Stack
- **Frontend**: Next.js 14, React 18, TypeScript, Tailwind CSS
- **Backend**: FastAPI, Python 3.12, PyTorch, SchNetPack
- **Database**: PostgreSQL 16 (users, sessions), SQLite (molecules, vault)
- **Auth**: NextAuth.js with JWT sessions, bcrypt hashing
- **ML**: QCMD-ECS diffusion framework with Stiefel manifold
- **Deployment**: Docker Compose, Vercel (frontend), Railway (backend)

---

## Configuration

### Environment Variables

**`.env.local`** (Frontend & Backend):
```bash
# Database
POSTGRES_PASSWORD=your_secure_password
DATABASE_URL=postgresql://postgres:password@localhost:5432/quantumlab

# Auth
NEXTAUTH_SECRET=generate_with_openssl_rand_base64_32
NEXTAUTH_URL=http://localhost:3000

# Backend
BACKEND_URL=http://localhost:8000

# Vault
VAULT_MASTER_KEY=generate_with_openssl_rand_base64_32
```

Generate secrets:
```bash
openssl rand -base64 32 # For NEXTAUTH_SECRET
openssl rand -base64 32 # For VAULT_MASTER_KEY
```

---

## Troubleshooting

### Port Already in Use
```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

### PostgreSQL Connection Failed
```bash
# Check if PostgreSQL is running
sudo service postgresql status

# Restart PostgreSQL
sudo service postgresql restart

# Check connection
psql -U postgres -d quantumlab -c "SELECT 1;"
```

### Frontend Build Errors
```bash
# Clear cache and reinstall
rm -rf node_modules .next
npm install --legacy-peer-deps
```

### Backend Module Not Found
```bash
cd backend
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Database Schema Issues
```bash
# Recreate tables
sudo -u postgres psql quantumlab -c "DROP TABLE IF EXISTS sessions, users CASCADE;"
# Then re-run initialization from setup.sh
```

---

## Documentation

- **Architecture**: See [docs/architecture/OVERVIEW.md](docs/architecture/OVERVIEW.md)
- **Theory**: See [docs/theory/STIEFEL_MANIFOLD_THEORY.md](docs/theory/STIEFEL_MANIFOLD_THEORY.md)
- **Scripts**: See [docs/guides/SCRIPTS_REFERENCE.md](docs/guides/SCRIPTS_REFERENCE.md)
- **Training**: See [docs/guides/TRAINING_OPTIMIZATION_GUIDE.md](docs/guides/TRAINING_OPTIMIZATION_GUIDE.md)

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **QCMD-ECS Framework**: Quantum Chemical Molecular Design with Equivariant Consistency Sampling
- **SchNetPack**: Deep learning for quantum chemistry
- **Next.js & Vercel**: Modern web framework
- **FastAPI**: High-performance Python API framework

---

## Support

- **Issues**: [GitHub Issues](https://github.com/Koussaisalem/algo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Koussaisalem/algo/discussions)

---

**Made with for quantum chemistry research**
