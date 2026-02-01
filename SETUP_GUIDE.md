# QuantumLab Platform - Quick Setup Guide

## One-Command Installation

```bash
git clone https://github.com/Koussaisalem/algo.git && cd algo/quantumlab && chmod +x setup.sh && ./setup.sh && ./start_all.sh
```

That's it! Open http://localhost:3000 in your browser.

---

## What's Included?

### Automated Setup Script (`setup.sh`)
- Checks all dependencies (Node.js, Python, PostgreSQL)
- Generates secure random secrets (NEXTAUTH_SECRET, POSTGRES_PASSWORD, VAULT_MASTER_KEY)
- Creates `.env.local` configuration file
- Sets up PostgreSQL database and user
- Installs frontend dependencies (npm)
- Installs backend dependencies (Python venv)
- Initializes database schema (users, sessions tables)
- Creates convenient startup scripts

### Docker Compose Setup
```bash
git clone https://github.com/Koussaisalem/algo.git
cd algo/quantumlab
cp .env.example .env.local
# Edit .env.local with your secrets
docker-compose up -d
```

Services included:
- PostgreSQL 16 (port 5432)
- Backend API (port 8000)
- Frontend UI (port 3000)

---

## Manual Setup (If Needed)

### Prerequisites
- Node.js 20+
- Python 3.12+
- PostgreSQL 16+

### Step-by-Step

1. **Clone & Navigate**
```bash
git clone https://github.com/Koussaisalem/algo.git
cd algo/quantumlab
```

2. **Setup Environment**
```bash
cp .env.example .env.local
# Edit .env.local with your configuration
```

3. **Install Dependencies**
```bash
# Frontend
npm install --legacy-peer-deps

# Backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..
```

4. **Setup Database**
```bash
sudo service postgresql start
sudo -u postgres psql -c "CREATE DATABASE quantumlab;"
sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD 'your_password';"
```

5. **Start Services**
```bash
# Terminal 1: Backend
cd backend && source venv/bin/activate && python inference_server.py

# Terminal 2: Frontend
npm run dev
```

---

## Features

### Molecular Generation
- Stiefel manifold diffusion
- Property-guided design (band gap targeting)
- Real-time inference with trained models
- 3D visualization

### Molecule Library
- Save/search/export molecules
- SQLite database with full-text search
- Batch operations
- Favorite molecules

### Cloud Training
- VM management
- SSH terminal (web-based)
- Secure credential vault (Fernet encryption)
- Training session tracking

### System Monitoring
- Auto-detect runtime environment
- Real-time GPU/CPU/RAM monitoring
- Storage statistics
- Smart recommendations

### Authentication
- NextAuth.js with PostgreSQL
- Email/password + OAuth ready
- Protected routes
- Session management

---

## Troubleshooting

### Port Already in Use
```bash
lsof -ti:3000 | xargs kill -9 # Frontend
lsof -ti:8000 | xargs kill -9 # Backend
```

### PostgreSQL Issues
```bash
sudo service postgresql restart
psql -U postgres -d quantumlab -c "SELECT 1;"
```

### Dependencies
```bash
# Frontend
rm -rf node_modules .next && npm install --legacy-peer-deps

# Backend
cd backend && source venv/bin/activate && pip install --upgrade -r requirements.txt
```

---

## Documentation

- **Quick Start**: [QUICKSTART.md](quantumlab/QUICKSTART.md)
- **Full README**: [quantumlab/README.md](quantumlab/README.md)
- **Architecture**: [docs/architecture/OVERVIEW.md](docs/architecture/OVERVIEW.md)
- **Theory**: [docs/theory/STIEFEL_MANIFOLD_THEORY.md](docs/theory/STIEFEL_MANIFOLD_THEORY.md)

---

## Support

- Issues: https://github.com/Koussaisalem/algo/issues
- Discussions: https://github.com/Koussaisalem/algo/discussions

**Happy molecule designing! **
